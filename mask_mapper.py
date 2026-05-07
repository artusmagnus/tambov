#!/usr/bin/env python3
"""Interactive wet/dry floor mask mapper for video files.

The tool opens a video, lets you navigate to frames where the floor state is
clear, and draw polygon masks for:
  1. the full floor area,
  2. dry floor regions,
  3. wet floor regions.

Masks are stored in video resolution as binary PNG files plus a combined label
map that can be used for later model training or rule-based experiments.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    import cv2
    import numpy as np


MASK_CLASSES = {
    "floor": {"id": 1, "color": (255, 180, 0), "key": "f"},
    "dry": {"id": 2, "color": (0, 220, 0), "key": "d"},
    "wet": {"id": 3, "color": (255, 0, 255), "key": "w"},
}

HELP_TEXT = """
Wet/dry floor mask mapper controls
----------------------------------
Mouse:
  Left click            add polygon point
  Right click           remove last polygon point
  Double left click     close and apply polygon to the selected mask

Mask selection:
  f                     select floor-area mask
  d                     select dry-floor mask
  w                     select wet-floor mask

Drawing/editing:
  c / Enter             close and apply current polygon
  x                     clear current unfinished polygon
  r                     reset selected mask
  u                     undo last applied polygon or reset
  i                     clip dry/wet masks to the floor-area mask

Video navigation:
  n / Right arrow       next frame
  p / Left arrow        previous frame
  ]                     jump forward 30 frames
  [                     jump backward 30 frames
  g                     jump to a frame number entered in the terminal

Output:
  s                     save masks and metadata
  h                     print this help in the terminal
  q / Esc               quit
""".strip()


@dataclass
class EditorState:
    video_path: Path
    out_dir: Path
    frame_count: int
    width: int
    height: int
    fps: float
    scale: float
    alpha: float
    frame_index: int = 0
    selected: str = "floor"
    points: list[tuple[int, int]] = field(default_factory=list)
    masks: dict[str, np.ndarray] = field(default_factory=dict)
    history: list[tuple[str, np.ndarray]] = field(default_factory=list)
    dirty: bool = False

    @property
    def display_size(self) -> tuple[int, int]:
        return (int(self.width * self.scale), int(self.height * self.scale))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open a video and interactively map floor, dry-floor, and wet-floor masks."
    )
    parser.add_argument("video", type=Path, help="Path to the input video file.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("mask_output"),
        help="Directory where mask PNGs and metadata JSON will be saved.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Display scale for large videos. Coordinates are saved at original resolution.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Overlay opacity for existing masks, from 0.0 to 1.0.",
    )
    return parser.parse_args()


def clamp_frame(index: int, frame_count: int) -> int:
    if frame_count <= 0:
        return max(index, 0)
    return min(max(index, 0), frame_count - 1)


def read_frame(capture: cv2.VideoCapture, index: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {index} from the video.")
    return frame


def scaled_point(x: int, y: int, scale: float, width: int, height: int) -> tuple[int, int]:
    original_x = int(round(x / scale))
    original_y = int(round(y / scale))
    return min(max(original_x, 0), width - 1), min(max(original_y, 0), height - 1)


def display_point(point: tuple[int, int], scale: float) -> tuple[int, int]:
    return int(round(point[0] * scale)), int(round(point[1] * scale))


def push_history(state: EditorState, label: str) -> None:
    state.history.append((label, state.masks[label].copy()))
    if len(state.history) > 100:
        state.history.pop(0)


def apply_polygon(state: EditorState) -> None:
    if len(state.points) < 3:
        print("Need at least 3 points before a polygon can be applied.")
        return

    push_history(state, state.selected)
    polygon = np.array(state.points, dtype=np.int32)
    cv2.fillPoly(state.masks[state.selected], [polygon], 255)
    print(f"Applied polygon with {len(state.points)} points to '{state.selected}' mask.")
    state.points.clear()
    state.dirty = True


def reset_selected_mask(state: EditorState) -> None:
    push_history(state, state.selected)
    state.masks[state.selected][:] = 0
    state.points.clear()
    state.dirty = True
    print(f"Reset '{state.selected}' mask.")


def undo(state: EditorState) -> None:
    if not state.history:
        print("Nothing to undo.")
        return
    label, previous = state.history.pop()
    state.masks[label] = previous
    state.dirty = True
    print(f"Undid last change to '{label}' mask.")


def clip_to_floor(state: EditorState) -> None:
    floor = state.masks["floor"]
    for label in ("dry", "wet"):
        push_history(state, label)
        state.masks[label] = cv2.bitwise_and(state.masks[label], floor)
    state.dirty = True
    print("Clipped dry and wet masks to the floor-area mask.")


def set_frame(state: EditorState, capture: cv2.VideoCapture, index: int) -> np.ndarray:
    state.frame_index = clamp_frame(index, state.frame_count)
    state.points.clear()
    return read_frame(capture, state.frame_index)


def save_outputs(state: EditorState) -> None:
    state.out_dir.mkdir(parents=True, exist_ok=True)
    stem = state.video_path.stem

    saved_masks: dict[str, str] = {}
    for label, mask in state.masks.items():
        path = state.out_dir / f"{stem}_{label}_mask.png"
        cv2.imwrite(str(path), mask)
        saved_masks[label] = str(path)

    combined = np.zeros((state.height, state.width), dtype=np.uint8)
    combined[state.masks["floor"] > 0] = MASK_CLASSES["floor"]["id"]
    combined[state.masks["dry"] > 0] = MASK_CLASSES["dry"]["id"]
    combined[state.masks["wet"] > 0] = MASK_CLASSES["wet"]["id"]
    combined_path = state.out_dir / f"{stem}_combined_labels.png"
    cv2.imwrite(str(combined_path), combined)

    metadata = {
        "video": str(state.video_path),
        "video_width": state.width,
        "video_height": state.height,
        "fps": state.fps,
        "frame_count": state.frame_count,
        "last_reference_frame": state.frame_index,
        "classes": {label: {"id": data["id"]} for label, data in MASK_CLASSES.items()},
        "binary_masks": saved_masks,
        "combined_label_map": str(combined_path),
        "notes": "In the combined label map, wet/dry labels overwrite floor where they overlap.",
    }
    metadata_path = state.out_dir / f"{stem}_mask_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    state.dirty = False
    print(f"Saved masks to: {state.out_dir.resolve()}")


def make_overlay(state: EditorState, frame: np.ndarray) -> np.ndarray:
    overlay = frame.copy()
    color_layer = np.zeros_like(frame)

    for label, config in MASK_CLASSES.items():
        mask = state.masks[label] > 0
        color_layer[mask] = config["color"]

    overlay = cv2.addWeighted(color_layer, state.alpha, overlay, 1.0 - state.alpha, 0)
    untouched = np.all(color_layer == 0, axis=2)
    overlay[untouched] = frame[untouched]

    if state.scale != 1.0:
        overlay = cv2.resize(overlay, state.display_size, interpolation=cv2.INTER_AREA)

    if state.points:
        display_points = [display_point(point, state.scale) for point in state.points]
        for point in display_points:
            cv2.circle(overlay, point, 4, MASK_CLASSES[state.selected]["color"], -1)
        for start, end in zip(display_points, display_points[1:]):
            cv2.line(overlay, start, end, MASK_CLASSES[state.selected]["color"], 2)

    status = (
        f"Frame {state.frame_index + 1}/{max(state.frame_count, 1)} | "
        f"selected: {state.selected} | points: {len(state.points)} | "
        "h=help s=save q=quit"
    )
    cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(overlay, status, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return overlay


def make_mouse_callback(state: EditorState) -> Callable[[int, int, int, int, object], None]:
    def on_mouse(event: int, x: int, y: int, flags: int, userdata: object) -> None:
        del flags, userdata
        if event == cv2.EVENT_LBUTTONDOWN:
            state.points.append(scaled_point(x, y, state.scale, state.width, state.height))
        elif event == cv2.EVENT_RBUTTONDOWN:
            if state.points:
                removed = state.points.pop()
                print(f"Removed point {removed}.")
        elif event == cv2.EVENT_LBUTTONDBLCLK:
            apply_polygon(state)

    return on_mouse


def prompt_for_frame(state: EditorState) -> int | None:
    raw_value = input(f"Enter frame number, 1-{max(state.frame_count, 1)}: ").strip()
    if not raw_value:
        return None
    try:
        return int(raw_value) - 1
    except ValueError:
        print(f"'{raw_value}' is not a valid frame number.")
        return None


def main() -> int:
    args = parse_args()

    global cv2, np
    import cv2 as cv2_module
    import numpy as np_module

    cv2 = cv2_module
    np = np_module
    if args.scale <= 0:
        raise ValueError("--scale must be greater than 0.")
    if not 0 <= args.alpha <= 1:
        raise ValueError("--alpha must be between 0.0 and 1.0.")

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))

    state = EditorState(
        video_path=args.video,
        out_dir=args.out_dir,
        frame_count=frame_count,
        width=width,
        height=height,
        fps=fps,
        scale=args.scale,
        alpha=args.alpha,
        masks={label: np.zeros((height, width), dtype=np.uint8) for label in MASK_CLASSES},
    )

    print(HELP_TEXT)
    frame = read_frame(capture, state.frame_index)
    window_name = "wet/dry floor mask mapper"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, make_mouse_callback(state))

    while True:
        cv2.imshow(window_name, make_overlay(state, frame))
        key = cv2.waitKey(20) & 0xFF
        if key == 255:
            continue

        if key in (ord("q"), 27):
            if state.dirty:
                print("Unsaved changes exist. Press 's' before quitting if you want to keep them.")
            break
        if key == ord("h"):
            print(HELP_TEXT)
        elif key in (ord("f"), ord("d"), ord("w")):
            for label, config in MASK_CLASSES.items():
                if key == ord(config["key"]):
                    state.selected = label
                    state.points.clear()
                    print(f"Selected '{label}' mask.")
                    break
        elif key in (ord("c"), 13):
            apply_polygon(state)
        elif key == ord("x"):
            state.points.clear()
            print("Cleared unfinished polygon.")
        elif key == ord("r"):
            reset_selected_mask(state)
        elif key == ord("u"):
            undo(state)
        elif key == ord("i"):
            clip_to_floor(state)
        elif key == ord("s"):
            save_outputs(state)
        elif key in (ord("n"), 83):
            frame = set_frame(state, capture, state.frame_index + 1)
        elif key in (ord("p"), 81):
            frame = set_frame(state, capture, state.frame_index - 1)
        elif key == ord("]"):
            frame = set_frame(state, capture, state.frame_index + 30)
        elif key == ord("["):
            frame = set_frame(state, capture, state.frame_index - 30)
        elif key == ord("g"):
            target = prompt_for_frame(state)
            if target is not None:
                frame = set_frame(state, capture, target)

    capture.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
