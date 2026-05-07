#!/usr/bin/env python3
"""Interactive brush-based wet/dry floor mask mapper for video files.

The tool opens a video, lets you select frames with an OpenCV horizontal
trackbar, and paint masks for:
  1. the full floor area,
  2. dry floor regions on individual frames,
  3. wet floor regions on individual frames,
  4. obstruction regions on individual frames.

The floor mask is persistent across the whole video. Dry, wet, and obstruction
masks are frame-specific labels stored at video resolution for the frames where
they are drawn.
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
    "floor": {"id": 1, "color": (255, 180, 0), "key": "1"},
    "dry": {"id": 2, "color": (0, 220, 0), "key": "2"},
    "wet": {"id": 3, "color": (255, 0, 255), "key": "3"},
    "obstruction": {"id": 4, "color": (0, 165, 255), "key": "4"},
}

CONDITION_LABELS = ("dry", "wet", "obstruction")

HELP_TEXT = """
Wet/dry floor brush mask mapper controls
----------------------------------------
Mouse:
  Left mouse drag       paint or erase on the selected mask
                        Dry/wet/obstruction painting is limited to the floor mask
                        Dry/wet/obstruction labels apply only to the selected frame

Brush mode:
  d                     draw/paint mode
  e                     erase mode
  + / =                 increase brush size
  - / _                 decrease brush size

Mask selection:
  1 / f                 select floor-area mask
  2                     select dry-floor mask
  3 / w                 select wet-floor mask
  4 / o                 select obstruction mask

Editing:
  r                     reset selected mask
  u                     undo last brush stroke or reset
  i                     clip all frame dry/wet/obstruction masks to the floor-area mask

Video navigation:
  Horizontal slider     choose the video frame to label against
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
class MaskSnapshot:
    label: str
    frame_index: int | None
    mask: np.ndarray


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
    brush_mode: str = "draw"
    brush_size: int = 20
    is_painting: bool = False
    cursor: tuple[int, int] | None = None
    last_paint_point: tuple[int, int] | None = None
    render_dirty: bool = True
    masks: dict[str, np.ndarray] = field(default_factory=dict)
    display_masks: dict[str, np.ndarray] = field(default_factory=dict)
    condition_masks_by_frame: dict[int, dict[str, np.ndarray]] = field(default_factory=dict)
    current_condition_frame: int = 0
    masks_dirty: bool = True
    history: list[list[MaskSnapshot]] = field(default_factory=list)
    dirty: bool = False

    @property
    def display_size(self) -> tuple[int, int]:
        return (max(1, int(self.width * self.scale)), max(1, int(self.height * self.scale)))


@dataclass
class FrameView:
    frame: np.ndarray
    display_frame: np.ndarray
    base_overlay: np.ndarray | None = None
    overlay: np.ndarray | None = None
    suppress_trackbar_callback: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open a video and interactively map floor, dry-floor, wet-floor, and obstruction masks."
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
    parser.add_argument(
        "--brush-size",
        type=int,
        default=20,
        help="Initial brush radius in original video pixels.",
    )
    parser.add_argument(
        "--max-display-width",
        type=int,
        default=1280,
        help=(
            "Automatically downscale the display window to this width for smoother editing. "
            "Use 0 to disable automatic downscaling."
        ),
    )
    return parser.parse_args()


def clamp_frame(index: int, frame_count: int) -> int:
    if frame_count <= 0:
        return 0
    return min(max(index, 0), frame_count - 1)


def read_frame(capture: cv2.VideoCapture, index: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {index} from the video.")
    return frame



def make_display_frame(frame: np.ndarray, state: EditorState) -> np.ndarray:
    if state.scale == 1.0:
        return frame.copy()
    return cv2.resize(frame, state.display_size, interpolation=cv2.INTER_AREA)


def set_frame_view_frame(view: FrameView, state: EditorState, frame: np.ndarray) -> None:
    view.frame = frame
    view.display_frame = make_display_frame(frame, state)
    view.base_overlay = None
    view.overlay = None
    state.render_dirty = True


def make_display_mask(mask: np.ndarray, state: EditorState) -> np.ndarray:
    if state.scale == 1.0:
        return mask.copy()
    return cv2.resize(mask, state.display_size, interpolation=cv2.INTER_NEAREST)


def refresh_display_mask(state: EditorState, label: str) -> None:
    state.display_masks[label] = make_display_mask(state.masks[label], state)
    state.masks_dirty = True
    state.render_dirty = True



def make_empty_mask(state: EditorState) -> np.ndarray:
    return np.zeros((state.height, state.width), dtype=np.uint8)


def make_empty_condition_masks(state: EditorState) -> dict[str, np.ndarray]:
    return {label: make_empty_mask(state) for label in CONDITION_LABELS}


def ensure_condition_masks(state: EditorState, frame_index: int) -> dict[str, np.ndarray]:
    if frame_index not in state.condition_masks_by_frame:
        state.condition_masks_by_frame[frame_index] = make_empty_condition_masks(state)
    return state.condition_masks_by_frame[frame_index]


def bind_condition_masks_to_frame(state: EditorState, frame_index: int) -> None:
    frame_masks = ensure_condition_masks(state, frame_index)
    state.current_condition_frame = frame_index
    for label in CONDITION_LABELS:
        state.masks[label] = frame_masks[label]
        refresh_display_mask(state, label)


def condition_history_entries(state: EditorState, labels: list[str] | tuple[str, ...], frame_index: int | None = None) -> list[tuple[str, int | None]]:
    if frame_index is None:
        frame_index = state.current_condition_frame
    return [(label, frame_index if label in CONDITION_LABELS else None) for label in labels]


def all_condition_history_entries(state: EditorState) -> list[tuple[str, int | None]]:
    entries: list[tuple[str, int | None]] = []
    for frame_index in sorted(state.condition_masks_by_frame):
        entries.extend((label, frame_index) for label in CONDITION_LABELS)
    return entries


def scaled_point(x: int, y: int, scale: float, width: int, height: int) -> tuple[int, int]:
    original_x = int(round(x / scale))
    original_y = int(round(y / scale))
    return min(max(original_x, 0), width - 1), min(max(original_y, 0), height - 1)


def display_point(point: tuple[int, int], scale: float) -> tuple[int, int]:
    return int(round(point[0] * scale)), int(round(point[1] * scale))


def history_mask(state: EditorState, label: str, frame_index: int | None) -> np.ndarray:
    if label == "floor":
        return state.masks["floor"]
    if frame_index is None:
        frame_index = state.current_condition_frame
    return ensure_condition_masks(state, frame_index)[label]


def push_history(state: EditorState, entries: str | list[str] | tuple[str, ...] | list[tuple[str, int | None]]) -> None:
    normalized_entries: list[tuple[str, int | None]] = []
    if isinstance(entries, str):
        normalized_entries = condition_history_entries(state, [entries])
    else:
        for entry in entries:
            if isinstance(entry, tuple):
                normalized_entries.append(entry)
            else:
                normalized_entries.extend(condition_history_entries(state, [entry]))

    unique_entries = list(dict.fromkeys(normalized_entries))
    snapshot = [
        MaskSnapshot(label=label, frame_index=frame_index, mask=history_mask(state, label, frame_index).copy())
        for label, frame_index in unique_entries
    ]
    state.history.append(snapshot)
    if len(state.history) > 100:
        state.history.pop(0)


def other_condition_labels(label: str) -> tuple[str, ...]:
    return tuple(other_label for other_label in CONDITION_LABELS if other_label != label)


def draw_stroke(stroke_mask: np.ndarray, start: tuple[int, int] | None, end: tuple[int, int], radius: int) -> None:
    if start is None:
        cv2.circle(stroke_mask, end, radius, 255, -1)
    else:
        cv2.line(stroke_mask, start, end, 255, max(1, radius * 2), cv2.LINE_8)


def apply_floor_condition_rules(state: EditorState, stroke_mask: np.ndarray, display_stroke_mask: np.ndarray) -> list[str]:
    selected = state.selected
    changed_labels = [selected]

    if selected == "floor":
        if state.brush_mode == "draw":
            state.masks["floor"][stroke_mask > 0] = 255
            state.display_masks["floor"][display_stroke_mask > 0] = 255
        else:
            state.masks["floor"][stroke_mask > 0] = 0
            state.display_masks["floor"][display_stroke_mask > 0] = 0
            for frame_masks in state.condition_masks_by_frame.values():
                for label in CONDITION_LABELS:
                    frame_masks[label][state.masks["floor"] == 0] = 0
            for label in CONDITION_LABELS:
                refresh_display_mask(state, label)
            changed_labels.extend(CONDITION_LABELS)
        return changed_labels

    if state.brush_mode == "erase":
        state.masks[selected][stroke_mask > 0] = 0
        state.display_masks[selected][display_stroke_mask > 0] = 0
        return changed_labels

    allowed_pixels = (stroke_mask > 0) & (state.masks["floor"] > 0)
    display_allowed_pixels = (display_stroke_mask > 0) & (state.display_masks["floor"] > 0)
    state.masks[selected][allowed_pixels] = 255
    state.display_masks[selected][display_allowed_pixels] = 255
    for other_label in other_condition_labels(selected):
        state.masks[other_label][allowed_pixels] = 0
        state.display_masks[other_label][display_allowed_pixels] = 0
        changed_labels.append(other_label)
    return changed_labels


def paint_at(state: EditorState, point: tuple[int, int]) -> None:
    stroke_mask = np.zeros((state.height, state.width), dtype=np.uint8)
    display_stroke_mask = np.zeros(state.display_size[::-1], dtype=np.uint8)
    display_current = display_point(point, state.scale)
    display_radius = max(1, int(round(state.brush_size * state.scale)))

    draw_stroke(stroke_mask, state.last_paint_point, point, state.brush_size)
    draw_stroke(
        display_stroke_mask,
        display_point(state.last_paint_point, state.scale) if state.last_paint_point else None,
        display_current,
        display_radius,
    )
    apply_floor_condition_rules(state, stroke_mask, display_stroke_mask)
    state.last_paint_point = point
    state.dirty = True
    state.masks_dirty = True
    state.render_dirty = True


def begin_stroke(state: EditorState, point: tuple[int, int]) -> None:
    labels = condition_history_entries(state, [state.selected])
    if state.selected == "floor" and state.brush_mode == "erase":
        labels.extend(all_condition_history_entries(state))
    elif state.selected in CONDITION_LABELS and state.brush_mode == "draw":
        labels.extend(condition_history_entries(state, list(other_condition_labels(state.selected))))
    push_history(state, labels)
    state.is_painting = True
    state.cursor = point
    state.last_paint_point = None
    paint_at(state, point)


def end_stroke(state: EditorState) -> None:
    state.is_painting = False
    state.last_paint_point = None


def reset_selected_mask(state: EditorState) -> None:
    labels = condition_history_entries(state, [state.selected])
    if state.selected == "floor":
        labels.extend(all_condition_history_entries(state))
    push_history(state, labels)
    state.masks[state.selected][:] = 0
    state.display_masks[state.selected][:] = 0
    if state.selected == "floor":
        for frame_masks in state.condition_masks_by_frame.values():
            for label in CONDITION_LABELS:
                frame_masks[label][:] = 0
        for label in CONDITION_LABELS:
            refresh_display_mask(state, label)
    state.dirty = True
    state.masks_dirty = True
    state.render_dirty = True
    print(f"Reset '{state.selected}' mask.")


def undo(state: EditorState) -> None:
    if not state.history:
        print("Nothing to undo.")
        return
    snapshot = state.history.pop()
    restored_labels = []
    for item in snapshot:
        if item.label == "floor":
            state.masks["floor"] = item.mask
            refresh_display_mask(state, "floor")
            restored_labels.append("floor")
        elif item.frame_index is not None:
            frame_masks = ensure_condition_masks(state, item.frame_index)
            frame_masks[item.label] = item.mask
            if item.frame_index == state.current_condition_frame:
                state.masks[item.label] = item.mask
                refresh_display_mask(state, item.label)
            restored_labels.append(f"{item.label}@frame{item.frame_index + 1}")
    state.dirty = True
    state.render_dirty = True
    print(f"Undid last change to {', '.join(restored_labels)} mask(s).")


def clip_to_floor(state: EditorState) -> None:
    floor = state.masks["floor"]
    push_history(state, all_condition_history_entries(state))
    for frame_masks in state.condition_masks_by_frame.values():
        for label in CONDITION_LABELS:
            frame_masks[label] = cv2.bitwise_and(frame_masks[label], floor)
    bind_condition_masks_to_frame(state, state.current_condition_frame)
    state.dirty = True
    state.render_dirty = True
    print("Clipped dry, wet, and obstruction masks on every annotated frame to the floor-area mask.")


def set_frame(
    state: EditorState,
    view: FrameView,
    capture: cv2.VideoCapture,
    window_name: str,
    index: int,
) -> None:
    state.frame_index = clamp_frame(index, state.frame_count)
    bind_condition_masks_to_frame(state, state.frame_index)
    set_frame_view_frame(view, state, read_frame(capture, state.frame_index))
    view.suppress_trackbar_callback = True
    cv2.setTrackbarPos("Frame", window_name, state.frame_index)
    view.suppress_trackbar_callback = False


def has_condition_labels(frame_masks: dict[str, np.ndarray]) -> bool:
    return any(np.any(frame_masks[label] > 0) for label in CONDITION_LABELS)


def save_outputs(state: EditorState) -> None:
    state.out_dir.mkdir(parents=True, exist_ok=True)
    stem = state.video_path.stem

    floor_path = state.out_dir / f"{stem}_floor_mask.png"
    cv2.imwrite(str(floor_path), state.masks["floor"])

    frame_outputs: dict[str, dict[str, str | int]] = {}
    for frame_index, frame_masks in sorted(state.condition_masks_by_frame.items()):
        if not has_condition_labels(frame_masks):
            continue

        frame_number = frame_index + 1
        prefix = f"{stem}_frame_{frame_number:06d}"
        frame_output: dict[str, str | int] = {"frame_index": frame_index}

        used_pixels = np.zeros((state.height, state.width), dtype=bool)
        for label in CONDITION_LABELS:
            clipped_mask = cv2.bitwise_and(frame_masks[label], state.masks["floor"])
            clipped_mask[used_pixels] = 0
            used_pixels |= clipped_mask > 0
            frame_masks[label] = clipped_mask

            mask_path = state.out_dir / f"{prefix}_{label}_mask.png"
            cv2.imwrite(str(mask_path), clipped_mask)
            frame_output[f"{label}_mask"] = str(mask_path)

        if frame_index == state.current_condition_frame:
            bind_condition_masks_to_frame(state, frame_index)

        frame_outputs[str(frame_number)] = frame_output

    metadata = {
        "video": str(state.video_path),
        "video_width": state.width,
        "video_height": state.height,
        "fps": state.fps,
        "frame_count": state.frame_count,
        "last_reference_frame": state.frame_index,
        "classes": {label: {"id": data["id"]} for label, data in MASK_CLASSES.items()},
        "floor_mask": str(floor_path),
        "frame_condition_masks": frame_outputs,
        "notes": (
            "Floor is a persistent region mask. Dry, wet, and obstruction are "
            "frame-specific, mutually exclusive masks clipped to the floor area."
        ),
    }
    metadata_path = state.out_dir / f"{stem}_mask_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    state.dirty = False
    print(f"Saved floor mask and {len(frame_outputs)} frame-specific annotation set(s) to: {state.out_dir.resolve()}")


def make_overlay(state: EditorState, view: FrameView) -> np.ndarray:
    if view.overlay is not None and not state.render_dirty:
        return view.overlay

    display_frame = view.display_frame
    if view.base_overlay is None or state.masks_dirty:
        base_overlay = display_frame.copy()
        color_layer = np.zeros_like(display_frame)

        for label, config in MASK_CLASSES.items():
            color_layer[state.display_masks[label] > 0] = config["color"]

        mask_pixels = np.any(color_layer > 0, axis=2)
        if np.any(mask_pixels):
            blended = cv2.addWeighted(color_layer, state.alpha, display_frame, 1.0 - state.alpha, 0)
            base_overlay[mask_pixels] = blended[mask_pixels]
        view.base_overlay = base_overlay
        state.masks_dirty = False

    overlay = view.base_overlay.copy()
    if state.cursor is not None:
        cursor = display_point(state.cursor, state.scale)
        brush_radius = max(1, int(round(state.brush_size * state.scale)))
        color = MASK_CLASSES[state.selected]["color"] if state.brush_mode == "draw" else (0, 0, 255)
        cv2.circle(overlay, cursor, brush_radius, color, 2)

    status = (
        f"Frame {state.frame_index + 1}/{max(state.frame_count, 1)} | "
        f"mask: {state.selected} | mode: {state.brush_mode} | "
        f"brush: {state.brush_size}px | h=help s=save q=quit"
    )
    cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(overlay, status, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    view.overlay = overlay
    state.render_dirty = False
    return overlay


def make_mouse_callback(state: EditorState) -> Callable[[int, int, int, int, object], None]:
    def on_mouse(event: int, x: int, y: int, flags: int, userdata: object) -> None:
        del userdata
        point = scaled_point(x, y, state.scale, state.width, state.height)
        if state.cursor != point:
            state.cursor = point
            state.render_dirty = True

        if event == cv2.EVENT_LBUTTONDOWN:
            begin_stroke(state, point)
        elif event == cv2.EVENT_MOUSEMOVE and state.is_painting and flags & cv2.EVENT_FLAG_LBUTTON:
            paint_at(state, point)
        elif event == cv2.EVENT_LBUTTONUP:
            if state.is_painting:
                paint_at(state, point)
            end_stroke(state)
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON) == 0:
            end_stroke(state)

    return on_mouse


def make_trackbar_callback(
    state: EditorState,
    view: FrameView,
    capture: cv2.VideoCapture,
) -> Callable[[int], None]:
    def on_frame_slider(position: int) -> None:
        if view.suppress_trackbar_callback:
            return
        state.frame_index = clamp_frame(position, state.frame_count)
        bind_condition_masks_to_frame(state, state.frame_index)
        set_frame_view_frame(view, state, read_frame(capture, state.frame_index))
        state.is_painting = False

    return on_frame_slider


def prompt_for_frame(state: EditorState) -> int | None:
    raw_value = input(f"Enter frame number, 1-{max(state.frame_count, 1)}: ").strip()
    if not raw_value:
        return None
    try:
        return int(raw_value) - 1
    except ValueError:
        print(f"'{raw_value}' is not a valid frame number.")
        return None


def set_selected_mask(state: EditorState, label: str) -> None:
    state.selected = label
    state.is_painting = False
    state.render_dirty = True
    print(f"Selected '{label}' mask.")


def adjust_brush_size(state: EditorState, delta: int) -> None:
    state.brush_size = min(max(state.brush_size + delta, 1), 500)
    state.render_dirty = True
    print(f"Brush size: {state.brush_size}px")


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
    if args.brush_size <= 0:
        raise ValueError("--brush-size must be greater than 0.")
    if args.max_display_width < 0:
        raise ValueError("--max-display-width must be 0 or greater.")

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))

    if args.max_display_width and width > args.max_display_width:
        args.scale = min(args.scale, args.max_display_width / width)
        print(f"Display scale set to {args.scale:.3f} for smoother editing.")

    state = EditorState(
        video_path=args.video,
        out_dir=args.out_dir,
        frame_count=frame_count,
        width=width,
        height=height,
        fps=fps,
        scale=args.scale,
        alpha=args.alpha,
        brush_size=args.brush_size,
        masks={label: np.zeros((height, width), dtype=np.uint8) for label in MASK_CLASSES},
    )
    state.condition_masks_by_frame[state.frame_index] = {
        label: state.masks[label] for label in CONDITION_LABELS
    }
    state.current_condition_frame = state.frame_index
    state.display_masks = {label: make_display_mask(mask, state) for label, mask in state.masks.items()}

    print(HELP_TEXT)
    window_name = "wet/dry floor brush mask mapper"
    first_frame = read_frame(capture, state.frame_index)
    view = FrameView(frame=first_frame, display_frame=make_display_frame(first_frame, state))
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Frame", window_name, 0, max(frame_count - 1, 1), make_trackbar_callback(state, view, capture))
    cv2.setMouseCallback(window_name, make_mouse_callback(state))

    while True:
        cv2.imshow(window_name, make_overlay(state, view))
        key = cv2.waitKey(10) & 0xFF
        if key == 255:
            continue

        if key in (ord("q"), 27):
            if state.dirty:
                print("Unsaved changes exist. Press 's' before quitting if you want to keep them.")
            break
        if key == ord("h"):
            print(HELP_TEXT)
        elif key == ord("d"):
            state.brush_mode = "draw"
            state.render_dirty = True
            print("Brush mode: draw")
        elif key == ord("e"):
            state.brush_mode = "erase"
            state.render_dirty = True
            print("Brush mode: erase")
        elif key in (ord("+"), ord("=")):
            adjust_brush_size(state, 5)
        elif key in (ord("-"), ord("_")):
            adjust_brush_size(state, -5)
        elif key in (ord("1"), ord("f")):
            set_selected_mask(state, "floor")
        elif key == ord("2"):
            set_selected_mask(state, "dry")
        elif key in (ord("3"), ord("w")):
            set_selected_mask(state, "wet")
        elif key in (ord("4"), ord("o")):
            set_selected_mask(state, "obstruction")
        elif key == ord("r"):
            reset_selected_mask(state)
        elif key == ord("u"):
            undo(state)
        elif key == ord("i"):
            clip_to_floor(state)
        elif key == ord("s"):
            save_outputs(state)
        elif key in (ord("n"), 83):
            set_frame(state, view, capture, window_name, state.frame_index + 1)
        elif key in (ord("p"), 81):
            set_frame(state, view, capture, window_name, state.frame_index - 1)
        elif key == ord("]"):
            set_frame(state, view, capture, window_name, state.frame_index + 30)
        elif key == ord("["):
            set_frame(state, view, capture, window_name, state.frame_index - 30)
        elif key == ord("g"):
            target = prompt_for_frame(state)
            if target is not None:
                set_frame(state, view, capture, window_name, target)

    capture.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
