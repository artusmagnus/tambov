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
import re
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
  + / =                 increase brush size, or hex cell size when hex view is on
  - / _                 decrease brush size, or hex cell size when hex view is on

Mask selection:
  1 / f                 select floor-area mask
  2                     select dry-floor mask
  3 / w                 select wet-floor mask
  4 / o                 select obstruction mask

Hex visualization:
  Space                 toggle averaged-color hexagon view
  Enter                 run dry-to-wet OKLab hex analysis (0=red, 50=yellow, 100=green)
                        Hexes too far from dry-wet colour line show checkers
  1 / 2 / 3 / 4         show hexagons only for the selected mask layer

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
  s                     save mask images
  h                     print this help in the terminal
  q / Esc               quit
""".strip()


@dataclass
class MaskSnapshot:
    label: str
    frame_index: int | None
    mask: np.ndarray


@dataclass
class HexWetnessModel:
    dry_oklab: np.ndarray
    wet_oklab: np.ndarray


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
    analysis_max_distance: float
    frame_index: int = 0
    selected: str = "floor"
    brush_mode: str = "draw"
    brush_size: int = 20
    hex_enabled: bool = False
    hex_cell_size: int = 40
    mask_revision: int = 0
    analysis_enabled: bool = False
    analysis_revision: int = 0
    wetness_models: dict[int, HexWetnessModel] = field(default_factory=dict)
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
    hex_overlay: np.ndarray | None = None
    hex_mask: np.ndarray | None = None
    hex_cache_key: tuple[int, str, int, int, int, int] | None = None
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
        help="Directory where mask PNGs will be saved.",
    )
    parser.add_argument(
        "--load-dir",
        type=Path,
        help="Reconstruct previously saved masks by scanning this directory for mask PNGs.",
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
        "--analysis-max-distance",
        type=float,
        default=0.08,
        help=(
            "Maximum OKLab perpendicular distance from a dry-to-wet colour line "
            "before a hex is treated as an unrelated colour change."
        ),
    )
    parser.add_argument(
        "--hex-size",
        type=int,
        default=40,
        help="Hex cell radius in original video pixels for analysis and batch video processing.",
    )
    parser.add_argument(
        "--process-video",
        action="store_true",
        help="Process the entire video with the analysis hex overlay using masks loaded from --load-dir.",
    )
    parser.add_argument(
        "--output-video",
        type=Path,
        help="Output video path for --process-video. Defaults to <out-dir>/<video>_hex_overlay.mp4.",
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
    view.hex_overlay = None
    view.hex_mask = None
    view.hex_cache_key = None
    state.render_dirty = True


def make_display_mask(mask: np.ndarray, state: EditorState) -> np.ndarray:
    if state.scale == 1.0:
        return mask.copy()
    return cv2.resize(mask, state.display_size, interpolation=cv2.INTER_NEAREST)


def refresh_display_mask(state: EditorState, label: str) -> None:
    state.display_masks[label] = make_display_mask(state.masks[label], state)
    state.masks_dirty = True
    state.render_dirty = True


def invalidate_masks(state: EditorState) -> None:
    state.mask_revision += 1
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
    invalidate_masks(state)


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
    invalidate_masks(state)
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
    invalidate_masks(state)
    print(f"Undid last change to {', '.join(restored_labels)} mask(s).")


def clip_to_floor(state: EditorState) -> None:
    floor = state.masks["floor"]
    push_history(state, all_condition_history_entries(state))
    for frame_masks in state.condition_masks_by_frame.values():
        for label in CONDITION_LABELS:
            frame_masks[label] = cv2.bitwise_and(frame_masks[label], floor)
    bind_condition_masks_to_frame(state, state.current_condition_frame)
    state.dirty = True
    invalidate_masks(state)
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

    state.dirty = False
    print(f"Saved floor mask and {len(frame_outputs)} frame-specific annotation set(s) to: {state.out_dir.resolve()}")


def load_binary_mask(path: Path, width: int, height: int) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Could not load mask image: {path}")
    if mask.shape != (height, width):
        raise RuntimeError(
            f"Mask {path} has shape {mask.shape}, expected {(height, width)} for this video."
        )
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def find_floor_mask(load_dir: Path, video_stem: str) -> Path:
    preferred_path = load_dir / f"{video_stem}_floor_mask.png"
    if preferred_path.exists():
        return preferred_path

    floor_paths = sorted(load_dir.glob("*_floor_mask.png"))
    if len(floor_paths) == 1:
        return floor_paths[0]
    if not floor_paths:
        raise RuntimeError(f"No floor mask image found in {load_dir}; expected {preferred_path.name}.")
    raise RuntimeError(
        f"Multiple floor mask images found in {load_dir}; expected the video-specific file {preferred_path.name}."
    )


def parse_condition_mask_filename(path: Path, video_stem: str) -> tuple[int, str] | None:
    escaped_stem = re.escape(video_stem)
    pattern = rf"^{escaped_stem}_frame_(\d+)_({'|'.join(CONDITION_LABELS)})_mask\.png$"
    match = re.match(pattern, path.name)
    if not match:
        return None

    frame_number = int(match.group(1))
    label = match.group(2)
    return frame_number - 1, label


def load_outputs(state: EditorState, load_dir: Path) -> None:
    if not load_dir.is_dir():
        raise RuntimeError(f"Mask load path must be a directory: {load_dir}")

    video_stem = state.video_path.stem
    state.masks["floor"] = load_binary_mask(find_floor_mask(load_dir, video_stem), state.width, state.height)
    state.condition_masks_by_frame.clear()

    loaded_frames: set[int] = set()
    for mask_path in sorted(load_dir.glob("*_frame_*_mask.png")):
        parsed = parse_condition_mask_filename(mask_path, video_stem)
        if parsed is None:
            continue

        frame_index, label = parsed
        if frame_index < 0 or (state.frame_count > 0 and frame_index >= state.frame_count):
            print(f"Skipping frame {frame_index + 1} mask because it is outside this video: {mask_path}")
            continue

        frame_masks = state.condition_masks_by_frame.setdefault(frame_index, make_empty_condition_masks(state))
        frame_masks[label] = load_binary_mask(mask_path, state.width, state.height)
        loaded_frames.add(frame_index)

    ensure_condition_masks(state, state.frame_index)
    bind_condition_masks_to_frame(state, state.frame_index)
    refresh_display_mask(state, "floor")
    state.history.clear()
    state.dirty = False
    invalidate_masks(state)
    print(f"Loaded floor mask and {len(loaded_frames)} frame-specific annotation set(s) from: {load_dir}")


def srgb_to_linear(value: np.ndarray) -> np.ndarray:
    return np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)


def bgr_to_oklab(bgr_color: np.ndarray) -> np.ndarray:
    rgb = np.array([bgr_color[2], bgr_color[1], bgr_color[0]], dtype=np.float64) / 255.0
    red, green, blue = srgb_to_linear(rgb)
    lightness = 0.4122214708 * red + 0.5363325363 * green + 0.0514459929 * blue
    medium = 0.2119034982 * red + 0.6806995451 * green + 0.1073969566 * blue
    short = 0.0883024619 * red + 0.2817188376 * green + 0.6299787005 * blue
    l_root, m_root, s_root = np.cbrt([lightness, medium, short])
    return np.array(
        [
            0.2104542553 * l_root + 0.7936177850 * m_root - 0.0040720468 * s_root,
            1.9779984951 * l_root - 2.4285922050 * m_root + 0.4505937099 * s_root,
            0.0259040371 * l_root + 0.7827717662 * m_root - 0.8086757660 * s_root,
        ],
        dtype=np.float64,
    )


def wetness_to_bgr(value: float) -> tuple[int, int, int]:
    normalized = min(max(value, 0.0), 100.0) / 100.0
    if normalized <= 0.5:
        green = int(round(255 * normalized * 2.0))
        red = 255
    else:
        green = 255
        red = int(round(255 * (1.0 - normalized) * 2.0))
    return (0, green, red)


def average_bgr_in_polygon(frame: np.ndarray, polygon: np.ndarray, bounds: tuple[int, int, int, int]) -> np.ndarray | None:
    x0, y0, x1, y1 = bounds
    local_polygon = polygon - np.array([x0, y0], dtype=np.int32)
    cell_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    cv2.fillPoly(cell_mask, [local_polygon], 255)
    cell_pixels = cell_mask > 0
    if not np.any(cell_pixels):
        return None
    return frame[y0:y1, x0:x1][cell_pixels].mean(axis=0)


def polygon_overlaps_mask(mask: np.ndarray, polygon: np.ndarray, bounds: tuple[int, int, int, int]) -> bool:
    x0, y0, x1, y1 = bounds
    local_polygon = polygon - np.array([x0, y0], dtype=np.int32)
    cell_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    cv2.fillPoly(cell_mask, [local_polygon], 255)
    return bool(np.any((cell_mask > 0) & (mask[y0:y1, x0:x1] > 0)))


def hexagon_points(center_x: float, center_y: float, radius: float) -> np.ndarray:
    points = []
    for angle_degrees in range(0, 360, 60):
        angle = np.deg2rad(angle_degrees)
        points.append((center_x + radius * np.cos(angle), center_y + radius * np.sin(angle)))
    return np.round(points).astype(np.int32)


def iter_hexagons(width: int, height: int, radius: int):
    hex_height = np.sqrt(3) * radius
    horizontal_step = 1.5 * radius
    col = 0
    center_x = radius
    while center_x < width + radius:
        row_offset = (hex_height / 2) if col % 2 else 0
        center_y = row_offset + radius
        while center_y < height + radius:
            yield hexagon_points(center_x, center_y, radius)
            center_y += hex_height
        col += 1
        center_x = radius + col * horizontal_step


def polygon_bounds(polygon: np.ndarray, width: int, height: int) -> tuple[int, int, int, int] | None:
    x0 = max(0, int(np.min(polygon[:, 0])))
    y0 = max(0, int(np.min(polygon[:, 1])))
    x1 = min(width, int(np.max(polygon[:, 0])) + 1)
    y1 = min(height, int(np.max(polygon[:, 1])) + 1)
    if x0 >= x1 or y0 >= y1:
        return None
    return x0, y0, x1, y1


def collect_hex_color_samples(
    state: EditorState,
    capture: cv2.VideoCapture,
) -> dict[int, dict[str, list[np.ndarray]]]:
    samples: dict[int, dict[str, list[np.ndarray]]] = {}
    polygons = list(iter_hexagons(state.width, state.height, max(1, state.hex_cell_size)))
    for frame_index, frame_masks in sorted(state.condition_masks_by_frame.items()):
        if not (np.any(frame_masks["dry"] > 0) or np.any(frame_masks["wet"] > 0)):
            continue
        frame = read_frame(capture, frame_index)
        for cell_index, polygon in enumerate(polygons):
            bounds = polygon_bounds(polygon, state.width, state.height)
            if bounds is None:
                continue
            average_color = average_bgr_in_polygon(frame, polygon, bounds)
            if average_color is None:
                continue
            cell_samples = samples.setdefault(cell_index, {"dry": [], "wet": []})
            if polygon_overlaps_mask(frame_masks["dry"], polygon, bounds):
                cell_samples["dry"].append(bgr_to_oklab(average_color))
            if polygon_overlaps_mask(frame_masks["wet"], polygon, bounds):
                cell_samples["wet"].append(bgr_to_oklab(average_color))
    return samples


def find_nearest_model_by_state_color(
    known_oklab: np.ndarray,
    models: dict[int, HexWetnessModel],
    known_label: str,
) -> HexWetnessModel | None:
    nearest_model: HexWetnessModel | None = None
    nearest_distance = float("inf")
    for model in models.values():
        model_oklab = model.dry_oklab if known_label == "dry" else model.wet_oklab
        distance = float(np.linalg.norm(known_oklab - model_oklab))
        if distance < nearest_distance:
            nearest_model = model
            nearest_distance = distance
    return nearest_model


def infer_single_state_wetness_models(
    samples: dict[int, dict[str, list[np.ndarray]]],
    complete_models: dict[int, HexWetnessModel],
) -> dict[int, HexWetnessModel]:
    inferred_models: dict[int, HexWetnessModel] = {}
    if not complete_models:
        return inferred_models

    for cell_index, cell_samples in samples.items():
        has_dry = bool(cell_samples["dry"])
        has_wet = bool(cell_samples["wet"])
        if has_dry == has_wet:
            continue

        if has_dry:
            dry_oklab = np.mean(cell_samples["dry"], axis=0)
            nearest_model = find_nearest_model_by_state_color(dry_oklab, complete_models, "dry")
            if nearest_model is not None:
                wet_oklab = nearest_model.wet_oklab.copy()
                if float(np.dot(wet_oklab - dry_oklab, wet_oklab - dry_oklab)) > 1e-12:
                    inferred_models[cell_index] = HexWetnessModel(dry_oklab=dry_oklab, wet_oklab=wet_oklab)
        else:
            wet_oklab = np.mean(cell_samples["wet"], axis=0)
            nearest_model = find_nearest_model_by_state_color(wet_oklab, complete_models, "wet")
            if nearest_model is not None:
                dry_oklab = nearest_model.dry_oklab.copy()
                if float(np.dot(wet_oklab - dry_oklab, wet_oklab - dry_oklab)) > 1e-12:
                    inferred_models[cell_index] = HexWetnessModel(dry_oklab=dry_oklab, wet_oklab=wet_oklab)
    return inferred_models


def run_wetness_analysis(state: EditorState, capture: cv2.VideoCapture) -> None:
    samples = collect_hex_color_samples(state, capture)
    complete_models: dict[int, HexWetnessModel] = {}
    for cell_index, cell_samples in samples.items():
        if not cell_samples["dry"] or not cell_samples["wet"]:
            continue
        dry_oklab = np.mean(cell_samples["dry"], axis=0)
        wet_oklab = np.mean(cell_samples["wet"], axis=0)
        if float(np.dot(wet_oklab - dry_oklab, wet_oklab - dry_oklab)) <= 1e-12:
            continue
        complete_models[cell_index] = HexWetnessModel(dry_oklab=dry_oklab, wet_oklab=wet_oklab)

    inferred_models = infer_single_state_wetness_models(samples, complete_models)
    models = {**complete_models, **inferred_models}

    state.wetness_models = models
    state.analysis_enabled = bool(models)
    state.hex_enabled = True
    state.analysis_revision += 1
    state.render_dirty = True
    if models:
        print(
            f"Wetness analysis ready for {len(models)} hex cell(s) "
            f"({len(complete_models)} directly paired, {len(inferred_models)} inferred from one-state samples). "
            "Navigate frames to view floor wetness estimates."
        )
    else:
        print("Wetness analysis found no usable hex cells; at least one hex needs both dry and wet examples.")


def project_onto_wetness_axis(model: HexWetnessModel, bgr_color: np.ndarray) -> tuple[float, float]:
    current_oklab = bgr_to_oklab(bgr_color)
    axis = model.wet_oklab - model.dry_oklab
    denominator = float(np.dot(axis, axis))
    if denominator <= 1e-12:
        return 0.0, float("inf")

    raw_position = float(np.dot(current_oklab - model.dry_oklab, axis) / denominator)
    projected_oklab = model.dry_oklab + raw_position * axis
    distance = float(np.linalg.norm(current_oklab - projected_oklab))
    wetness = min(max(raw_position, 0.0), 1.0) * 100.0
    return wetness, distance


def estimate_wetness_value(model: HexWetnessModel, bgr_color: np.ndarray) -> float:
    wetness, _distance = project_onto_wetness_axis(model, bgr_color)
    return wetness


def analysis_visibility_mask(state: EditorState) -> np.ndarray:
    visibility_mask = state.masks["floor"].copy()
    visibility_mask[state.masks["obstruction"] > 0] = 0
    return visibility_mask


def fill_missing_hex_texture(hex_overlay: np.ndarray, display_polygon: np.ndarray, square_size: int) -> None:
    bounds = polygon_bounds(display_polygon, hex_overlay.shape[1], hex_overlay.shape[0])
    if bounds is None:
        return
    x0, y0, x1, y1 = bounds
    local_polygon = display_polygon - np.array([x0, y0], dtype=np.int32)
    polygon_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    cv2.fillPoly(polygon_mask, [local_polygon], 255)

    rows, cols = np.indices(polygon_mask.shape)
    checker = ((rows // square_size) + (cols // square_size)) % 2 == 0
    patch = hex_overlay[y0:y1, x0:x1]
    patch[(polygon_mask > 0) & checker] = (255, 0, 255)
    patch[(polygon_mask > 0) & ~checker] = (0, 0, 0)


def make_hex_overlay(state: EditorState, view: FrameView) -> np.ndarray:
    layer_key = "analysis_floor" if state.analysis_enabled else state.selected
    cache_key = (
        state.frame_index,
        layer_key,
        state.hex_cell_size,
        state.mask_revision,
        state.analysis_revision if state.analysis_enabled else 0,
        id(view.frame),
    )
    if view.hex_overlay is not None and view.hex_cache_key == cache_key:
        return view.hex_overlay

    hex_overlay = np.zeros_like(view.display_frame)
    hex_mask = np.zeros(view.display_frame.shape[:2], dtype=np.uint8)
    source_mask = analysis_visibility_mask(state) if state.analysis_enabled else state.masks[state.selected]
    radius = max(1, state.hex_cell_size)

    for cell_index, polygon in enumerate(iter_hexagons(state.width, state.height, radius)):
        bounds = polygon_bounds(polygon, state.width, state.height)
        if bounds is None:
            continue
        if not polygon_overlaps_mask(source_mask, polygon, bounds):
            continue

        average_color = average_bgr_in_polygon(view.frame, polygon, bounds)
        if average_color is None:
            continue

        display_polygon = np.round(polygon * state.scale).astype(np.int32)
        wetness_text: str | None = None
        if state.analysis_enabled:
            if cell_index not in state.wetness_models:
                square_size = max(4, int(round(state.hex_cell_size * state.scale / 4)))
                fill_missing_hex_texture(hex_overlay, display_polygon, square_size)
                cv2.fillPoly(hex_mask, [display_polygon], 255)
                cv2.polylines(hex_overlay, [display_polygon], True, (30, 30, 30), 1, cv2.LINE_AA)
                continue
            wetness_value, distance = project_onto_wetness_axis(state.wetness_models[cell_index], average_color)
            if distance > state.analysis_max_distance:
                square_size = max(4, int(round(state.hex_cell_size * state.scale / 4)))
                fill_missing_hex_texture(hex_overlay, display_polygon, square_size)
                cv2.fillPoly(hex_mask, [display_polygon], 255)
                cv2.polylines(hex_overlay, [display_polygon], True, (30, 30, 30), 1, cv2.LINE_AA)
                continue
            color = wetness_to_bgr(wetness_value)
            if state.hex_cell_size * state.scale >= 18:
                wetness_text = f"{wetness_value:.0f}"
        else:
            color = tuple(int(channel) for channel in average_color)
        cv2.fillPoly(hex_overlay, [display_polygon], color)
        cv2.fillPoly(hex_mask, [display_polygon], 255)
        cv2.polylines(hex_overlay, [display_polygon], True, (30, 30, 30), 1, cv2.LINE_AA)
        if wetness_text is not None:
            center = tuple(np.round(display_polygon.mean(axis=0)).astype(int))
            cv2.putText(
                hex_overlay,
                wetness_text,
                center,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    view.hex_overlay = hex_overlay
    view.hex_mask = hex_mask
    view.hex_cache_key = cache_key
    return hex_overlay


def make_analysis_video_frame(state: EditorState, frame: np.ndarray) -> np.ndarray:
    view = FrameView(frame=frame, display_frame=frame.copy())
    hex_overlay = make_hex_overlay(state, view)
    output_frame = frame.copy()
    if view.hex_mask is not None:
        hex_pixels = view.hex_mask > 0
        blended_hex = cv2.addWeighted(hex_overlay, 0.2, frame, 0.8, 0)
        output_frame[hex_pixels] = blended_hex[hex_pixels]
    return output_frame


def default_output_video_path(state: EditorState) -> Path:
    return state.out_dir / f"{state.video_path.stem}_hex_overlay.mp4"


def process_video_with_analysis_overlay(
    state: EditorState,
    capture: cv2.VideoCapture,
    output_path: Path | None,
) -> int:
    if not state.condition_masks_by_frame:
        raise RuntimeError("No mask annotations are loaded; pass --load-dir with saved mask PNGs.")

    state.scale = 1.0
    state.hex_enabled = True
    state.analysis_enabled = False
    run_wetness_analysis(state, capture)
    if not state.analysis_enabled:
        raise RuntimeError("Cannot process video because wetness analysis found no dry/wet hex models.")

    output_path = output_path or default_output_video_path(state)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fps = state.fps if state.fps > 0 else 30.0
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (state.width, state.height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video for writing: {output_path}")

    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    processed_frames = 0
    total_frames = max(state.frame_count, 0)
    empty_condition_masks = make_empty_condition_masks(state)
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break

            state.frame_index = processed_frames
            frame_masks = state.condition_masks_by_frame.get(processed_frames, empty_condition_masks)
            state.current_condition_frame = processed_frames
            for label in CONDITION_LABELS:
                state.masks[label] = frame_masks[label]
            output_frame = make_analysis_video_frame(state, frame)
            writer.write(output_frame)
            processed_frames += 1

            if processed_frames % 100 == 0:
                if total_frames:
                    print(f"Processed {processed_frames}/{total_frames} frames...")
                else:
                    print(f"Processed {processed_frames} frames...")
    finally:
        writer.release()

    print(f"Wrote analysis hex overlay video with {processed_frames} frame(s) to: {output_path.resolve()}")
    return 0


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

    overlay = display_frame.copy() if state.hex_enabled else view.base_overlay.copy()
    if state.hex_enabled:
        hex_overlay = make_hex_overlay(state, view)
        if view.hex_mask is not None:
            hex_pixels = view.hex_mask > 0
            blended_hex = cv2.addWeighted(hex_overlay, 0.2, display_frame, 0.8, 0)
            overlay[hex_pixels] = blended_hex[hex_pixels]

    if state.cursor is not None:
        cursor = display_point(state.cursor, state.scale)
        brush_radius = max(1, int(round(state.brush_size * state.scale)))
        color = MASK_CLASSES[state.selected]["color"] if state.brush_mode == "draw" else (0, 0, 255)
        cv2.circle(overlay, cursor, brush_radius, color, 2)

    status = (
        f"Frame {state.frame_index + 1}/{max(state.frame_count, 1)} | "
        f"mask: {state.selected} | mode: {state.brush_mode} | "
        f"brush: {state.brush_size}px | hex: {'analysis' if state.analysis_enabled else ('on' if state.hex_enabled else 'off')} {state.hex_cell_size}px | h=help s=save q=quit"
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


def adjust_hex_cell_size(state: EditorState, delta: int) -> None:
    old_size = state.hex_cell_size
    state.hex_cell_size = min(max(state.hex_cell_size + delta, 5), 500)
    if state.hex_cell_size != old_size and state.analysis_enabled:
        state.analysis_enabled = False
        print("Hex cell size changed; press Enter to rerun wetness analysis.")
    state.render_dirty = True
    print(f"Hex cell size: {state.hex_cell_size}px")


def toggle_hex_view(state: EditorState) -> None:
    state.hex_enabled = not state.hex_enabled
    state.render_dirty = True
    print(f"Hex view: {'on' if state.hex_enabled else 'off'}")


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
    if args.hex_size <= 0:
        raise ValueError("--hex-size must be greater than 0.")
    if args.process_video and not args.load_dir:
        raise ValueError("--process-video requires --load-dir so masks can be reconstructed.")
    if args.analysis_max_distance < 0:
        raise ValueError("--analysis-max-distance must be 0 or greater.")
    if args.max_display_width < 0:
        raise ValueError("--max-display-width must be 0 or greater.")

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))

    if args.process_video:
        args.scale = 1.0
    elif args.max_display_width and width > args.max_display_width:
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
        analysis_max_distance=args.analysis_max_distance,
        brush_size=args.brush_size,
        hex_cell_size=args.hex_size,
        masks={label: np.zeros((height, width), dtype=np.uint8) for label in MASK_CLASSES},
    )
    state.condition_masks_by_frame[state.frame_index] = {
        label: state.masks[label] for label in CONDITION_LABELS
    }
    state.current_condition_frame = state.frame_index
    state.display_masks = {label: make_display_mask(mask, state) for label, mask in state.masks.items()}
    if args.load_dir:
        load_outputs(state, args.load_dir)

    if args.process_video:
        try:
            return process_video_with_analysis_overlay(state, capture, args.output_video)
        finally:
            capture.release()

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
        elif key in (10, 13):
            run_wetness_analysis(state, capture)
        elif key == ord(" "):
            toggle_hex_view(state)
        elif key in (ord("+"), ord("=")):
            if state.hex_enabled:
                adjust_hex_cell_size(state, 5)
            else:
                adjust_brush_size(state, 5)
        elif key in (ord("-"), ord("_")):
            if state.hex_enabled:
                adjust_hex_cell_size(state, -5)
            else:
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
