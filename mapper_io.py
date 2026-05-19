from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

def save_outputs(state: EditorState, capture: cv2.VideoCapture | None = None) -> None:
    state.out_dir.mkdir(parents=True, exist_ok=True)
    stem = state.source_stem

    floor_path = state.out_dir / f"{stem}_floor_mask.png"
    cv2.imwrite(str(floor_path), state.masks["floor"])

    frame_outputs: dict[str, dict[str, str | int]] = {}
    for frame_index, frame_masks in sorted(state.condition_masks_by_frame.items()):
        if not has_condition_labels(frame_masks):
            continue

        frame_number = frame_index + 1
        prefix = f"{stem}_frame_{frame_number:06d}"
        frame_output: dict[str, str | int] = {"frame_index": frame_index}

        frame = frame_for_annotation_save(state, capture, frame_index)
        if frame is not None:
            frame_path = state.out_dir / f"{prefix}_image.png"
            cv2.imwrite(str(frame_path), frame)
            frame_output["image"] = str(frame_path)

        used_pixels = np.zeros((state.height, state.width), dtype=bool)
        for label in CONDITION_LABELS:
            clipped_mask = cv2.bitwise_and(frame_masks[label], state.masks["floor"])
            clipped_mask[used_pixels] = 0
            used_pixels |= clipped_mask > 0
            frame_masks[label] = clipped_mask

            mask_path = state.out_dir / f"{prefix}_{label}_mask.png"
            cv2.imwrite(str(mask_path), clipped_mask)
            frame_output[f"{label}_mask"] = str(mask_path)

        numbered_mask_path = state.out_dir / f"{prefix}_labels.png"
        cv2.imwrite(str(numbered_mask_path), make_numbered_annotation_mask(state, frame_masks))
        frame_output["labels"] = str(numbered_mask_path)

        if frame_index == state.current_condition_frame:
            bind_condition_masks_to_frame(state, frame_index)

        frame_outputs[str(frame_number)] = frame_output

    if capture is not None:
        capture.set(cv2.CAP_PROP_POS_FRAMES, state.frame_index)

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


def load_annotation_frame(path: Path, width: int, height: int) -> np.ndarray:
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Could not load annotation frame image: {path}")
    if frame.shape[:2] != (height, width):
        raise RuntimeError(
            f"Annotation frame {path} has shape {frame.shape[:2]}, expected {(height, width)}."
        )
    return frame


def load_numbered_annotation_mask(path: Path, width: int, height: int) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Could not load numbered annotation mask: {path}")
    if mask.shape != (height, width):
        raise RuntimeError(
            f"Numbered annotation mask {path} has shape {mask.shape}, expected {(height, width)}."
        )
    return mask.astype(np.uint8)


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


def parse_annotation_frame_filename(path: Path, video_stem: str) -> int | None:
    escaped_stem = re.escape(video_stem)
    match = re.match(rf"^{escaped_stem}_frame_(\d+)_image\.png$", path.name)
    if not match:
        return None
    return int(match.group(1)) - 1


def parse_numbered_annotation_mask_filename(path: Path, video_stem: str) -> int | None:
    escaped_stem = re.escape(video_stem)
    match = re.match(rf"^{escaped_stem}_frame_(\d+)_labels\.png$", path.name)
    if not match:
        return None
    return int(match.group(1)) - 1


def load_dir_has_saved_annotation_frames(load_dir: Path | None) -> bool:
    return bool(load_dir and load_dir.is_dir() and any(load_dir.glob("*_frame_*_image.png")))


def load_numbered_annotation_into_masks(state: EditorState, path: Path, frame_index: int) -> None:
    numbered_mask = load_numbered_annotation_mask(path, state.width, state.height)
    state.masks["floor"][numbered_mask > 0] = 255
    frame_masks = state.condition_masks_by_frame.setdefault(frame_index, make_empty_condition_masks(state))
    for label in CONDITION_LABELS:
        frame_masks[label] = np.where(numbered_mask == MASK_CLASSES[label]["id"], 255, 0).astype(np.uint8)


def load_outputs(state: EditorState, load_dir: Path) -> None:
    if not load_dir.is_dir():
        raise RuntimeError(f"Mask load path must be a directory: {load_dir}")

    video_stem = state.source_stem
    floor_mask_path = find_floor_mask(load_dir, video_stem)
    annotation_stem = floor_mask_path.name.removesuffix("_floor_mask.png")
    state.masks["floor"] = load_binary_mask(floor_mask_path, state.width, state.height)
    state.condition_masks_by_frame.clear()
    state.annotation_frames_by_index.clear()

    loaded_frames: set[int] = set()
    loaded_annotation_images: set[int] = set()

    for frame_path in sorted(load_dir.glob("*_frame_*_image.png")):
        frame_index = parse_annotation_frame_filename(frame_path, annotation_stem)
        if frame_index is None:
            continue
        if frame_index < 0 or (state.frame_count > 0 and frame_index >= state.frame_count):
            print(f"Skipping frame {frame_index + 1} image because it is outside this video/source: {frame_path}")
            continue
        state.annotation_frames_by_index[frame_index] = load_annotation_frame(frame_path, state.width, state.height)
        loaded_annotation_images.add(frame_index)

    for numbered_mask_path in sorted(load_dir.glob("*_frame_*_labels.png")):
        frame_index = parse_numbered_annotation_mask_filename(numbered_mask_path, annotation_stem)
        if frame_index is None:
            continue
        if frame_index < 0 or (state.frame_count > 0 and frame_index >= state.frame_count):
            print(f"Skipping frame {frame_index + 1} numbered mask because it is outside this video/source: {numbered_mask_path}")
            continue
        load_numbered_annotation_into_masks(state, numbered_mask_path, frame_index)
        loaded_frames.add(frame_index)

    for mask_path in sorted(load_dir.glob("*_frame_*_mask.png")):
        parsed = parse_condition_mask_filename(mask_path, annotation_stem)
        if parsed is None:
            continue

        frame_index, label = parsed
        if frame_index < 0 or (state.frame_count > 0 and frame_index >= state.frame_count):
            print(f"Skipping frame {frame_index + 1} mask because it is outside this video/source: {mask_path}")
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
    print(
        f"Loaded floor mask, {len(loaded_frames)} frame-specific annotation set(s), "
        f"and {len(loaded_annotation_images)} saved source frame image(s) from: {load_dir}"
    )
