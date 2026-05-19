from __future__ import annotations

import cv2
import numpy as np

def configure_opencl(requested: bool) -> bool:
    if not requested:
        return False

    if not hasattr(cv2, "ocl") or not cv2.ocl.haveOpenCL():
        print("OpenCL acceleration requested, but this OpenCV build cannot use OpenCL; falling back to CPU.")
        return False

    cv2.ocl.setUseOpenCL(True)
    if cv2.ocl.useOpenCL():
        device_name = "unknown device"
        if hasattr(cv2.ocl, "Device_getDefault"):
            device_name = cv2.ocl.Device_getDefault().name()
        print(f"OpenCL acceleration enabled for supported OpenCV operations ({device_name}).")
        return True

    print("OpenCL acceleration requested, but OpenCV did not enable it; falling back to CPU.")
    return False


def accelerated_resize(image: np.ndarray, size: tuple[int, int], interpolation: int, state: EditorState) -> np.ndarray:
    if not state.use_opencl:
        return cv2.resize(image, size, interpolation=interpolation)
    return cv2.resize(cv2.UMat(image), size, interpolation=interpolation).get()


def accelerated_add_weighted(
    first: np.ndarray,
    first_weight: float,
    second: np.ndarray,
    second_weight: float,
    gamma: float,
    state: EditorState,
) -> np.ndarray:
    if not state.use_opencl:
        return cv2.addWeighted(first, first_weight, second, second_weight, gamma)
    return cv2.addWeighted(cv2.UMat(first), first_weight, cv2.UMat(second), second_weight, gamma).get()


def make_temporal_accumulator(frame: np.ndarray, state: EditorState) -> object:
    accumulator = np.zeros_like(frame, dtype=np.float32)
    if state.use_opencl:
        return cv2.UMat(accumulator)
    return accumulator


def add_to_temporal_accumulator(accumulator: object, frame: np.ndarray, state: EditorState) -> object:
    if state.use_opencl:
        cv2.accumulate(cv2.UMat(frame), accumulator)
        return accumulator
    accumulator += frame.astype(np.float32)
    return accumulator


def subtract_from_temporal_accumulator(accumulator: object, frame: np.ndarray, state: EditorState) -> object:
    if state.use_opencl:
        return cv2.subtract(accumulator, cv2.UMat(frame.astype(np.float32)))
    accumulator -= frame.astype(np.float32)
    return accumulator


def finish_temporal_average(accumulator: object, frame_count: int, state: EditorState) -> np.ndarray:
    if frame_count <= 0:
        raise ValueError("Cannot average an empty frame list.")
    if state.use_opencl:
        return cv2.convertScaleAbs(accumulator, alpha=1.0 / frame_count).get()
    return np.clip(accumulator / frame_count, 0, 255).astype(np.uint8)


def accelerated_temporal_average(frames: list[np.ndarray], state: EditorState) -> np.ndarray:
    if not frames:
        raise ValueError("Cannot average an empty frame list.")
    accumulator = make_temporal_accumulator(frames[0], state)
    for frame in frames:
        accumulator = add_to_temporal_accumulator(accumulator, frame, state)
    return finish_temporal_average(accumulator, len(frames), state)
