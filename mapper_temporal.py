from __future__ import annotations

from collections import deque

import cv2
import numpy as np

def read_frame(capture: cv2.VideoCapture, index: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {index} from the video.")
    return frame



def analysis_window_radius_frames(state: EditorState) -> int:
    fps = state.fps if state.fps > 0 else 30.0
    return max(0, int(round(state.analysis_time_window * fps / 2.0)))


def analysis_sample_interval_frames(state: EditorState) -> int:
    if state.analysis_sample_interval <= 0:
        return 1
    fps = state.fps if state.fps > 0 else 30.0
    return max(1, int(round(state.analysis_sample_interval * fps)))


def first_sample_index_in_range(start_index: int, interval_frames: int) -> int:
    if interval_frames <= 1:
        return start_index
    return start_index + ((interval_frames - (start_index % interval_frames)) % interval_frames)


def read_temporal_average_frame(
    capture: cv2.VideoCapture,
    state: EditorState,
    frame_index: int,
) -> np.ndarray:
    radius = analysis_window_radius_frames(state)
    if radius <= 0:
        return read_frame(capture, frame_index)

    if state.frame_count > 0:
        start_index = clamp_frame(frame_index - radius, state.frame_count)
        end_index = clamp_frame(frame_index + radius, state.frame_count)
    else:
        start_index = max(0, frame_index - radius)
        end_index = frame_index + radius
    interval_frames = analysis_sample_interval_frames(state)
    first_sample_index = first_sample_index_in_range(start_index, interval_frames)
    frames = [read_frame(capture, sample_index) for sample_index in range(first_sample_index, end_index + 1, interval_frames)]
    if not frames:
        return read_frame(capture, frame_index)
    return accelerated_temporal_average(frames, state)


class TemporalFrameAverager:
    def __init__(self, state: EditorState, capture: cv2.VideoCapture) -> None:
        self.state = state
        self.capture = capture
        self.radius = analysis_window_radius_frames(state)
        self.interval_frames = analysis_sample_interval_frames(state)
        self.next_frame_index = 0
        self.frames: deque[tuple[int, np.ndarray]] = deque()
        self.accumulator: object | None = None
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def _append_frame(self, frame_index: int, frame: np.ndarray) -> None:
        if self.accumulator is None:
            self.accumulator = make_temporal_accumulator(frame, self.state)
        self.accumulator = add_to_temporal_accumulator(self.accumulator, frame, self.state)
        self.frames.append((frame_index, frame))

    def _drop_before(self, start_index: int) -> None:
        while self.frames and self.frames[0][0] < start_index:
            _frame_index, frame = self.frames.popleft()
            if self.accumulator is not None:
                self.accumulator = subtract_from_temporal_accumulator(self.accumulator, frame, self.state)

    def average_for_frame(self, frame_index: int, fallback_frame: np.ndarray) -> np.ndarray:
        if self.radius <= 0:
            return fallback_frame

        if self.state.frame_count > 0:
            start_index = clamp_frame(frame_index - self.radius, self.state.frame_count)
            end_index = clamp_frame(frame_index + self.radius, self.state.frame_count)
        else:
            start_index = max(0, frame_index - self.radius)
            end_index = frame_index + self.radius
        while self.next_frame_index <= end_index:
            ok, frame = self.capture.read()
            if not ok or frame is None:
                break
            if self.next_frame_index % self.interval_frames == 0:
                self._append_frame(self.next_frame_index, frame)
            self.next_frame_index += 1

        self._drop_before(start_index)
        if self.accumulator is None or not self.frames:
            return fallback_frame
        return finish_temporal_average(self.accumulator, len(self.frames), self.state)


class LiveFrameAverager:
    def __init__(self, state: EditorState) -> None:
        self.state = state
        self.window_frames = max(0, int(round(state.analysis_time_window * (state.fps if state.fps > 0 else 30.0))))
        self.interval_frames = analysis_sample_interval_frames(state)
        self.frames: deque[tuple[int, np.ndarray]] = deque()
        self.accumulator: object | None = None

    def average_for_frame(self, frame_index: int, frame: np.ndarray) -> np.ndarray:
        if self.window_frames <= 0:
            return frame

        if frame_index % self.interval_frames == 0:
            if self.accumulator is None:
                self.accumulator = make_temporal_accumulator(frame, self.state)
            self.accumulator = add_to_temporal_accumulator(self.accumulator, frame, self.state)
            self.frames.append((frame_index, frame))

        oldest_allowed = max(0, frame_index - self.window_frames + 1)
        while self.frames and self.frames[0][0] < oldest_allowed:
            _old_frame_index, old_frame = self.frames.popleft()
            self.accumulator = subtract_from_temporal_accumulator(self.accumulator, old_frame, self.state)

        if self.accumulator is None or not self.frames:
            return frame
        return finish_temporal_average(self.accumulator, len(self.frames), self.state)


def update_analysis_sample_frame(
    state: EditorState,
    view: FrameView,
    capture: cv2.VideoCapture,
) -> None:
    if not state.analysis_enabled:
        view.analysis_sample_frame = None
        return
    view.analysis_sample_frame = read_temporal_average_frame(capture, state, state.frame_index)
    view.hex_overlay = None
    view.hex_mask = None
    view.hex_cache_key = None
    view.analysis_average_wetness = None
    state.render_dirty = True
