from __future__ import annotations

import time

import cv2
import numpy as np


def live_analysis_interval_frames(state) -> int:
    if state.live_analysis_interval <= 0:
        return 1
    source_fps = state.fps if state.fps > 0 else 30.0
    effective_fps = min(source_fps, state.live_target_fps) if state.live_target_fps is not None else source_fps
    return max(1, int(round(state.live_analysis_interval * effective_fps)))


def play_live_stream_with_analysis_overlay(
    state,
    capture: cv2.VideoCapture,
    analysis_capture: cv2.VideoCapture | None,
    *,
    run_wetness_analysis,
    make_empty_condition_masks,
    live_frame_stride,
    LiveFrameAverager,
    FrameView,
    make_hex_overlay,
    render_analysis_overlay_from_cache,
    open_capture,
    CONDITION_LABELS,
) -> int:
    if not state.condition_masks_by_frame:
        raise RuntimeError("No mask annotations are loaded; pass --load-dir with saved mask PNGs.")

    state.scale = 1.0
    state.hex_enabled = True
    state.analysis_enabled = False
    run_wetness_analysis(state, analysis_capture)
    if not state.analysis_enabled:
        raise RuntimeError("Cannot play live stream because wetness analysis found no dry/wet hex models.")

    if not state.is_live_source:
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    stream_window_name = "wet/dry floor live analysis"
    cv2.namedWindow(stream_window_name, cv2.WINDOW_NORMAL)
    fps = state.fps if state.fps > 0 else 30.0
    display_fps = min(fps, state.live_target_fps) if state.live_target_fps is not None else fps
    frame_delay_ms = max(1, int(round(1000.0 / display_fps)))
    frame_period_seconds = 1.0 / display_fps
    analysis_interval_frames = live_analysis_interval_frames(state)
    process_every_n = live_frame_stride(state)
    live_averager = LiveFrameAverager(state)
    empty_condition_masks = make_empty_condition_masks(state)
    cached_hex_overlay: np.ndarray | None = None
    cached_hex_mask: np.ndarray | None = None
    cached_average_wetness: float | None = None
    processed_frames = 0
    stream_start_time = time.monotonic()

    while True:
        target_elapsed = processed_frames * frame_period_seconds
        now_elapsed = time.monotonic() - stream_start_time
        behind_seconds = now_elapsed - target_elapsed
        catch_up_frames = max(0, int(behind_seconds / frame_period_seconds))

        read_failed = False
        drop_frames = max(0, process_every_n - 1 + catch_up_frames)
        for _ in range(drop_frames):
            if not capture.grab():
                read_failed = True
                break

        ok, frame = (False, None) if read_failed else capture.read()
        if not ok or frame is None:
            if not state.is_live_source:
                break
            print(f"[stream] read failed; reconnecting to {state.source!r} in {state.stream_reconnect_delay:.1f}s...")
            capture.release()
            time.sleep(state.stream_reconnect_delay)
            try:
                capture = open_capture(state.source, state.rtsp_transport)
            except RuntimeError as error:
                print(f"[stream] reconnect failed: {error}")
                time.sleep(state.stream_reconnect_delay)
            live_averager = LiveFrameAverager(state)
            cached_hex_overlay = None
            cached_hex_mask = None
            cached_average_wetness = None
            stream_start_time = time.monotonic() - (processed_frames * frame_period_seconds)
            continue

        state.frame_index = processed_frames
        frame_masks = state.condition_masks_by_frame.get(processed_frames, empty_condition_masks)
        state.current_condition_frame = processed_frames
        for label in CONDITION_LABELS:
            state.masks[label] = frame_masks[label]

        analysis_sample_frame = live_averager.average_for_frame(processed_frames, frame)
        should_recalculate = processed_frames % analysis_interval_frames == 0 or cached_hex_overlay is None
        if should_recalculate:
            view = FrameView(frame=frame, display_frame=frame.copy(), analysis_sample_frame=analysis_sample_frame)
            cached_hex_overlay = make_hex_overlay(state, view)
            cached_hex_mask = view.hex_mask
            cached_average_wetness = view.analysis_average_wetness

        output_frame = render_analysis_overlay_from_cache(
            state, frame, cached_hex_overlay, cached_hex_mask, cached_average_wetness
        )
        cv2.imshow(stream_window_name, output_frame)

        remaining_delay_seconds = ((processed_frames + 1) * frame_period_seconds) - (time.monotonic() - stream_start_time)
        dynamic_wait_ms = min(frame_delay_ms, max(1, int(round(remaining_delay_seconds * 1000.0))))
        key = cv2.waitKey(dynamic_wait_ms) & 0xFF
        if key in (ord("q"), 27):
            break

        processed_frames += 1

    capture.release()
    cv2.destroyWindow(stream_window_name)
    print(f"Played {processed_frames} live-stream frame(s).")
    return 0
