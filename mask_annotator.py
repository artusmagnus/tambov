#!/usr/bin/env python3
"""Interactive annotation entrypoint for wet/dry floor masks."""

from __future__ import annotations

import argparse
from pathlib import Path

import mask_mapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactively annotate floor, dry, wet, and obstruction masks for a video source."
    )
    parser.add_argument("source", nargs="?", help="Path, camera index, RTSP/HTTP URL, or other OpenCV video source.")
    parser.add_argument(
        "--video",
        "--source",
        "--input",
        "--stream",
        dest="source_option",
        help="Path, camera index, RTSP/HTTP URL, or other OpenCV video source.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("mask_output"), help="Directory where annotations are saved.")
    parser.add_argument("--load-dir", type=Path, help="Load an existing annotation folder before editing.")
    parser.add_argument("--scale", type=float, default=1.0, help="Interactive display scale.")
    parser.add_argument("--alpha", type=float, default=0.45, help="Mask overlay opacity from 0.0 to 1.0.")
    parser.add_argument("--analysis-max-distance", type=float, default=0.08, help="OKLab rejection threshold.")
    parser.add_argument("--analysis-time-window", type=float, default=5.0, help="Seconds to average for analysis frames.")
    parser.add_argument("--analysis-sample-interval", type=float, default=1.0, help="Seconds between analysis samples.")
    parser.add_argument("--hex-size", type=int, default=40, help="Hex cell radius in original source pixels.")
    parser.add_argument("--show-hex-values", action="store_true", help="Draw numeric wetness values inside analysis hexes.")
    parser.add_argument("--use-opencl", action="store_true", help="Use OpenCV OpenCL acceleration when available.")
    parser.add_argument("--brush-size", type=int, default=20, help="Initial brush radius in source pixels.")
    parser.add_argument(
        "--max-display-width",
        type=int,
        default=1280,
        help="Automatically downscale the display window to this width. Use 0 to disable.",
    )
    args = parser.parse_args()
    if args.source and args.source_option and args.source != args.source_option:
        parser.error("Provide the input source either positionally or with --video/--source/--input/--stream, not both.")
    args.source = args.source_option or args.source
    if not args.source and not args.ui:
        parser.error("an input source is required (positional source or --video/--source/--input/--stream), unless --ui is used.")
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.scale <= 0:
        raise ValueError("--scale must be greater than 0.")
    if not 0 <= args.alpha <= 1:
        raise ValueError("--alpha must be between 0.0 and 1.0.")
    if args.brush_size <= 0:
        raise ValueError("--brush-size must be greater than 0.")
    if args.hex_size <= 0:
        raise ValueError("--hex-size must be greater than 0.")
    if args.analysis_max_distance < 0:
        raise ValueError("--analysis-max-distance must be 0 or greater.")
    if args.analysis_time_window < 0:
        raise ValueError("--analysis-time-window must be 0 or greater.")
    if args.analysis_sample_interval < 0:
        raise ValueError("--analysis-sample-interval must be 0 or greater.")
    if args.max_display_width < 0:
        raise ValueError("--max-display-width must be 0 or greater.")


def main() -> int:
    args = parse_args()

    if args.ui:
        import settings_launcher
        settings_launcher.launch_with_defaults({
            "script": "mask_annotator.py",
            "source": args.source,
            "out_dir": str(args.out_dir),
            "load_dir": str(args.load_dir) if args.load_dir else None,
            "scale": args.scale,
            "alpha": args.alpha,
            "analysis_max_distance": args.analysis_max_distance,
            "analysis_time_window": args.analysis_time_window,
            "analysis_sample_interval": args.analysis_sample_interval,
            "hex_size": args.hex_size,
            "show_hex_values": args.show_hex_values,
            "use_opencl": args.use_opencl,
            "brush_size": args.brush_size,
            "max_display_width": args.max_display_width,
        })
        return 0

    import cv2 as cv2_module
    import numpy as np_module

    mask_mapper.cv2 = cv2_module
    mask_mapper.np = np_module
    validate_args(args)

    use_opencl = mask_mapper.configure_opencl(args.use_opencl)
    capture = mask_mapper.open_capture(args.source)
    try:
        width = int(capture.get(mask_mapper.cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(mask_mapper.cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = mask_mapper.source_frame_count(capture)
        fps = mask_mapper.source_fps_with_fallback(capture)
        if width <= 0 or height <= 0:
            raise RuntimeError(f"Could not determine frame size for source: {args.source}")

        if args.max_display_width and width > args.max_display_width:
            args.scale = min(args.scale, args.max_display_width / width)
            print(f"Display scale set to {args.scale:.3f} for smoother editing.")

        state = mask_mapper.EditorState(
            source=args.source,
            source_stem=mask_mapper.source_name_stem(args.source),
            video_path=Path(args.source),
            out_dir=args.out_dir,
            frame_count=frame_count,
            width=width,
            height=height,
            fps=fps,
            scale=args.scale,
            alpha=args.alpha,
            analysis_max_distance=args.analysis_max_distance,
            analysis_time_window=args.analysis_time_window,
            analysis_sample_interval=args.analysis_sample_interval,
            live_analysis_interval=1.0,
            live_target_fps=None,
            stream_reconnect_delay=2.0,
            rtsp_transport="auto",
            is_live_source=mask_mapper.is_probably_live_source(args.source, frame_count),
            show_hex_values=args.show_hex_values,
            average_wetness_only=False,
            use_opencl=use_opencl,
            use_obstruction_colors=False,
            brush_size=args.brush_size,
            hex_cell_size=args.hex_size,
            masks={label: mask_mapper.np.zeros((height, width), dtype=mask_mapper.np.uint8) for label in mask_mapper.MASK_CLASSES},
        )
        state.condition_masks_by_frame[state.frame_index] = {label: state.masks[label] for label in mask_mapper.CONDITION_LABELS}
        state.current_condition_frame = state.frame_index
        state.display_masks = {label: mask_mapper.make_display_mask(mask, state) for label, mask in state.masks.items()}
        if args.load_dir:
            mask_mapper.load_outputs(state, args.load_dir)

        return mask_mapper.run_annotation_editor(state, capture)
    finally:
        capture.release()
        mask_mapper.cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
