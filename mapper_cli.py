from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run wet/dry floor analysis overlays from saved annotations."
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
        "--analysis-time-window",
        type=float,
        default=5.0,
        help=(
            "Seconds of video to average for each analysed frame before projecting hex wetness. "
            "Use 0 to disable temporal averaging."
        ),
    )
    parser.add_argument(
        "--analysis-sample-interval",
        type=float,
        default=1.0,
        help=(
            "Seconds between frames sampled into each temporal analysis average. "
            "Use 0 to sample every frame."
        ),
    )
    parser.add_argument(
        "--hex-size",
        type=int,
        default=40,
        help="Hex cell radius in original video pixels for analysis and batch video processing.",
    )
    parser.add_argument(
        "--show-hex-values",
        action="store_true",
        help="Draw numeric wetness values inside analysis hexes. Disabled by default for faster rendering.",
    )
    parser.add_argument(
        "--average-wetness-only",
        action="store_true",
        help=(
            "In analysis rendering, hide the hex/checker overlay and draw only the calculated "
            "average wetness label over the source frame."
        ),
    )
    parser.add_argument(
        "--use-opencl",
        action="store_true",
        help=(
            "Use OpenCV OpenCL acceleration when available for resize, blending, "
            "and temporal averaging operations."
        ),
    )
    parser.add_argument(
        "--process-video",
        action="store_true",
        help="Process the entire video with the analysis hex overlay using masks loaded from --load-dir.",
    )
    parser.add_argument(
        "--live-stream",
        action="store_true",
        help="Play the input video like a live stream with the analysis hex overlay using masks loaded from --load-dir.",
    )
    parser.add_argument(
        "--live-analysis-interval",
        type=float,
        default=1.0,
        help=(
            "Seconds between live-stream analysis overlay recalculations. "
            "Use 0 to recalculate on every frame."
        ),
    )
    parser.add_argument(
        "--fps",
        type=float,
        help=(
            "Target processing/display FPS for --live-stream. If lower than source FPS, "
            "intermediate frames are skipped with VideoCapture.grab()."
        ),
    )
    parser.add_argument(
        "--stream-reconnect-delay",
        type=float,
        default=2.0,
        help="Seconds to wait before reopening a failed live source read.",
    )
    parser.add_argument(
        "--rtsp-transport",
        choices=("auto", "tcp", "udp", "udp_multicast", "http"),
        default="auto",
        help=(
            "RTSP transport passed to OpenCV/FFmpeg through OPENCV_FFMPEG_CAPTURE_OPTIONS. "
            "The default auto tries tcp, udp, udp_multicast, http, then OpenCV's default."
        ),
    )
    parser.add_argument(
        "--use-obstruction-colors",
        action="store_true",
        help=(
            "Use obstruction annotations as weighted color references during analysis outlier filtering. "
            "Disabled by default."
        ),
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
    parser.add_argument("--ui", action="store_true", help="Open settings UI prefilled from provided CLI args.")
    parser.add_argument(
        "--max-display-width",
        type=int,
        default=1280,
        help=(
            "Automatically downscale the display window to this width for smoother editing. "
            "Use 0 to disable automatic downscaling."
        ),
    )
    args = parser.parse_args()
    if args.source and args.source_option and args.source != args.source_option:
        parser.error("Provide the input source either positionally or with --video/--source/--input/--stream, not both.")
    args.source = args.source_option or args.source
    if not args.source and not args.ui:
        parser.error("an input source is required (positional source or --video/--source/--input/--stream), unless --ui is used.")
    return args
