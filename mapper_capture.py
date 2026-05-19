from __future__ import annotations

import os
from urllib.parse import quote, unquote, urlparse, urlsplit, urlunsplit

import cv2


def clamp_frame(index: int, frame_count: int) -> int:
    if frame_count <= 0:
        return max(0, index)
    return max(0, min(frame_count - 1, index))


def source_capture_value(source: str) -> str | int:
    return int(source) if source.isdigit() else source


def is_rtsp_source(source: str) -> bool:
    return urlparse(source).scheme.lower() == "rtsp"


def redact_source_credentials(source: str) -> str:
    parsed = urlsplit(source)
    if parsed.scheme.lower() != "rtsp" or parsed.username is None:
        return source
    host = parsed.hostname or ""
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    auth = "***"
    if parsed.password is not None:
        auth += ":***"
    netloc = f"{auth}@{host}" if host else auth
    return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))


def quote_url_part(value: str | None) -> str | None:
    return None if value is None else quote(unquote(value), safe="")


def rtsp_url_with_encoded_credentials(source: str) -> str | None:
    if not is_rtsp_source(source):
        return None
    parsed = urlsplit(source)
    if parsed.username is None:
        return None
    host = parsed.hostname or ""
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    username = quote_url_part(parsed.username)
    password = quote_url_part(parsed.password)
    auth = username if username is not None else ""
    if password is not None:
        auth = f"{auth}:{password}"
    netloc = f"{auth}@{host}" if host else auth
    return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))


def rtsp_url_with_trailing_slash(source: str) -> str | None:
    if not is_rtsp_source(source):
        return None
    parsed = urlsplit(source)
    if parsed.path.endswith("/"):
        return None
    return urlunsplit((parsed.scheme, parsed.netloc, f"{parsed.path}/", parsed.query, parsed.fragment))


def add_unique_source_attempt(attempts: list[tuple[str, str]], source: str | None, label: str) -> None:
    if source is None:
        return
    if any(existing_source == source for existing_source, _ in attempts):
        return
    attempts.append((source, label))


def rtsp_source_attempts(source: str) -> tuple[tuple[str, str], ...]:
    encoded_source = rtsp_url_with_encoded_credentials(source)
    slash_source = rtsp_url_with_trailing_slash(source)
    encoded_slash_source = rtsp_url_with_trailing_slash(encoded_source) if encoded_source is not None else None

    attempts: list[tuple[str, str]] = []
    add_unique_source_attempt(attempts, source, "original URL")
    add_unique_source_attempt(attempts, slash_source, "original URL with trailing slash")
    add_unique_source_attempt(attempts, encoded_source, "URL-encoded credentials")
    add_unique_source_attempt(attempts, encoded_slash_source, "URL-encoded credentials with trailing slash")
    return tuple(attempts)


def rtsp_transport_attempts(source: str, transport: str) -> tuple[str, ...]:
    if not is_rtsp_source(source):
        return ("auto",)
    if transport != "auto":
        return (transport,)
    return ("tcp", "udp", "udp_multicast", "http", "auto")


def rtsp_transport_option(source: str, transport: str) -> str | None:
    if transport == "auto" or not is_rtsp_source(source):
        return None
    return f"rtsp_transport;{transport}"


def ffmpeg_capture_options_with_rtsp_transport(source: str, transport: str) -> str | None:
    transport_option = rtsp_transport_option(source, transport)
    existing = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS", "")
    if not existing:
        return transport_option
    existing_options = [
        option
        for option in existing.split("|")
        if option and not option.startswith("rtsp_transport;")
    ]
    if transport_option is not None:
        existing_options.append(transport_option)
    return "|".join(existing_options) if existing_options else None


def capture_backend(source: str) -> int | None:
    if is_rtsp_source(source):
        return cv2.CAP_FFMPEG
    return None


def capture_with_ffmpeg_options(source: str, rtsp_transport: str) -> cv2.VideoCapture:
    ffmpeg_options = ffmpeg_capture_options_with_rtsp_transport(source, rtsp_transport)
    previous_options = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS")
    if ffmpeg_options is None:
        os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
    else:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = ffmpeg_options
    try:
        backend = capture_backend(source)
        if backend is None:
            return cv2.VideoCapture(source_capture_value(source))
        return cv2.VideoCapture(source_capture_value(source), backend)
    finally:
        if previous_options is None:
            os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
        else:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = previous_options


def format_rtsp_transport_attempt(transport: str) -> str:
    return "OpenCV default" if transport == "auto" else transport


def open_capture(source: str, rtsp_transport: str = "auto") -> cv2.VideoCapture:
    attempted: list[str] = []
    source_attempts = rtsp_source_attempts(source) if is_rtsp_source(source) else ((source, "source"),)
    for source_attempt, source_label in source_attempts:
        for transport_attempt in rtsp_transport_attempts(source_attempt, rtsp_transport):
            capture = capture_with_ffmpeg_options(source_attempt, transport_attempt)
            formatted_transport = format_rtsp_transport_attempt(transport_attempt)
            if capture.isOpened():
                if is_rtsp_source(source):
                    if source_attempt == source:
                        print(f"Opened RTSP source {source!r} using {formatted_transport} transport.")
                    else:
                        print(
                            f"Opened RTSP source {source!r} using {source_label} and {formatted_transport} transport."
                        )
                return capture
            capture.release()
            attempted.append(f"{source_label} / {formatted_transport}")

    safe_source = redact_source_credentials(source) if is_rtsp_source(source) else source
    hint = ""
    if is_rtsp_source(source):
        attempts_text = ", ".join(attempted)
        hint = (
            f" (tried: {attempts_text}; if SETUP still fails, verify the RTSP URL, credentials, channel path, "
            "and that VLC/ffplay can open the stream; RTSP attempts are forced through OpenCV's FFmpeg "
            "backend to avoid image-sequence fallback warnings)"
        )
    raise RuntimeError(f"Could not open source: {safe_source}{hint}")


def source_name_stem(source: str) -> str:
    candidate = source
    parsed = urlparse(source)
    if parsed.scheme:
        candidate = parsed.path.rsplit("/", 1)[-1] or parsed.netloc or source
    stem = candidate.rsplit(".", 1)[0]
    stem = stem.strip() or "video"
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in stem)


def is_probably_live_source(source: str, frame_count: int) -> bool:
    parsed = urlparse(source)
    if parsed.scheme.lower() in {"rtsp", "rtmp", "http", "https", "udp", "tcp"}:
        return True
    if frame_count <= 0:
        return True
    return source.isdigit()


def source_fps_with_fallback(capture: cv2.VideoCapture) -> float:
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    return fps if fps > 0 else 30.0


def source_frame_count(capture: cv2.VideoCapture) -> int:
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    return frame_count if frame_count > 0 else 0
