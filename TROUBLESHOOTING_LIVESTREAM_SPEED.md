# Live-stream video plays too fast: fix checklist

If live video appears sped up, the issue is usually timestamp/fps mismatch in the ingest/transcode pipeline.

## 1) Enforce real-time ingest

For FFmpeg-based ingest/relay, read input in wall-clock real time:

```bash
ffmpeg -re -i input.mp4 ...
```

If source is a device/camera, avoid forcing fps unless needed.

## 2) Stop forcing output fps incorrectly

Do **not** set `-r` on output unless you intentionally resample.

Bad (can speed up/slow down):

```bash
ffmpeg -i input -r 60 output
```

Prefer preserving source cadence:

```bash
ffmpeg -i input -vsync vfr output
```

Or explicitly normalize safely (if required):

```bash
ffmpeg -i input -vf fps=30 -r 30 output
```

## 3) Generate proper timestamps

When inputs have broken timestamps, regenerate PTS and keep them monotonic:

```bash
ffmpeg -fflags +genpts -use_wallclock_as_timestamps 1 -i input ...
```

## 4) HLS-specific settings

Ensure segment timing is stable and not too small:

```bash
-hls_time 2 -hls_list_size 6 -hls_flags delete_segments+independent_segments
```

## 5) WebRTC-specific checks

- Ensure sender RTP timestamp clock rate matches codec expectations.
- Do not enqueue frames faster than capture cadence.
- Verify client is not set to >1.0 playback rate.

## 6) Player sanity checks

- Playback rate = 1.0x.
- Test another browser/device.
- Toggle hardware acceleration to rule out decode bug.

## 7) Minimal known-good FFmpeg relay template

```bash
ffmpeg -re -fflags +genpts -i INPUT \
  -c:v libx264 -preset veryfast -tune zerolatency -pix_fmt yuv420p \
  -g 60 -keyint_min 60 \
  -c:a aac -ar 48000 -ac 2 -b:a 128k \
  -f flv RTMP_URL
```

## 8) How to validate the fix

1. Log input fps + output fps.
2. Compare 60 seconds of wall clock with 60 seconds of displayed stream time.
3. Check A/V sync drift over 5+ minutes.

If stream time runs faster than wall clock, revisit steps 2 and 3 first.
