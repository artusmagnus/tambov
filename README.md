# tambov

Utilities for preparing a wet/dry floor detection workflow from video.

## Interactive wet/dry floor mask mapper

`mask_mapper.py` opens a video and lets you paint four masks with an adjustable brush:

1. **floor** — the persistent full visible floor region that should be considered by the detector.
2. **dry** — parts of the floor that are visibly dry in the selected frame only.
3. **wet** — parts of the floor that are visibly wet in the selected frame only.
4. **obstruction** — frame-specific obstructions that cover the floor and should not be labeled dry or wet.

Dry, wet, and obstruction are mutually exclusive, frame-specific condition masks: painting one removes the other condition labels in the same pixels on the current frame. All three condition masks are clipped to the persistent floor mask, so define the floor area first; the floor mask itself may overlap condition labels and applies across all frames.

You can choose the reference frame with the horizontal slider before painting. For example, paint the persistent floor mask on a frame with a clear view, move the slider to a frame where dry floor is visible to paint that frame's dry mask, then move to frames where wet floor or obstructions are visible to paint those frame-specific masks.

### Install

```bash
python -m pip install -r requirements.txt
```

### Run

```bash
python mask_mapper.py /path/to/video.mp4 --out-dir mask_output
```

For smoother editing on large videos, the tool automatically limits the display window to 1280 pixels wide while keeping saved masks at the original video resolution. You can also set the display scale or brush size manually:

```bash
python mask_mapper.py /path/to/video.mp4 --scale 0.5 --brush-size 15 --out-dir mask_output
```

Adjust the OKLab analysis rejection threshold if moving objects or lighting changes are being accepted/rejected too aggressively. Analysis accumulates source frames into a temporal average over a 5-second window by default, sampling one frame per second inside that window, then runs the hex analysis on that averaged frame to reduce short-lived interference. Use `--analysis-time-window` to change the window, `0` for `--analysis-time-window` to disable temporal averaging, and `--analysis-sample-interval` to change how often frames are sampled into the average (`0` samples every frame):

```bash
python mask_mapper.py /path/to/video.mp4 --analysis-max-distance 0.06 --analysis-time-window 3 --analysis-sample-interval 0.5 --out-dir mask_output
```

Load a previous annotation session by passing the folder that contains the saved mask images. The loader reconstructs annotations from PNG filenames, so it does not need a JSON metadata file:

```bash
python mask_mapper.py /path/to/video.mp4 --load-dir mask_output --out-dir mask_output
```

Disable automatic display downscaling if you need a full-resolution window:

```bash
python mask_mapper.py /path/to/video.mp4 --max-display-width 0 --out-dir mask_output
```

### Process an entire video

After saving masks, run batch processing to reconstruct the annotations from a mask folder, build the same OKLab dry-to-wet hex analysis used by the interactive Enter key, and write a full video with the analysis hex overlay applied to every frame:

```bash
python mask_mapper.py /path/to/video.mp4 --process-video --load-dir mask_output --output-video wetness_overlay.mp4
```

If `--output-video` is omitted, the processed video is written to `<out-dir>/<video>_hex_overlay.mp4`. Use `--hex-size` to change the hex cell radius in original video pixels; if it is omitted, the default hex size is `40` pixels. Add `--average-wetness-only` when you want the output frames left unmodified except for the calculated average wetness label:

```bash
python mask_mapper.py /path/to/video.mp4 --process-video --load-dir mask_output --hex-size 30
python mask_mapper.py /path/to/video.mp4 --process-video --load-dir mask_output --average-wetness-only
```

### Play as a live stream

To review the same analysis overlay without writing a file, play the input video like a live stream. The stream uses the saved masks, builds the dry-to-wet model once, displays frames at the source FPS, applies temporal averaging only over frames that have already arrived, and recalculates the displayed analysis overlay once per second by default:

```bash
python mask_mapper.py /path/to/video.mp4 --live-stream --load-dir mask_output
```

Use `--live-analysis-interval` to control how often the live overlay is recalculated; for example, `0.5` updates twice per second, and `0` updates on every frame. Add `--average-wetness-only` to hide the hex/checker overlay and display only the average wetness label. Press `q` or Esc to stop the live stream window.

### Command-line flags

| Argument | Default | Description |
| --- | --- | --- |
| `video` | Required | Path to the input video file. |
| `--out-dir` | `mask_output` | Directory where saved mask PNGs are written. Batch mode also uses it for the default output video path. |
| `--load-dir` | Not set | Reconstruct a prior annotation session by scanning this directory for saved mask PNGs. Required with `--process-video` and `--live-stream`. |
| `--scale` | `1.0` | Interactive display scale. Painting coordinates and saved masks still use original video resolution. Ignored by `--process-video` and `--live-stream`, which render at full resolution. |
| `--alpha` | `0.45` | Opacity for the regular mask-color overlay, from `0.0` to `1.0`. |
| `--analysis-max-distance` | `0.08` | Maximum OKLab perpendicular distance from a dry-to-wet colour line before an analysis hex is treated as an unrelated colour change and shown with the checker texture. |
| `--analysis-time-window` | `5.0` | Seconds of video to accumulate into an averaged source frame before projecting hex wetness. Use `0` to disable temporal averaging. |
| `--analysis-sample-interval` | `1.0` | Seconds between source frames sampled into the temporal average. Use `0` to sample every frame. |
| `--hex-size` | `40` | Hex cell radius in original video pixels for interactive analysis and batch processing. |
| `--show-hex-values` | Off | Draw numeric wetness values inside analysis hexes. Disabled by default to avoid expensive per-hex text rendering. |
| `--average-wetness-only` | Off | Hide the analysis hex/checker overlay and draw only the calculated average wetness label over the source frame. |
| `--use-opencl` | Off | Use OpenCV OpenCL acceleration when available for resize, alpha blending, and temporal frame averaging. Falls back to CPU if the installed OpenCV build cannot use OpenCL. |
| `--process-video` | Off | Run non-interactive batch mode: load masks from `--load-dir`, build the analysis model, and write a full video with the analysis hex overlay. Cannot be combined with `--live-stream`. |
| `--live-stream` | Off | Play the input video in an OpenCV window with the analysis hex overlay at the source FPS. Requires `--load-dir` and cannot be combined with `--process-video`. |
| `--live-analysis-interval` | `1.0` | Seconds between live-stream analysis overlay recalculations. Use `0` to recalculate on every frame. |
| `--output-video` | `<out-dir>/<video>_hex_overlay.mp4` | Output path for `--process-video`. |
| `--brush-size` | `20` | Initial brush radius in original video pixels. |
| `--max-display-width` | `1280` | Automatically downscale the interactive display window to this width for smoother editing. Use `0` to disable automatic downscaling. |

### Controls

| Action | Control |
| --- | --- |
| Paint or erase | Hold and drag left mouse button |
| Draw mode | `d` |
| Erase mode | `e` |
| Increase/decrease brush size | `+` / `-` when hex view is off |
| Toggle averaged-color hex view | Space |
| Run OKLab dry-to-wet analysis | Enter |
| Increase/decrease hex cell size | `+` / `-` when hex view is on |
| Select floor mask / floor hex layer | `1` or `f` |
| Select dry mask / dry hex layer | `2` |
| Select wet mask / wet hex layer | `3` or `w` |
| Select obstruction mask / obstruction hex layer | `4` or `o` |
| Reset selected mask | `r` |
| Undo last brush stroke or reset | `u` |
| Clip all frame-specific condition masks to the floor mask | `i` |
| Select frame | Horizontal `Frame` slider |
| Next/previous frame | `n` / `p`, or right/left arrow |
| Jump 30 frames | `]` / `[` |
| Go to frame number | `g` |
| Save masks | `s` |
| Quit | `q` or Esc |

### Hex visualization

Press Space to toggle a hexagon visualization of the current frame. The frame is divided into a hexagon mesh, each visible hexagon is filled with the average color of the original image pixels inside that cell, and `+` / `-` changes the hex cell size while the view is enabled. Press `1`, `2`, `3`, or `4` to show hexagons only for the corresponding mask layer: floor, dry, wet, or obstruction. While hex view is enabled, the regular mask-color overlay is hidden and hexes are drawn 80% transparent over the original frame.

Press Enter to run dry-to-wet analysis for the current hex cell size. The tool first finds hex cells that have both dry and wet examples somewhere in the annotated video, converts their average colors to OKLab, and treats the line from the dry color to the wet color as a 0-100 wetness axis. When analysis is displayed, source frames inside the configured `--analysis-time-window` are sampled according to `--analysis-sample-interval` and accumulated into an averaged frame first, then hex wetness is projected from that averaged frame so brief interference affects the result less. If another hex has only dry or only wet examples, the tool now keeps it usable by finding the closest same-state color among the fully paired hexes and borrowing that similar hex's missing counterpart color. After analysis, navigate to any frame to evaluate the persistent floor area, including frames that were not annotated. The analysis view ignores the active mask layer and shows modeled floor hexes, excluding any obstruction mask on the current frame, with red for dry-like cells near `0`, yellow for midrange cells near `50`, and green for wet-like cells near `100`. The average wetness value across all valid analysis hexes in the current frame is shown at the top right. Use `--show-hex-values` if you also want per-hex numeric labels; they are off by default for faster rendering. Floor hexes that do not have a learned or inferred dry-to-wet model, or whose colour is too far from the OKLab dry-to-wet line, are shown with a pink/black checker texture as irrelevant/non-floor colour changes.

### GPU acceleration

Some frame-wide OpenCV operations can be moved to the graphics card with OpenCL:

```bash
python mask_mapper.py /path/to/video.mp4 --process-video --load-dir mask_output --use-opencl
```

`--use-opencl` accelerates supported resize, overlay blending, and temporal frame averaging operations through OpenCV's OpenCL backend when the installed OpenCV package and hardware support it. The per-hex mask geometry and per-cell wetness projection still run on the CPU because those calculations operate on irregular hex masks and small per-cell reductions. If OpenCL is unavailable, the tool prints a warning and continues on the CPU.

### Output files

Saving writes these mask images to `--out-dir`:

- `<video>_floor_mask.png` — persistent binary floor-area mask shared by all frames.
- `<video>_frame_<frame>_dry_mask.png` — frame-specific binary dry-floor mask for each frame that has condition labels.
- `<video>_frame_<frame>_wet_mask.png` — frame-specific binary wet-floor mask for each frame that has condition labels.
- `<video>_frame_<frame>_obstruction_mask.png` — frame-specific binary obstruction mask for each frame that has condition labels.
