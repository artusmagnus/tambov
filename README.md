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

Adjust the OKLab analysis rejection threshold if moving objects or lighting changes are being accepted/rejected too aggressively:

```bash
python mask_mapper.py /path/to/video.mp4 --analysis-max-distance 0.06 --out-dir mask_output
```

Load a previous annotation session by passing the saved metadata JSON:

```bash
python mask_mapper.py /path/to/video.mp4 --load-metadata mask_output/video_mask_metadata.json --out-dir mask_output
```

Disable automatic display downscaling if you need a full-resolution window:

```bash
python mask_mapper.py /path/to/video.mp4 --max-display-width 0 --out-dir mask_output
```

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

Press Enter to run dry-to-wet analysis for the current hex cell size. The tool finds hex cells that have both dry and wet examples somewhere in the annotated video, converts their average colors to OKLab, and treats the line from the dry color to the wet color as a 0-100 wetness axis. After analysis, navigate to any frame to evaluate the persistent floor area, including frames that were not annotated. The analysis view ignores the active mask layer and shows modeled floor hexes, excluding any obstruction mask on the current frame, with red for dry-like cells near `0` and green for wet-like cells near `100`. Floor hexes that do not have a learned dry-to-wet model, or whose colour is too far from the OKLab dry-to-wet line, are shown with a pink/black checker texture as irrelevant/non-floor colour changes.

### Output files

Saving writes these files to `--out-dir`:

- `<video>_floor_mask.png` — persistent binary floor-area mask shared by all frames.
- `<video>_frame_<frame>_dry_mask.png` — frame-specific binary dry-floor mask for each frame that has condition labels.
- `<video>_frame_<frame>_wet_mask.png` — frame-specific binary wet-floor mask for each frame that has condition labels.
- `<video>_frame_<frame>_obstruction_mask.png` — frame-specific binary obstruction mask for each frame that has condition labels.
- `<video>_mask_metadata.json` — video dimensions, class IDs, floor-mask path, per-frame dry/wet/obstruction mask paths, and the last reference frame used. Pass this file to `--load-metadata` to continue editing saved masks.
