# tambov

Utilities for preparing a wet/dry floor detection workflow from video.

## Interactive wet/dry floor mask mapper

`mask_mapper.py` opens a video and lets you paint three masks with an adjustable brush:

1. **floor** — the persistent full visible floor region that should be considered by the detector.
2. **dry** — parts of the floor that are visibly dry in the selected frame only.
3. **wet** — parts of the floor that are visibly wet in the selected frame only.

Dry and wet are mutually exclusive, frame-specific condition masks: painting dry removes wet in the same pixels on the current frame, and painting wet removes dry on the current frame. Both dry and wet are clipped to the persistent floor mask, so define the floor area first; the floor mask itself may overlap dry/wet and applies across all frames.

You can choose the reference frame with the horizontal slider before painting. For example, paint the persistent floor mask on a frame with a clear view, move the slider to a frame where dry floor is visible to paint that frame's dry mask, then move to a frame where wet floor is visible to paint that frame's wet mask.

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
| Increase/decrease brush size | `+` / `-` |
| Select floor mask | `1` or `f` |
| Select dry mask | `2` |
| Select wet mask | `3` or `w` |
| Reset selected mask | `r` |
| Undo last brush stroke or reset | `u` |
| Clip all frame-specific dry/wet masks to the floor mask | `i` |
| Select frame | Horizontal `Frame` slider |
| Next/previous frame | `n` / `p`, or right/left arrow |
| Jump 30 frames | `]` / `[` |
| Go to frame number | `g` |
| Save masks | `s` |
| Quit | `q` or Esc |

### Output files

Saving writes these files to `--out-dir`:

- `<video>_floor_mask.png` — persistent binary floor-area mask shared by all frames.
- `<video>_frame_<frame>_dry_mask.png` — frame-specific binary dry-floor mask for each frame that has dry/wet labels.
- `<video>_frame_<frame>_wet_mask.png` — frame-specific binary wet-floor mask for each frame that has dry/wet labels.
- `<video>_frame_<frame>_combined_labels.png` — frame-specific single-channel label image where `0=background`, `1=floor`, `2=dry`, and `3=wet`. Dry/wet labels are mutually exclusive and clipped to floor; dry/wet labels overwrite floor where they overlap in this export.
- `<video>_mask_metadata.json` — video dimensions, class IDs, floor-mask path, per-frame dry/wet/combined paths, and the last reference frame used.
