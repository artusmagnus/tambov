# tambov

Utilities for preparing a wet/dry floor detection workflow from video.

## Interactive wet/dry floor mask mapper

`mask_mapper.py` opens a video and lets you draw three separate masks:

1. **floor** — the full visible floor region that should be considered by the detector.
2. **dry** — parts of the floor that are visibly dry in the chosen reference frame.
3. **wet** — parts of the floor that are visibly wet in the chosen reference frame.

You can navigate through the video before drawing, so choose frames where each area is easiest to label. For example, draw the floor mask on a frame with a clear view, move to a frame where dry floor is visible to draw the dry mask, then move to a frame where wet floor is visible to draw the wet mask.

### Install

```bash
python -m pip install -r requirements.txt
```

### Run

```bash
python mask_mapper.py /path/to/video.mp4 --out-dir mask_output
```

For large videos, reduce the display size while keeping saved masks at the original video resolution:

```bash
python mask_mapper.py /path/to/video.mp4 --scale 0.5 --out-dir mask_output
```

### Controls

| Action | Control |
| --- | --- |
| Add polygon point | Left mouse click |
| Remove last polygon point | Right mouse click |
| Close/apply polygon | Double left click, `c`, or Enter |
| Select floor mask | `f` |
| Select dry mask | `d` |
| Select wet mask | `w` |
| Clear unfinished polygon | `x` |
| Reset selected mask | `r` |
| Undo last mask edit | `u` |
| Clip dry/wet masks to the floor mask | `i` |
| Next/previous frame | `n` / `p`, or right/left arrow |
| Jump 30 frames | `]` / `[` |
| Go to frame number | `g` |
| Save masks | `s` |
| Quit | `q` or Esc |

### Output files

Saving writes these files to `--out-dir`:

- `<video>_floor_mask.png` — binary floor-area mask.
- `<video>_dry_mask.png` — binary dry-floor mask.
- `<video>_wet_mask.png` — binary wet-floor mask.
- `<video>_combined_labels.png` — single-channel label image where `0=background`, `1=floor`, `2=dry`, and `3=wet`. Wet/dry labels overwrite floor where they overlap.
- `<video>_mask_metadata.json` — video dimensions, class IDs, paths, and the last reference frame used.
