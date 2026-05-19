from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

MASK_CLASSES = {
    "floor": {"id": 1, "color": (255, 180, 0), "key": "1"},
    "dry": {"id": 2, "color": (0, 220, 0), "key": "2"},
    "wet": {"id": 3, "color": (255, 0, 255), "key": "3"},
    "obstruction": {"id": 4, "color": (0, 165, 255), "key": "4"},
}

CONDITION_LABELS = ("dry", "wet", "obstruction")

HELP_TEXT = """
Wet/dry floor brush mask mapper controls
----------------------------------------
Mouse:
  Left mouse drag       paint or erase on the selected mask
                        Dry/wet/obstruction painting is limited to the floor mask
                        Dry/wet/obstruction labels apply only to the selected frame

Brush mode:
  d                     draw/paint mode
  e                     erase mode
  + / =                 increase brush size, or hex cell size when hex view is on
  - / _                 decrease brush size, or hex cell size when hex view is on

Mask selection:
  1 / f                 select floor-area mask
  2                     select dry-floor mask
  3 / w                 select wet-floor mask
  4 / o                 select obstruction mask

Hex visualization:
  Space                 toggle averaged-color hexagon view
  Enter                 run dry-to-wet OKLab hex analysis (0=red, 50=yellow, 100=green)
                        Hexes too far from dry-wet colour line show checkers
  1 / 2 / 3 / 4         show hexagons only for the selected mask layer

Editing:
  r                     reset selected mask
  u                     undo last brush stroke or reset
  i                     clip all frame dry/wet/obstruction masks to the floor-area mask

Video navigation:
  Horizontal slider     choose the video frame to label against
  n / Right arrow       next frame
  p / Left arrow        previous frame
  ]                     jump forward 30 frames
  [                     jump backward 30 frames
  g                     jump to a frame number entered in the terminal

Output:
  s                     save mask images
  h                     print this help in the terminal
  q / Esc               quit
""".strip()


@dataclass
class MaskSnapshot:
    label: str
    frame_index: int | None
    mask: np.ndarray


@dataclass
class HexWetnessModel:
    dry_oklab: np.ndarray
    wet_oklab: np.ndarray


@dataclass
class HexCell:
    index: int
    polygon: np.ndarray
    bounds: tuple[int, int, int, int]
    cell_pixels: np.ndarray


@dataclass
class EditorState:
    source: str
    source_stem: str
    video_path: Path
    out_dir: Path
    frame_count: int
    width: int
    height: int
    fps: float
    scale: float
    alpha: float
    analysis_max_distance: float
    analysis_time_window: float
    analysis_sample_interval: float
    live_analysis_interval: float
    live_target_fps: float | None
    stream_reconnect_delay: float
    rtsp_transport: str
    is_live_source: bool
    show_hex_values: bool
    average_wetness_only: bool
    use_opencl: bool
    frame_index: int = 0
    selected: str = "floor"
    brush_mode: str = "draw"
    brush_size: int = 20
    hex_enabled: bool = False
    hex_cell_size: int = 40
    mask_revision: int = 0
    analysis_enabled: bool = False
    analysis_revision: int = 0
    wetness_models: dict[int, HexWetnessModel] = field(default_factory=dict)
    hex_cells_cache_key: tuple[int, int, int] | None = None
    hex_cells: list[HexCell] = field(default_factory=list)
    floor_hex_cells_cache_key: tuple[int, int, int, int, int] | None = None
    floor_hex_cells: list[HexCell] = field(default_factory=list)
    display_polygons_cache_key: tuple[int, int, int, float] | None = None
    display_polygons: dict[int, np.ndarray] = field(default_factory=dict)
    is_painting: bool = False
    cursor: tuple[int, int] | None = None
    last_paint_point: tuple[int, int] | None = None
    render_dirty: bool = True
    masks: dict[str, np.ndarray] = field(default_factory=dict)
    display_masks: dict[str, np.ndarray] = field(default_factory=dict)
    condition_masks_by_frame: dict[int, dict[str, np.ndarray]] = field(default_factory=dict)
    annotation_frames_by_index: dict[int, np.ndarray] = field(default_factory=dict)
    current_condition_frame: int = 0
    masks_dirty: bool = True
    history: list[list[MaskSnapshot]] = field(default_factory=list)
    dirty: bool = False

    @property
    def display_size(self) -> tuple[int, int]:
        return (max(1, int(self.width * self.scale)), max(1, int(self.height * self.scale)))


@dataclass
class FrameView:
    frame: np.ndarray
    display_frame: np.ndarray
    base_overlay: np.ndarray | None = None
    overlay: np.ndarray | None = None
    hex_overlay: np.ndarray | None = None
    hex_mask: np.ndarray | None = None
    hex_cache_key: tuple[int, str, int, int, int, int] | None = None
    analysis_sample_frame: np.ndarray | None = None
    analysis_average_wetness: float | None = None
    suppress_trackbar_callback: bool = False
