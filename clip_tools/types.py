from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from clip_tools.structs.blob_parsers import BrushEffector, parse_brush_effector


@dataclass
class LayerEntry:
    """A parsed layer image and its classification.

    type is "raster" or "vector" for renderable entries; auxiliary entries
    use values like "invalid", "clipped", "group", "mipmap", "other".
    """

    type: str
    image: np.ndarray


@dataclass
class ExternalIdEntry:
    """Describes where an external-id chunk belongs in the SQLite schema."""

    table_name: str
    column_name: str
    found: bool = False


@dataclass
class VectorPoint:
    """One control point of a parsed vector stroke.

    Modulator inputs (pressure / velocity / smooth / tilt) are read directly
    off the per-control disk record — they're baked into the saved stroke at
    draw time, not re-derived on reload. `size` and `flow` are per-control
    multipliers; lock bits in `flags` decide whether they're used as literal
    overrides (locked) or fed through the brush effector chain (unlocked).
    """

    x: float
    y: float
    pressure: float
    velocity: float = 0.0
    smooth: float = 0.0
    angle_deg: float = 0.0  # +0x50 BE f32: tilt azimuth / stroke-tangent angle, deg
    tilt_x: float = 0.0
    size: float = 1.0  # per-control SIZE multiplier (lock bit 12)
    flow: float = 1.0  # per-control FLOW multiplier (lock bit 13)
    stroke_opacity: float = 0.0  # +0x60: outline broadcast scale
    outline_dx: float = 0.0  # +0x64: outline-preview delta (selection only)
    outline_dy: float = 0.0  # +0x68: outline-preview delta (selection only)
    pattern_cache_lo: float = 0.0  # +0x6c: runtime pattern-resume scratch
    pattern_cache_hi: int = 0  # +0x70..+0x77: same scratch (8-byte qword half)
    flags: int = 0  # bit 12 = size locked, bit 13 = flow locked
    curve: Optional[tuple] = None  # (cx, cy) for CURVE-type quadratic handles

    @property
    def lock_size(self) -> bool:
        return bool(self.flags & 0x1000)

    @property
    def lock_flow(self) -> bool:
        return bool(self.flags & 0x2000)


@dataclass
class VectorStroke:
    """A single stroke inside a vector object — geometry + brush reference."""

    vtype: "object"  # VectorType enum, forward-ref to avoid circular import
    color: tuple  # (r, g, b) ints
    stroke_opacity: float
    brush_size: float
    brush_id: int
    points: list  # list[VectorPoint]
    # Per-stroke random seed read out of the blob header. Spray / ribbon
    # paths feed this to the RNG so re-renders match the saved randomness;
    # size/flow effector chains run with random gating off and ignore it.
    random_seed: int = 0


@dataclass
class VectorSample:
    """One densely-interpolated sample along a stroke's path."""

    x: float
    y: float
    pressure: float
    velocity: float = 0.0
    smooth: float = 0.0
    tilt_x: float = 0.0
    angle_deg: float = 0.0
    size: float = 1.0
    flow: float = 1.0
    stroke_opacity: float = 0.0
    flags: int = 0

    @property
    def lock_size(self) -> bool:
        return bool(self.flags & 0x1000)

    @property
    def lock_flow(self) -> bool:
        return bool(self.flags & 0x2000)


@dataclass
class BrushStyle:
    """A structured view of one row of the BrushStyle SQLite table.

    Only the fields we actually consume in rendering are named. The full CLIP
    BrushStyle row has ~69 columns; the rest (effectors, water-color params,
    spray internals we don't implement yet) are left out until they become
    relevant.
    """

    main_id: int

    # Pattern / texture linkage
    pattern_style: int  # FK into BrushPatternStyle.MainId; 0 = solid brush
    texture_pattern: int  # FK into BrushPatternImage.MainId (secondary paper texture)

    # Shape / edge
    hardness: float  # [0,1] edge softness
    thickness_base: float  # scales nominal brush_size (sub-linear, see renderer)
    anti_alias: int  # 0=off, 1=weak, 2=middle, 3=strong (CLIP's 4-level enum)

    # Flow / opacity
    flow_base: float

    # Stamp spacing
    interval_base: float
    auto_interval_type: int

    # Rotation (per-stroke)
    rotation_base: float  # in degrees
    rotation_random: float  # [0,1] random component
    rotation_effector: (
        int  # bitfield: source(s) driving rotation (tangent, tilt, pressure)
    )

    # Texture transform
    texture_scale: float
    texture_rotate: float  # in degrees
    texture_offset_x: float
    texture_offset_y: float
    texture_density_base: float

    # Compositing
    composite_mode: int  # 0=normal, 2=multiply, etc.

    # Spray
    spray_flag: int  # 0 = line-stamp brush; non-zero = spray/scatter
    spray_size_base: float
    spray_density_base: float
    spray_bias: float

    # Decoded pressure / dynamics effectors. `None` when the corresponding
    # SQLite column is empty (= no per-stamp modulation for that channel).
    size_effector: Optional[BrushEffector] = None
    opacity_effector: Optional[BrushEffector] = None
    flow_effector: Optional[BrushEffector] = None
    thickness_effector: Optional[BrushEffector] = None
    interval_effector: Optional[BrushEffector] = None
    # Spray-specific size and density effectors.
    spray_size_effector: Optional[BrushEffector] = None
    spray_density_effector: Optional[BrushEffector] = None

    # Dispatcher bit field. Bit 5 set → bend brush; bit 5 clear → spray
    # brush (which subsumes single-stamp basic brushes via num_stamps=1).
    style_flag: int = 0

    # Per-stamp rotation inside spray. Bit 7 of ``rotation_effector_in_spray``
    # gates the per-stamp jitter RNG draw; ``rotation_random_in_spray`` is
    # the jitter amplitude.
    rotation_in_spray_base: float = 0.0
    rotation_effector_in_spray: int = 0
    rotation_random_in_spray: float = 0.0

    # Per-stamp colour-jitter effectors. Each draws an RNG value when its
    # ``random`` modulator is present. We don't apply the jittered colour
    # (the renderer paints the stroke's flat colour), but we still need
    # the fields parsed so callers can reason about per-brush behaviour.
    sub_color_effector: Optional[BrushEffector] = None
    hue_change_effector: Optional[BrushEffector] = None
    saturation_change_effector: Optional[BrushEffector] = None
    value_change_effector: Optional[BrushEffector] = None
    mix_color_effector: Optional[BrushEffector] = None
    mix_alpha_effector: Optional[BrushEffector] = None

    @classmethod
    def from_row(cls, row) -> "BrushStyle":
        """Build from a pandas Series (one row of the BrushStyle DataFrame).

        Raises KeyError if a required column is missing — use this as a loud
        signal when CLIP adds/removes BrushStyle fields between versions.
        """

        def as_float(key, default=0.0):
            v = row[key]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return default
            return float(v)

        def as_int(key, default=0):
            v = row[key]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return default
            return int(v)

        def as_bytes(key):
            v = row[key]
            return v if isinstance(v, (bytes, bytearray)) else b""

        def as_effector(key):
            blob = as_bytes(key)
            return parse_brush_effector(blob if blob else None)

        return cls(
            main_id=as_int("MainId"),
            pattern_style=as_int("PatternStyle"),
            texture_pattern=as_int("TexturePattern"),
            hardness=as_float("Hardness", 1.0),
            thickness_base=as_float("ThicknessBase", 1.0),
            anti_alias=as_int("AntiAlias", 2),
            flow_base=as_float("FlowBase", 1.0),
            interval_base=as_float("IntervalBase", 1.0),
            auto_interval_type=as_int("AutoIntervalType"),
            rotation_base=as_float("RotationBase"),
            rotation_random=as_float("RotationRandom"),
            rotation_effector=as_int("RotationEffector"),
            rotation_in_spray_base=as_float("RotationInSprayBase"),
            rotation_effector_in_spray=as_int("RotationEffectorInSpray"),
            rotation_random_in_spray=as_float("RotationRandomInSpray"),
            texture_scale=as_float("TextureScale", 1.0),
            texture_rotate=as_float("TextureRotate"),
            texture_offset_x=as_float("TextureOffsetX"),
            texture_offset_y=as_float("TextureOffsetY"),
            texture_density_base=as_float("TextureDensityBase", 1.0),
            composite_mode=as_int("CompositeMode"),
            spray_flag=as_int("SprayFlag"),
            spray_size_base=as_float("SpraySizeBase"),
            spray_density_base=as_float("SprayDensityBase"),
            spray_bias=as_float("SprayBias"),
            size_effector=as_effector("SizeEffector"),
            opacity_effector=as_effector("OpacityEffector"),
            flow_effector=as_effector("FlowEffector"),
            thickness_effector=as_effector("ThicknessEffector"),
            interval_effector=as_effector("IntervalEffector"),
            spray_size_effector=as_effector("SpraySizeEffector"),
            spray_density_effector=as_effector("SprayDensityEffector"),
            style_flag=as_int("StyleFlag"),
            sub_color_effector=as_effector("SubColorEffector"),
            hue_change_effector=as_effector("HueChangeEffector"),
            saturation_change_effector=as_effector("SaturationChangeEffector"),
            value_change_effector=as_effector("ValueChangeEffector"),
            mix_color_effector=as_effector("MixColorEffector"),
            mix_alpha_effector=as_effector("MixAlphaEffector"),
        )


# Columns from the Layer SQLite schema that exist but we don't bind to a
# dataclass field — either always-zero/empty, or carry data we explicitly
# don't model yet. Anything outside this set + the LayerRecord field set
# raises in `from_row`, so a new CSP version that adds a column gets caught
# at parse time.
LAYER_RECORD_IGNORED_COLUMNS = frozenset()


_LAYER_RECORD_FIELD_TO_COLUMN = {
    "pw_id": "_PW_ID",
    "main_id": "MainId",
    "canvas_id": "CanvasId",
    "layer_uuid": "LayerUuid",
    "layer_name": "LayerName",
    "layer_type": "LayerType",
    "layer_folder": "LayerFolder",
    "layer_lock": "LayerLock",
    "layer_masking": "LayerMasking",
    "layer_visibility": "LayerVisibility",
    "layer_clip": "LayerClip",
    "layer_select": "LayerSelect",
    "layer_composite": "LayerComposite",
    "layer_opacity": "LayerOpacity",
    "layer_first_child_index": "LayerFirstChildIndex",
    "layer_next_index": "LayerNextIndex",
    "parent_layer": "ParentLayer",
    "prefix": "Prefix",
    "layer_offset_x": "LayerOffsetX",
    "layer_offset_y": "LayerOffsetY",
    "layer_render_offscr_offset_x": "LayerRenderOffscrOffsetX",
    "layer_render_offscr_offset_y": "LayerRenderOffscrOffsetY",
    "layer_mask_offset_x": "LayerMaskOffsetX",
    "layer_mask_offset_y": "LayerMaskOffsetY",
    "layer_mask_offscr_offset_x": "LayerMaskOffscrOffsetX",
    "layer_mask_offscr_offset_y": "LayerMaskOffscrOffsetY",
    "layer_render_mipmap": "LayerRenderMipmap",
    "layer_layer_mask_mipmap": "LayerLayerMaskMipmap",
    "layer_render_thumbnail": "LayerRenderThumbnail",
    "layer_layer_mask_thumbnail": "LayerLayerMaskThumbnail",
    "layer_use_palette_color": "LayerUsePaletteColor",
    "layer_noticeable_palette_color": "LayerNoticeablePaletteColor",
    "layer_palette_red": "LayerPaletteRed",
    "layer_palette_green": "LayerPaletteGreen",
    "layer_palette_blue": "LayerPaletteBlue",
    "draw_color_enable": "DrawColorEnable",
    "draw_color_main_red": "DrawColorMainRed",
    "draw_color_main_green": "DrawColorMainGreen",
    "draw_color_main_blue": "DrawColorMainBlue",
    "layer_color_type_index": "LayerColorTypeIndex",
    "layer_color_type_black_checked": "LayerColorTypeBlackChecked",
    "layer_color_type_white_checked": "LayerColorTypeWhiteChecked",
    "monochrome_fill_info": "MonochromeFillInfo",
    "draw_render_thumbnail_type": "DrawRenderThumbnailType",
    "draw_to_render_mipmap_type": "DrawToRenderMipmapType",
    "draw_to_render_offscreen_type": "DrawToRenderOffscreenType",
    "fix_offset_and_expand_type": "FixOffsetAndExpandType",
    "move_offset_and_expand_type": "MoveOffsetAndExpandType",
    "render_bound_for_layer_move_type": "RenderBoundForLayerMoveType",
    "set_render_thumbnail_info_type": "SetRenderThumbnailInfoType",
    "special_render_type": "SpecialRenderType",
    "special_ruler_manager": "SpecialRulerManager",
    "mix_sub_color_for_every_plot": "MixSubColorForEveryPlot",
    "light_table_info": "LightTableInfo",
    "vector_normal_stroke_index": "VectorNormalStrokeIndex",
    "vector_normal_fill_index": "VectorNormalFillIndex",
    "vector_normal_balloon_index": "VectorNormalBalloonIndex",
    "vector_normal_type": "VectorNormalType",
    "text_layer_type": "TextLayerType",
    "text_layer_attributes": "TextLayerAttributes",
    "text_layer_add_attributes_v01": "TextLayerAddAttributesV01",
    "text_layer_string": "TextLayerString",
    "text_layer_attributes_version": "TextLayerAttributesVersion",
    "resizable_image_info": "ResizableImageInfo",
    "camera_2d_resizable_image_info": "Camera2DResizableImageInfo",
    "camera_2d_apply_transform": "Camera2DApplyTransform",
    "camera_2d_original_frame_center_x": "Camera2DOriginalFrameCenterX",
    "camera_2d_original_frame_center_y": "Camera2DOriginalFrameCenterY",
    "animation_folder": "AnimationFolder",
    "animation_cel_current_uuid": "AnimationCelCurrentUuid",
    "timeline_layer_keyframe_enabled": "TimeLineLayerKeyFrameEnabled",
    "timeline_render_fix_aspect_ratio": "TimeLineRenderFixAspectRatio",
    "material_content_type": "MaterialContentType",
    "ruler_range": "RulerRange",
    "guide_move": "GuideMove",
}

LAYER_RECORD_KNOWN_COLUMNS = (
    frozenset(_LAYER_RECORD_FIELD_TO_COLUMN.values()) | LAYER_RECORD_IGNORED_COLUMNS
)


@dataclass
class LayerRecord:
    """A typed view of one row of the Layer SQLite table.

    Field set covers every column observed populated in real .clip files
    (test fixtures + production wn_* samples). Unpopulated columns from the
    raw schema are intentionally omitted — add them as they become relevant
    rather than carrying an exhaustive 200-field surface.
    """

    # --- Identity ---
    pw_id: int  # `_PW_ID` — primary-key index added by augment_layer_df
    main_id: int
    canvas_id: int
    layer_uuid: str
    layer_name: str

    # --- Type / flags ---
    layer_type: int  # see LayerKind enum
    layer_folder: int  # see LayerFolderBit
    layer_lock: int  # see LayerLockBit
    layer_masking: int
    layer_visibility: int
    layer_clip: int
    layer_select: int
    layer_composite: int  # see LayerComposite enum
    layer_opacity: int  # 0..256

    # --- Tree pointers ---
    layer_first_child_index: int
    layer_next_index: int
    parent_layer: int  # injected by augment_layer_df
    prefix: List[str]  # injected by augment_layer_df

    # --- Offsets ---
    layer_offset_x: int
    layer_offset_y: int
    layer_render_offscr_offset_x: int
    layer_render_offscr_offset_y: int
    layer_mask_offset_x: int
    layer_mask_offset_y: int
    layer_mask_offscr_offset_x: int
    layer_mask_offscr_offset_y: int

    # --- Mipmap / thumbnail FKs ---
    layer_render_mipmap: int
    layer_layer_mask_mipmap: int
    layer_render_thumbnail: int
    layer_layer_mask_thumbnail: int

    # --- Palette / sticker color ---
    layer_use_palette_color: int
    layer_noticeable_palette_color: int
    layer_palette_red: int
    layer_palette_green: int
    layer_palette_blue: int

    # --- Fill color (correction layers) ---
    draw_color_enable: Optional[float] = None
    draw_color_main_red: Optional[float] = None  # u32 channel value (0..2^32-1)
    draw_color_main_green: Optional[float] = None
    draw_color_main_blue: Optional[float] = None

    # --- Color-type / monochrome (manga ink/tone) ---
    layer_color_type_index: Optional[float] = None
    layer_color_type_black_checked: Optional[float] = None
    layer_color_type_white_checked: Optional[float] = None
    monochrome_fill_info: Optional[bytes] = None

    # --- Render pipeline / dirty flags ---
    draw_render_thumbnail_type: Optional[float] = None
    draw_to_render_mipmap_type: Optional[float] = None
    draw_to_render_offscreen_type: Optional[float] = None
    fix_offset_and_expand_type: Optional[float] = None
    move_offset_and_expand_type: Optional[float] = None
    render_bound_for_layer_move_type: Optional[float] = None
    set_render_thumbnail_info_type: Optional[float] = None
    special_render_type: Optional[float] = None
    special_ruler_manager: Optional[float] = None
    mix_sub_color_for_every_plot: Optional[float] = None

    # --- Light table (per-folder reference image) ---
    light_table_info: Optional[bytes] = None

    # --- Vector layer state ---
    vector_normal_stroke_index: Optional[float] = None
    vector_normal_fill_index: Optional[float] = None
    vector_normal_balloon_index: Optional[float] = None
    vector_normal_type: Optional[float] = None

    # --- Text layer ---
    text_layer_type: Optional[float] = None
    text_layer_attributes: Optional[bytes] = None
    text_layer_add_attributes_v01: Optional[bytes] = None
    text_layer_string: Optional[bytes] = None
    text_layer_attributes_version: Optional[float] = None

    # --- Resizable / Camera 2D ---
    resizable_image_info: Optional[bytes] = None
    camera_2d_resizable_image_info: Optional[bytes] = None
    camera_2d_apply_transform: Optional[float] = None
    camera_2d_original_frame_center_x: Optional[float] = None
    camera_2d_original_frame_center_y: Optional[float] = None

    # --- Animation ---
    animation_folder: Optional[float] = None
    animation_cel_current_uuid: Optional[str] = None
    timeline_layer_keyframe_enabled: Optional[float] = None
    timeline_render_fix_aspect_ratio: Optional[float] = None

    # --- Misc ---
    material_content_type: Optional[float] = None
    ruler_range: Optional[float] = None
    guide_move: Optional[float] = None

    @classmethod
    def from_row(cls, row) -> "LayerRecord":
        """Build from a pandas Series (one row of the Layer DataFrame).

        Loud KeyError on missing required columns; silent default for optional
        float/bytes columns (CLIP omits them when the layer doesn't use that
        feature, so they show up as NaN rather than 0).

        Loud ValueError on unrecognized populated columns — if CSP adds a new
        Layer column we don't know about, we want to find out at parse time
        rather than silently drop the data. Add the field to the dataclass (or
        to LAYER_RECORD_IGNORED_COLUMNS if it's noise).
        """
        unknown = []
        for col in row.index:
            if col in LAYER_RECORD_KNOWN_COLUMNS:
                continue
            v = row[col]
            if v is None:
                continue
            if isinstance(v, float) and np.isnan(v):
                continue
            if isinstance(v, (bytes, bytearray)) and len(v) == 0:
                continue
            unknown.append(col)
        if unknown:
            raise ValueError(
                f"LayerRecord: unrecognized populated Layer column(s): {unknown}. "
                f"Add to the dataclass or to LAYER_RECORD_IGNORED_COLUMNS."
            )

        def as_float(key, default=0.0):
            if key not in row:
                return default
            v = row[key]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return default
            return float(v)

        def as_int(key, default=0):
            if key not in row:
                return default
            v = row[key]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return default
            return int(v)

        def as_str(key, default=""):
            if key not in row:
                return default
            v = row[key]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return default
            if isinstance(v, (bytes, bytearray)):
                return v.decode("utf-8", errors="replace")
            return str(v)

        def opt_float(key):
            if key not in row:
                return None
            v = row[key]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return None
            return float(v)

        def opt_str(key):
            if key not in row:
                return None
            v = row[key]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return None
            if isinstance(v, (bytes, bytearray)):
                return v.decode("utf-8", errors="replace")
            return str(v)

        def opt_bytes(key):
            if key not in row:
                return None
            v = row[key]
            if isinstance(v, (bytes, bytearray)) and len(v) > 0:
                return bytes(v)
            return None

        def as_list(key):
            if key not in row:
                return []
            v = row[key]
            return list(v) if isinstance(v, (list, tuple)) else []

        return cls(
            pw_id=as_int("_PW_ID"),
            main_id=as_int("MainId"),
            canvas_id=as_int("CanvasId"),
            layer_uuid=as_str("LayerUuid"),
            layer_name=as_str("LayerName"),
            layer_type=as_int("LayerType"),
            layer_folder=as_int("LayerFolder"),
            layer_lock=as_int("LayerLock"),
            layer_masking=as_int("LayerMasking"),
            layer_visibility=as_int("LayerVisibility"),
            layer_clip=as_int("LayerClip"),
            layer_select=as_int("LayerSelect"),
            layer_composite=as_int("LayerComposite"),
            layer_opacity=as_int("LayerOpacity", 256),
            layer_first_child_index=as_int("LayerFirstChildIndex"),
            layer_next_index=as_int("LayerNextIndex"),
            parent_layer=as_int("ParentLayer"),
            prefix=as_list("Prefix"),
            layer_offset_x=as_int("LayerOffsetX"),
            layer_offset_y=as_int("LayerOffsetY"),
            layer_render_offscr_offset_x=as_int("LayerRenderOffscrOffsetX"),
            layer_render_offscr_offset_y=as_int("LayerRenderOffscrOffsetY"),
            layer_mask_offset_x=as_int("LayerMaskOffsetX"),
            layer_mask_offset_y=as_int("LayerMaskOffsetY"),
            layer_mask_offscr_offset_x=as_int("LayerMaskOffscrOffsetX"),
            layer_mask_offscr_offset_y=as_int("LayerMaskOffscrOffsetY"),
            layer_render_mipmap=as_int("LayerRenderMipmap"),
            layer_layer_mask_mipmap=as_int("LayerLayerMaskMipmap"),
            layer_render_thumbnail=as_int("LayerRenderThumbnail"),
            layer_layer_mask_thumbnail=as_int("LayerLayerMaskThumbnail"),
            layer_use_palette_color=as_int("LayerUsePaletteColor"),
            layer_noticeable_palette_color=as_int("LayerNoticeablePaletteColor"),
            layer_palette_red=as_int("LayerPaletteRed"),
            layer_palette_green=as_int("LayerPaletteGreen"),
            layer_palette_blue=as_int("LayerPaletteBlue"),
            draw_color_enable=opt_float("DrawColorEnable"),
            draw_color_main_red=opt_float("DrawColorMainRed"),
            draw_color_main_green=opt_float("DrawColorMainGreen"),
            draw_color_main_blue=opt_float("DrawColorMainBlue"),
            layer_color_type_index=opt_float("LayerColorTypeIndex"),
            layer_color_type_black_checked=opt_float("LayerColorTypeBlackChecked"),
            layer_color_type_white_checked=opt_float("LayerColorTypeWhiteChecked"),
            monochrome_fill_info=opt_bytes("MonochromeFillInfo"),
            draw_render_thumbnail_type=opt_float("DrawRenderThumbnailType"),
            draw_to_render_mipmap_type=opt_float("DrawToRenderMipmapType"),
            draw_to_render_offscreen_type=opt_float("DrawToRenderOffscreenType"),
            fix_offset_and_expand_type=opt_float("FixOffsetAndExpandType"),
            move_offset_and_expand_type=opt_float("MoveOffsetAndExpandType"),
            render_bound_for_layer_move_type=opt_float("RenderBoundForLayerMoveType"),
            set_render_thumbnail_info_type=opt_float("SetRenderThumbnailInfoType"),
            special_render_type=opt_float("SpecialRenderType"),
            special_ruler_manager=opt_float("SpecialRulerManager"),
            mix_sub_color_for_every_plot=opt_float("MixSubColorForEveryPlot"),
            light_table_info=opt_bytes("LightTableInfo"),
            vector_normal_stroke_index=opt_float("VectorNormalStrokeIndex"),
            vector_normal_fill_index=opt_float("VectorNormalFillIndex"),
            vector_normal_balloon_index=opt_float("VectorNormalBalloonIndex"),
            vector_normal_type=opt_float("VectorNormalType"),
            text_layer_type=opt_float("TextLayerType"),
            text_layer_attributes=opt_bytes("TextLayerAttributes"),
            text_layer_add_attributes_v01=opt_bytes("TextLayerAddAttributesV01"),
            text_layer_string=opt_bytes("TextLayerString"),
            text_layer_attributes_version=opt_float("TextLayerAttributesVersion"),
            resizable_image_info=opt_bytes("ResizableImageInfo"),
            camera_2d_resizable_image_info=opt_bytes("Camera2DResizableImageInfo"),
            camera_2d_apply_transform=opt_float("Camera2DApplyTransform"),
            camera_2d_original_frame_center_x=opt_float("Camera2DOriginalFrameCenterX"),
            camera_2d_original_frame_center_y=opt_float("Camera2DOriginalFrameCenterY"),
            animation_folder=opt_float("AnimationFolder"),
            animation_cel_current_uuid=opt_str("AnimationCelCurrentUuid"),
            timeline_layer_keyframe_enabled=opt_float("TimeLineLayerKeyFrameEnabled"),
            timeline_render_fix_aspect_ratio=opt_float("TimeLineRenderFixAspectRatio"),
            material_content_type=opt_float("MaterialContentType"),
            ruler_range=opt_float("RulerRange"),
            guide_move=opt_float("GuideMove"),
        )
