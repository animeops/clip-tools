"""Typed dataclasses for every populated row of every CLIP SQLite table.

One `from_row(row)` per record class, mirroring `LayerRecord.from_row` /
`BrushStyle.from_row` in `types.py`. Each class binds the columns we know
about; columns we explicitly ignore go into `<RECORD>_IGNORED_COLUMNS`.
Any populated column outside the known + ignored set raises in
`from_row` so a future CSP version that adds a column gets caught at parse
time.

Row dataclasses don't parse their own bytes blobs — those go through the
parsers in `clip_tools/structs/`. The dataclass holds the raw bytes (or
parsed sub-dataclass for blobs we already decode) and lets the consumer
decide whether to lazy-parse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from clip_tools.structs.blob_parsers import (
    SmallObjectFlag,
    format_uuid,
    parse_brush_effector,
    parse_comp_layer_info,
    parse_effector_control_points,
    parse_small_object_flag,
    parse_track_value_map,
    parse_vanish_point_guide,
)
from clip_tools.types import BrushStyle, LayerRecord


def _row_get(row, key, default=None):
    if key not in row:
        return default
    v = row[key]
    if v is None:
        return default
    if isinstance(v, float) and np.isnan(v):
        return default
    return v


def as_int(row, key, default=0):
    v = _row_get(row, key)
    return default if v is None else int(v)


def as_float(row, key, default=0.0):
    v = _row_get(row, key)
    return default if v is None else float(v)


def as_str(row, key, default=""):
    v = _row_get(row, key)
    if v is None:
        return default
    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", errors="replace")
    return str(v)


def opt_int(row, key) -> Optional[int]:
    v = _row_get(row, key)
    return None if v is None else int(v)


def opt_float(row, key) -> Optional[float]:
    v = _row_get(row, key)
    return None if v is None else float(v)


def opt_str(row, key) -> Optional[str]:
    v = _row_get(row, key)
    if v is None:
        return None
    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", errors="replace")
    return str(v)


def opt_bytes(row, key) -> Optional[bytes]:
    if key not in row:
        return None
    v = row[key]
    if isinstance(v, (bytes, bytearray)) and len(v) > 0:
        return bytes(v)
    return None


def check_unknown(row, known_cols, ignored_cols, record_name: str) -> None:
    """Raise if any populated column on `row` is outside known + ignored.

    Mirrors the strictness check in `LayerRecord.from_row`. Ensures CSP
    schema additions surface as a clear error rather than silent data loss.
    """
    unknown = []
    for col in row.index:
        if col in known_cols or col in ignored_cols:
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
            f"{record_name}: unrecognized populated column(s): {unknown}. "
            f"Add to the dataclass or to {record_name.upper()}_IGNORED_COLUMNS."
        )


# ---------------------------------------------------------------------------
# Canvas / Project (singletons)
# ---------------------------------------------------------------------------


CANVAS_IGNORED_COLUMNS = frozenset(
    {
        # Default-color / channel-format settings used at file creation;
        # don't drive runtime rendering.
        "CanvasDefaultChannelOrder",
        "CanvasDefaultColorBlackChecked",
        "CanvasDefaultColorTypeIndex",
        "CanvasDefaultColorWhiteChecked",
        "CanvasDefaultToneLine",
        "CanvasDoSimulateColor",
        "CanvasDoublePage",
        "CanvasRenderMipmapForceSaved",
        # UI-state: grid, light table, onion-skin, work-time tracker.
        "ShowGrid",
        "GridDitch",
        "GridDitchUnit",
        "GridDivide",
        "GridOriginType",
        "GridOriginX",
        "GridOriginXUnit",
        "GridOriginY",
        "GridOriginYUnit",
        "LightTableRotateCenterX",
        "LightTableRotateCenterY",
        "LightTableEnableWhenSave",
        "OnionSkinStatus",
        "OnionSkinSettingRegisted",
        "OnionSkinPrevDisplayCount",
        "OnionSkinNextDisplayCount",
        "OnionSkinUseOpacity",
        "OnionSkinStartOpacity",
        "OnionSkinStepOpaciy",  # CSP misspelling — column name is verbatim
        "OnionSkinUseColor",
        "OnionSkinHalfColor",
        "OnionSkinFrontColorRed",
        "OnionSkinFrontColorGreen",
        "OnionSkinFrontColorBlue",
        "OnionSkinBehindColorRed",
        "OnionSkinBehindColorGreen",
        "OnionSkinBehindColorBlue",
        "CanvasWorkTime",
        "TimeLineFrameDisplay",
        "TimeLineEditMode",
        "TimeLineResizeQuality",
        # Brush-pattern bookkeeping pointers we don't follow yet.
        "BrushStyleReadProtect070",
        # Comic-story metadata mirrored from Project — not used for rendering.
        "ComicStoryStoryName",
        "ComicStoryUseStoryIndex",
        "ComicStoryStoryIndex",
        "ComicStorySubTitle",
        "ComicStoryStoryPosition",
        "ComicStoryAuthorName",
        "ComicStoryAuthorPosition",
        "ComicStoryUsePageNumber",
        "ComicStoryPageNumberStart",
        "ComicStoryPageNumberPosition",
        "ComicStoryNombreStart",
        "ComicStoryNombreFont",
        "ComicStoryNombreFontSize",
        "ComicStoryNombreFontPostScriptName",
        "ComicStoryUseShownNombre",
        "ComicStoryNombrePosition",
        "ComicStoryNombrePrefix",
        "ComicStoryNombreSuffix",
        "ComicStoryUseHiddenNombre",
        "ComicStoryNombreColor",
        "ComicStoryNombreUseEdge",
        "ComicStoryNombreEdgeWidth",
        "ComicStoryNombreEdgeUnit",
        "ComicStoryYPageNombreOffset",
        "ComicStoryYPageNombreOffsetUnit",
        "ComicStoryXPageNombreOffset",
        "ComicStoryXPageNombreOffsetUnit",
        "ComicBindPosition",
        "ComicHasComicCover",
        "ComicHasDoubleCover",
        "ComicPageIndex",
        "ComicIsLeftPage",
    }
)


@dataclass
class CanvasRecord:
    """A row of the `Canvas` table — global canvas geometry + DPI."""

    pw_id: int
    main_id: int
    canvas_unit: int  # see CanvasUnit enum
    canvas_width: float
    canvas_height: float
    canvas_resolution: float
    canvas_channel_bytes: int  # 1, 2, or 4 (8/16/32 bpc)
    canvas_root_folder: int  # FK into Layer.MainId — root of layer tree
    canvas_current_layer: int  # FK into Layer.MainId — UI selection
    brush_style_manager: int  # FK into BrushStyleManager.MainId
    canvas_3d_model_data_loader_index: int

    @classmethod
    def from_row(cls, row) -> "CanvasRecord":
        known = {
            "_PW_ID",
            "MainId",
            "CanvasUnit",
            "CanvasWidth",
            "CanvasHeight",
            "CanvasResolution",
            "CanvasChannelBytes",
            "CanvasRootFolder",
            "CanvasCurrentLayer",
            "BrushStyleManager",
            "Canvas3DModelDataLoaderIndex",
        }
        check_unknown(row, known, CANVAS_IGNORED_COLUMNS, "CanvasRecord")
        return cls(
            pw_id=as_int(row, "_PW_ID"),
            main_id=as_int(row, "MainId"),
            canvas_unit=as_int(row, "CanvasUnit"),
            canvas_width=as_float(row, "CanvasWidth"),
            canvas_height=as_float(row, "CanvasHeight"),
            canvas_resolution=as_float(row, "CanvasResolution"),
            canvas_channel_bytes=as_int(row, "CanvasChannelBytes"),
            canvas_root_folder=as_int(row, "CanvasRootFolder"),
            canvas_current_layer=as_int(row, "CanvasCurrentLayer"),
            brush_style_manager=as_int(row, "BrushStyleManager"),
            canvas_3d_model_data_loader_index=as_int(
                row, "Canvas3DModelDataLoaderIndex"
            ),
        )


PROJECT_IGNORED_COLUMNS = frozenset(
    {
        # Comic-book story metadata (titles, page numbering, nombre style).
        # Only relevant for manga export workflows we don't currently support.
        "ComicStoryAuthorName",
        "ComicStoryAuthorPosition",
        "ComicStoryNombreColor",
        "ComicStoryNombreEdgeUnit",
        "ComicStoryNombreEdgeWidth",
        "ComicStoryNombreFont",
        "ComicStoryNombreFontPostScriptName",
        "ComicStoryNombreFontSize",
        "ComicStoryNombrePosition",
        "ComicStoryNombrePrefix",
        "ComicStoryNombreStart",
        "ComicStoryNombreSuffix",
        "ComicStoryNombreUseEdge",
        "ComicStoryPageNumberPosition",
        "ComicStoryPageNumberStart",
        "ComicStoryStoryIndex",
        "ComicStoryStoryName",
        "ComicStoryStoryPosition",
        "ComicStorySubTitle",
        "ComicStoryUseHiddenNombre",
        "ComicStoryUsePageNumber",
        "ComicStoryUseShownNombre",
        "ComicStoryUseStoryIndex",
        "ComicStoryXPageNombreOffset",
        "ComicStoryXPageNombreOffsetUnit",
        "ComicStoryYPageNombreOffset",
        "ComicStoryYPageNombreOffsetUnit",
        # Default-page settings used by the New File dialog. Stored on the
        # project but unrelated to the canvas the file actually contains.
        "DefaultPageBlackChecked",
        "DefaultPageCelIsUserTemplate",
        "DefaultPageCelTemplateName",
        "DefaultPageCelTemplatePath",
        "DefaultPageCelTemplatePath2",
        "DefaultPageCelTemplateUUID",
        "DefaultPageCelUseTemplate",
        "DefaultPageChannelBytes",
        "DefaultPageChannelOrder",
        "DefaultPageCheckBookBinding",
        "DefaultPageColorType",
        "DefaultPageDoublePage",
        "DefaultPageHeight",
        "DefaultPageIsUserTemplate",
        "DefaultPagePaperBlue",
        "DefaultPagePaperGreen",
        "DefaultPagePaperRed",
        "DefaultPagePresetID",
        "DefaultPageRecordTimeLapse",
        "DefaultPageResolution",
        "DefaultPageSettingType",
        "DefaultPageTemplateName",
        "DefaultPageTemplatePath",
        "DefaultPageTemplatePath2",
        "DefaultPageTemplateUUID",
        "DefaultPageToneLine",
        "DefaultPageUnit",
        "DefaultPageUseCropFrame",
        "DefaultPageUseDefaultCoverInfo",
        "DefaultPageUsePaper",
        "DefaultPageUseTemplate",
        "DefaultPageWhiteChecked",
        "DefaultPageWidth",
    }
)


@dataclass
class ProjectRecord:
    """A row of the `Project` table — top-level project metadata."""

    pw_id: int
    main_id: int
    project_name: str
    project_internal_version: str  # observed e.g. "1.1.0"
    project_canvas: int  # FK into Canvas.MainId
    project_item_bank: int  # FK into CanvasItemBank.MainId
    project_cut_bank: int  # FK into AnimationCutBank.MainId
    project_root_canvas_node: int
    project_layer_comp_manager: Optional[int]  # only present when LayerComp exists

    @classmethod
    def from_row(cls, row) -> "ProjectRecord":
        known = {
            "_PW_ID",
            "MainId",
            "ProjectName",
            "ProjectInternalVersion",
            "ProjectCanvas",
            "ProjectItemBank",
            "ProjectCutBank",
            "ProjectRootCanvasNode",
            "ProjectLayerCompManager",
        }
        check_unknown(row, known, PROJECT_IGNORED_COLUMNS, "ProjectRecord")
        return cls(
            pw_id=as_int(row, "_PW_ID"),
            main_id=as_int(row, "MainId"),
            project_name=as_str(row, "ProjectName"),
            project_internal_version=as_str(row, "ProjectInternalVersion"),
            project_canvas=as_int(row, "ProjectCanvas"),
            project_item_bank=as_int(row, "ProjectItemBank"),
            project_cut_bank=as_int(row, "ProjectCutBank"),
            project_root_canvas_node=as_int(row, "ProjectRootCanvasNode"),
            project_layer_comp_manager=opt_int(row, "ProjectLayerCompManager"),
        )


# ---------------------------------------------------------------------------
# Offscreen / Mipmap
# ---------------------------------------------------------------------------


@dataclass
class OffscreenRecord:
    """A row of the `Offscreen` table — one chunked raster image."""

    pw_id: int
    main_id: int
    canvas_id: int
    layer_id: int
    next_index: int
    block_data: Optional[bytes]  # 32-byte external-id reference
    attribute: Optional[bytes]  # parse via `process_offscreen_attributes`
    flag: int

    @classmethod
    def from_row(cls, row) -> "OffscreenRecord":
        known = {
            "_PW_ID",
            "MainId",
            "CanvasId",
            "LayerId",
            "NextIndex",
            "BlockData",
            "Attribute",
            "Flag",
        }
        check_unknown(row, known, frozenset(), "OffscreenRecord")
        return cls(
            pw_id=as_int(row, "_PW_ID"),
            main_id=as_int(row, "MainId"),
            canvas_id=as_int(row, "CanvasId"),
            layer_id=as_int(row, "LayerId"),
            next_index=as_int(row, "NextIndex"),
            block_data=opt_bytes(row, "BlockData"),
            attribute=opt_bytes(row, "Attribute"),
            flag=as_int(row, "Flag"),
        )


@dataclass
class MipmapRecord:
    """A row of the `Mipmap` table — head of a mipmap level chain."""

    pw_id: int
    main_id: int
    canvas_id: int
    base_mipmap_info: int
    mipmap_count: int  # number of MipmapInfo rows reachable via NextIndex chain
    layer_id: int
    next_index: int
    flag: int

    @classmethod
    def from_row(cls, row) -> "MipmapRecord":
        known = {
            "_PW_ID",
            "MainId",
            "CanvasId",
            "BaseMipmapInfo",
            "MipmapCount",
            "LayerId",
            "NextIndex",
            "Flag",
        }
        check_unknown(row, known, frozenset(), "MipmapRecord")
        return cls(
            pw_id=as_int(row, "_PW_ID"),
            main_id=as_int(row, "MainId"),
            canvas_id=as_int(row, "CanvasId"),
            base_mipmap_info=as_int(row, "BaseMipmapInfo"),
            mipmap_count=as_int(row, "MipmapCount"),
            layer_id=as_int(row, "LayerId"),
            next_index=as_int(row, "NextIndex"),
            flag=as_int(row, "Flag"),
        )


@dataclass
class MipmapInfoRecord:
    """A row of the `MipmapInfo` table — one mipmap level."""

    pw_id: int
    main_id: int
    canvas_id: int
    layer_id: int
    this_scale: float
    offscreen: int  # FK into Offscreen.MainId
    next_index: int  # 0 = end of chain

    @classmethod
    def from_row(cls, row) -> "MipmapInfoRecord":
        known = {
            "_PW_ID",
            "MainId",
            "CanvasId",
            "LayerId",
            "ThisScale",
            "Offscreen",
            "NextIndex",
        }
        check_unknown(row, known, frozenset(), "MipmapInfoRecord")
        return cls(
            pw_id=as_int(row, "_PW_ID"),
            main_id=as_int(row, "MainId"),
            canvas_id=as_int(row, "CanvasId"),
            layer_id=as_int(row, "LayerId"),
            this_scale=as_float(row, "ThisScale"),
            offscreen=as_int(row, "Offscreen"),
            next_index=as_int(row, "NextIndex"),
        )


LAYER_THUMBNAIL_IGNORED_COLUMNS = frozenset(
    {
        # Per-size dirty flags driving CSP's thumbnail re-rasterization. The
        # ones with `…1` suffixes are duplicates / version-bumped columns.
        "ThumbnailSmallerNeedRefresh",
        "ThumbnailSmallerNeedRefresh1",
        "ThumbnailSmallNeedRefresh",
        "ThumbnailSmallNeedRefresh1",
        "ThumbnailMiddleNeedRefresh",
        "ThumbnailMiddleNeedRefresh1",
        "ThumbnailMiddle2xNeedRefresh",
        "ThumbnailMiddle2xNeedRefresh1",
        "ThumbnailLargeNeedRefresh",
        "ThumbnailLargeNeedRefresh1",
        "ThumbnailLargerNeedRefresh",
        "ThumbnailLargerNeedRefresh1",
        "ThumbnailLarger2xNeedRefresh",
        "ThumbnailLarger2xNeedRefresh1",
        # Drawing-mode / color-type / preview overrides for the thumbnail
        # rasterizer. None of these affect compositing.
        "ThumbnailDrewMode",
        "ThumbnailDrewUseCanvasAspect0",
        "ThumbnailDrewUseCanvasAspect1",
        "ThumbnailFixMode",
        "ThumbnailColorTypeBlack",
        "ThumbnailColorTypeIndex",
        "ThumbnailColorTypeWhite",
        "ThumbnailMainColorBlue",
        "ThumbnailMainColorGreen",
        "ThumbnailMainColorRed",
        "ThumbnailSubColorBlue",
        "ThumbnailSubColorGreen",
        "ThumbnailSubColorRed",
        "ThumbnailUseDrawColor",
        "ThumbnailPrewviewColorTypeAlpha",  # CSP misspelling — verbatim
        "ThumbnailPrewviewColorTypeBlack",
        "ThumbnailPrewviewColorTypeImage",
        "ThumbnailPrewviewColorTypeIndex",
        "ThumbnailPrewviewColorTypeOpacity",
        "ThumbnailPrewviewColorTypeWhite",
        "ThumbnailPrewviewMaskBinarize",
        "ThumbnailPrewviewMaskThreshold",
    }
)


@dataclass
class LayerThumbnailRecord:
    """A row of the `LayerThumbnail` table — a per-layer thumbnail."""

    pw_id: int
    main_id: int
    canvas_id: int
    layer_id: int
    thumbnail_canvas_width: float
    thumbnail_canvas_height: float
    thumbnail_offscreen: int  # FK into Offscreen.MainId

    @classmethod
    def from_row(cls, row) -> "LayerThumbnailRecord":
        known = {
            "_PW_ID",
            "MainId",
            "CanvasId",
            "LayerId",
            "ThumbnailCanvasWidth",
            "ThumbnailCanvasHeight",
            "ThumbnailOffscreen",
        }
        check_unknown(
            row, known, LAYER_THUMBNAIL_IGNORED_COLUMNS, "LayerThumbnailRecord"
        )
        return cls(
            pw_id=as_int(row, "_PW_ID"),
            main_id=as_int(row, "MainId"),
            canvas_id=as_int(row, "CanvasId"),
            layer_id=as_int(row, "LayerId"),
            thumbnail_canvas_width=as_float(row, "ThumbnailCanvasWidth"),
            thumbnail_canvas_height=as_float(row, "ThumbnailCanvasHeight"),
            thumbnail_offscreen=as_int(row, "ThumbnailOffscreen"),
        )


# ---------------------------------------------------------------------------
# Vector / Brush
# ---------------------------------------------------------------------------


@dataclass
class VectorObjectRecord:
    """A row of the `VectorObjectList` table — pointer to a vector chunk."""

    pw_id: int
    main_id: int
    canvas_id: int
    layer_id: int
    next_index: int
    vector_data: Optional[bytes]  # 32-byte external-id reference

    @classmethod
    def from_row(cls, row) -> "VectorObjectRecord":
        known = {
            "_PW_ID",
            "MainId",
            "CanvasId",
            "LayerId",
            "NextIndex",
            "VectorData",
        }
        check_unknown(row, known, frozenset(), "VectorObjectRecord")
        return cls(
            pw_id=as_int(row, "_PW_ID"),
            main_id=as_int(row, "MainId"),
            canvas_id=as_int(row, "CanvasId"),
            layer_id=as_int(row, "LayerId"),
            next_index=as_int(row, "NextIndex"),
            vector_data=opt_bytes(row, "VectorData"),
        )


@dataclass
class BrushPatternImageRecord:
    pw_id: int
    main_id: int
    canvas_id: int
    name: Optional[str]
    uuid: Optional[str]
    mipmap: int
    next_index: int

    @classmethod
    def from_row(cls, row) -> "BrushPatternImageRecord":
        known = {
            "_PW_ID",
            "MainId",
            "CanvasId",
            "Name",
            "Uuid",
            "Mipmap",
            "NextIndex",
        }
        check_unknown(row, known, frozenset(), "BrushPatternImageRecord")
        return cls(
            pw_id=as_int(row, "_PW_ID"),
            main_id=as_int(row, "MainId"),
            canvas_id=as_int(row, "CanvasId"),
            name=opt_str(row, "Name"),
            uuid=format_uuid(opt_bytes(row, "Uuid")),
            mipmap=as_int(row, "Mipmap"),
            next_index=as_int(row, "NextIndex"),
        )


@dataclass
class BrushPatternStyleRecord:
    pw_id: int
    main_id: int
    canvas_id: int
    next_index: int
    image_number: int  # number of pattern-image references packed into image_index
    image_index: Optional[bytes]  # parse via `parse_brush_pattern_image_index`
    order_type: int
    reverse2: int

    @classmethod
    def from_row(cls, row) -> "BrushPatternStyleRecord":
        known = {
            "_PW_ID",
            "MainId",
            "CanvasId",
            "NextIndex",
            "ImageNumber",
            "ImageIndex",
            "OrderType",
            "Reverse2",
        }
        check_unknown(row, known, frozenset(), "BrushPatternStyleRecord")
        return cls(
            pw_id=as_int(row, "_PW_ID"),
            main_id=as_int(row, "MainId"),
            canvas_id=as_int(row, "CanvasId"),
            next_index=as_int(row, "NextIndex"),
            image_number=as_int(row, "ImageNumber"),
            image_index=opt_bytes(row, "ImageIndex"),
            order_type=as_int(row, "OrderType"),
            reverse2=as_int(row, "Reverse2"),
        )


@dataclass
class BrushEffectorGraphDataRecord:
    """A row of `BrushEffectorGraphData` — one pressure-curve definition."""

    pw_id: int
    main_id: int
    canvas_id: int
    next_index: int
    control_number: int  # number of control points in `control_points`
    control_data_size: int  # bytes-per-point (always 16: f64 x + f64 y)
    control_points: List[tuple]  # list of (x: float, y: float)

    @classmethod
    def from_row(cls, row) -> "BrushEffectorGraphDataRecord":
        known = {
            "_PW_ID",
            "MainId",
            "CanvasId",
            "NextIndex",
            "ControlNumber",
            "ControlDataSize",
            "ControlPoints",
        }
        check_unknown(row, known, frozenset(), "BrushEffectorGraphDataRecord")
        n = as_int(row, "ControlNumber")
        raw = opt_bytes(row, "ControlPoints")
        pts = parse_effector_control_points(raw, n) if raw is not None else []
        return cls(
            pw_id=as_int(row, "_PW_ID"),
            main_id=as_int(row, "MainId"),
            canvas_id=as_int(row, "CanvasId"),
            next_index=as_int(row, "NextIndex"),
            control_number=n,
            control_data_size=as_int(row, "ControlDataSize", default=16),
            control_points=pts,
        )


@dataclass
class BrushStyleManagerRecord:
    """A row of `BrushStyleManager` — first-element pointers per brush-related table."""

    pw_id: int
    main_id: int
    canvas_id: int
    first_brush_style: int
    first_pattern: int
    first_pattern_image: int
    first_graph_data: int
    first_fill_style: int
    first_fixed_spray: int

    @classmethod
    def from_row(cls, row) -> "BrushStyleManagerRecord":
        known = {
            "_PW_ID",
            "MainId",
            "CanvasId",
            "FirstBrushStyle",
            "FirstPattern",
            "FirstPatternImage",
            "FirstGraphData",
            "FirstFillStyle",
            "FirstFixedSpray",
        }
        check_unknown(row, known, frozenset(), "BrushStyleManagerRecord")
        return cls(
            pw_id=as_int(row, "_PW_ID"),
            main_id=as_int(row, "MainId"),
            canvas_id=as_int(row, "CanvasId"),
            first_brush_style=as_int(row, "FirstBrushStyle"),
            first_pattern=as_int(row, "FirstPattern"),
            first_pattern_image=as_int(row, "FirstPatternImage"),
            first_graph_data=as_int(row, "FirstGraphData"),
            first_fill_style=as_int(row, "FirstFillStyle"),
            first_fixed_spray=as_int(row, "FirstFixedSpray"),
        )


# ---------------------------------------------------------------------------
# Animation / Track / TimeLine
# ---------------------------------------------------------------------------


@dataclass
class TrackRecord:
    """A row of the `Track` table — one animation track binding a layer."""

    pw_id: int
    main_id: int
    bank_id: int
    item_id: int
    track_next_index: int
    track_action_mixer_size: int
    track_action_mixer: Optional[bytes]  # zlib-prefixed binc; parse with parse_binc
    track_action_mixer2_size: int
    track_action_mixer2: Optional[bytes]
    track_value_map: Optional[bytes]  # parse with parse_track_value_map
    track_open: int
    track_content_open: int
    track_uuid: Optional[str]
    layer_uuid_with_track: Optional[str]
    track_kind: int
    track_option_flag: int
    track_layer_object_uuid: Optional[str]

    @classmethod
    def from_row(cls, row) -> "TrackRecord":
        known = {
            "_PW_ID",
            "MainId",
            "BankId",
            "ItemId",
            "TrackNextIndex",
            "TrackActionMixerSize",
            "TrackActionMixer",
            "TrackActionMixer2Size",
            "TrackActionMixer2",
            "TrackValueMap",
            "TrackOpen",
            "TrackContentOpen",
            "TrackUuid",
            "LayerUuidWithTrack",
            "TrackKind",
            "TrackOptionFlag",
            "TrackLayerObjectUuid",
        }
        check_unknown(row, known, frozenset(), "TrackRecord")
        return cls(
            pw_id=as_int(row, "_PW_ID"),
            main_id=as_int(row, "MainId"),
            bank_id=as_int(row, "BankId"),
            item_id=as_int(row, "ItemId"),
            track_next_index=as_int(row, "TrackNextIndex"),
            track_action_mixer_size=as_int(row, "TrackActionMixerSize"),
            track_action_mixer=opt_bytes(row, "TrackActionMixer"),
            track_action_mixer2_size=as_int(row, "TrackActionMixer2Size"),
            track_action_mixer2=opt_bytes(row, "TrackActionMixer2"),
            track_value_map=opt_bytes(row, "TrackValueMap"),
            track_open=as_int(row, "TrackOpen"),
            track_content_open=as_int(row, "TrackContentOpen"),
            track_uuid=format_uuid(opt_bytes(row, "TrackUuid")),
            layer_uuid_with_track=format_uuid(opt_bytes(row, "LayerUuidWithTrack")),
            track_kind=as_int(row, "TrackKind"),
            track_option_flag=as_int(row, "TrackOptionFlag"),
            track_layer_object_uuid=format_uuid(opt_bytes(row, "TrackLayerObjectUuid")),
        )


@dataclass
class TimeLineRecord:
    pw_id: int
    main_id: int
    bank_id: int
    timeline_name: str
    timeline_uuid: Optional[str]
    next_timeline: int
    next_scenario: int
    first_track: int
    label_first_index: int
    frame_rate: int
    guideline_frame_rate: int
    start_frame: int
    end_frame: int
    current_frame: int
    smallest_start_frame: int
    biggest_end_frame: int
    cut_index_for_name: int
    scene_index_for_name: int

    @classmethod
    def from_row(cls, row) -> "TimeLineRecord":
        known = {
            "_PW_ID",
            "MainId",
            "BankId",
            "TimeLineName",
            "TimeLineUuid",
            "NextTimeLine",
            "NextScenario",
            "FirstTrack",
            "LabelFirstIndex",
            "FrameRate",
            "GuidelineFrameRate",
            "StartFrame",
            "EndFrame",
            "CurrentFrame",
            "SmallestStartFrame",
            "BiggestEndFrame",
            "CutIndexForName",
            "SceneIndexForName",
        }
        check_unknown(row, known, frozenset(), "TimeLineRecord")
        return cls(
            pw_id=as_int(row, "_PW_ID"),
            main_id=as_int(row, "MainId"),
            bank_id=as_int(row, "BankId"),
            timeline_name=as_str(row, "TimeLineName"),
            timeline_uuid=format_uuid(opt_bytes(row, "TimeLineUuid")),
            next_timeline=as_int(row, "NextTimeLine"),
            next_scenario=as_int(row, "NextScenario"),
            first_track=as_int(row, "FirstTrack"),
            label_first_index=as_int(row, "LabelFirstIndex"),
            frame_rate=as_int(row, "FrameRate"),
            guideline_frame_rate=as_int(row, "GuidelineFrameRate"),
            start_frame=as_int(row, "StartFrame"),
            end_frame=as_int(row, "EndFrame"),
            current_frame=as_int(row, "CurrentFrame"),
            smallest_start_frame=as_int(row, "SmallestStartFrame"),
            biggest_end_frame=as_int(row, "BiggestEndFrame"),
            cut_index_for_name=as_int(row, "CutIndexForName"),
            scene_index_for_name=as_int(row, "SceneIndexForName"),
        )


@dataclass
class AnimationCutBankRecord:
    pw_id: int
    main_id: int
    enable: int
    current_index: int
    first_timeline: int
    first_scenario: int
    flag_scenario_v155: int

    @classmethod
    def from_row(cls, row) -> "AnimationCutBankRecord":
        known = {
            "_PW_ID",
            "MainId",
            "Enable",
            "CurrentIndex",
            "FirstTimeLine",
            "FirstScenario",
            "FlagScenarioV155",
        }
        check_unknown(row, known, frozenset(), "AnimationCutBankRecord")
        return cls(
            pw_id=as_int(row, "_PW_ID"),
            main_id=as_int(row, "MainId"),
            enable=as_int(row, "Enable"),
            current_index=as_int(row, "CurrentIndex"),
            first_timeline=as_int(row, "FirstTimeLine"),
            first_scenario=as_int(row, "FirstScenario"),
            flag_scenario_v155=as_int(row, "FlagScenarioV155"),
        )


# ---------------------------------------------------------------------------
# Layer Composition (saved layer-state snapshots)
# ---------------------------------------------------------------------------


@dataclass
class LayerCompRecord:
    pw_id: int
    main_id: int
    bank_id: int
    comp_next_index: int
    comp_name: str
    comp_uuid: Optional[str]
    comp_layer_info: Optional[bytes]  # parse with parse_comp_layer_info

    @classmethod
    def from_row(cls, row) -> "LayerCompRecord":
        known = {
            "_PW_ID",
            "MainId",
            "BankId",
            "CompNextIndex",
            "CompName",
            "CompUuid",
            "CompLayerInfo",
        }
        check_unknown(row, known, frozenset(), "LayerCompRecord")
        return cls(
            pw_id=as_int(row, "_PW_ID"),
            main_id=as_int(row, "MainId"),
            bank_id=as_int(row, "BankId"),
            comp_next_index=as_int(row, "CompNextIndex"),
            comp_name=as_str(row, "CompName"),
            comp_uuid=format_uuid(opt_bytes(row, "CompUuid")),
            comp_layer_info=opt_bytes(row, "CompLayerInfo"),
        )


@dataclass
class LayerCompManagerRecord:
    pw_id: int
    main_id: int
    first_layer_comp_index: (
        int  # head of the LayerComp chain (next_index is `CompNextIndex`)
    )
    last_state_layer_comp_index: int
    applied_layer_comp_index: int

    @classmethod
    def from_row(cls, row) -> "LayerCompManagerRecord":
        known = {
            "_PW_ID",
            "MainId",
            "FirstLayerCompIndex",
            "LastStateLayerCompIndex",
            "AppliedLayerCompIndex",
        }
        check_unknown(row, known, frozenset(), "LayerCompManagerRecord")
        return cls(
            pw_id=as_int(row, "_PW_ID"),
            main_id=as_int(row, "MainId"),
            first_layer_comp_index=as_int(row, "FirstLayerCompIndex"),
            last_state_layer_comp_index=as_int(row, "LastStateLayerCompIndex"),
            applied_layer_comp_index=as_int(row, "AppliedLayerCompIndex"),
        )


# ---------------------------------------------------------------------------
# 3D scene state
# ---------------------------------------------------------------------------


@dataclass
class LightInfoRecord:
    pw_id: int
    main_id: int
    canvas_id: int
    layer_id: int
    layer_object_id: int
    light_index: int
    light_type: int
    light_uuid: Optional[str]

    @classmethod
    def from_row(cls, row) -> "LightInfoRecord":
        known = {
            "_PW_ID",
            "MainId",
            "CanvasId",
            "LayerId",
            "LayerObjectId",
            "LightIndex",
            "LightType",
            "LightUuid",
        }
        check_unknown(row, known, frozenset(), "LightInfoRecord")
        return cls(
            pw_id=as_int(row, "_PW_ID"),
            main_id=as_int(row, "MainId"),
            canvas_id=as_int(row, "CanvasId"),
            layer_id=as_int(row, "LayerId"),
            layer_object_id=as_int(row, "LayerObjectId"),
            light_index=as_int(row, "LightIndex"),
            light_type=as_int(row, "LightType"),
            light_uuid=opt_str(row, "LightUuid"),
        )


@dataclass
class CameraInfoRecord:
    """A row of `CameraInfo` — 3D camera position + frustum + viewport."""

    pw_id: int
    main_id: int
    canvas_id: int
    layer_id: int
    layer_object_id: int
    camera_uuid_main: Optional[str]
    camera_uuid_sub: Tuple[
        Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]
    ]
    camera_position: Tuple[float, float, float]
    camera_target: Tuple[float, float, float]
    camera_up: Tuple[float, float, float]
    camera_twist: float
    layer_optical_axis_pt: Tuple[float, float]
    frustum_left: float
    frustum_right: float
    frustum_top: float
    frustum_bottom: float
    frustum_near: float
    frustum_far: float
    frustum_ortho: int
    viewport_xmin: float
    viewport_ymin: float
    viewport_width: float
    viewport_height: float

    @classmethod
    def from_row(cls, row) -> "CameraInfoRecord":
        known = {
            "_PW_ID",
            "MainId",
            "CanvasId",
            "LayerId",
            "LayerObjectId",
            "CameraUuidMain",
            "CameraUuidSub0",
            "CameraUuidSub1",
            "CameraUuidSub2",
            "CameraUuidSub3",
            "CameraUuidSub4",
            "CameraPositionX",
            "CameraPositionY",
            "CameraPositionZ",
            "CameraTargetX",
            "CameraTargetY",
            "CameraTargetZ",
            "CameraUpX",
            "CameraUpY",
            "CameraUpZ",
            "CameraTwist",
            "LayerOpticalAxisPtX",
            "LayerOpticalAxisPtY",
            "FrustumLeft",
            "FrustumRight",
            "FrustumTop",
            "FrustumBottom",
            "FrustumNear",
            "FrustumFar",
            "FrustumOrtho",
            "ViewportXmin",
            "ViewportYmin",
            "ViewportWidth",
            "ViewportHeight",
        }
        check_unknown(row, known, frozenset(), "CameraInfoRecord")
        return cls(
            pw_id=as_int(row, "_PW_ID"),
            main_id=as_int(row, "MainId"),
            canvas_id=as_int(row, "CanvasId"),
            layer_id=as_int(row, "LayerId"),
            layer_object_id=as_int(row, "LayerObjectId"),
            camera_uuid_main=opt_str(row, "CameraUuidMain"),
            camera_uuid_sub=(
                opt_str(row, "CameraUuidSub0"),
                opt_str(row, "CameraUuidSub1"),
                opt_str(row, "CameraUuidSub2"),
                opt_str(row, "CameraUuidSub3"),
                opt_str(row, "CameraUuidSub4"),
            ),
            camera_position=(
                as_float(row, "CameraPositionX"),
                as_float(row, "CameraPositionY"),
                as_float(row, "CameraPositionZ"),
            ),
            camera_target=(
                as_float(row, "CameraTargetX"),
                as_float(row, "CameraTargetY"),
                as_float(row, "CameraTargetZ"),
            ),
            camera_up=(
                as_float(row, "CameraUpX"),
                as_float(row, "CameraUpY"),
                as_float(row, "CameraUpZ"),
            ),
            camera_twist=as_float(row, "CameraTwist"),
            layer_optical_axis_pt=(
                as_float(row, "LayerOpticalAxisPtX"),
                as_float(row, "LayerOpticalAxisPtY"),
            ),
            frustum_left=as_float(row, "FrustumLeft"),
            frustum_right=as_float(row, "FrustumRight"),
            frustum_top=as_float(row, "FrustumTop"),
            frustum_bottom=as_float(row, "FrustumBottom"),
            frustum_near=as_float(row, "FrustumNear"),
            frustum_far=as_float(row, "FrustumFar"),
            frustum_ortho=as_int(row, "FrustumOrtho"),
            viewport_xmin=as_float(row, "ViewportXmin"),
            viewport_ymin=as_float(row, "ViewportYmin"),
            viewport_width=as_float(row, "ViewportWidth"),
            viewport_height=as_float(row, "ViewportHeight"),
        )


@dataclass
class Canvas3DModelBankRecord:
    pw_id: int
    main_id: int
    bank_id: int
    first_loader_index: int

    @classmethod
    def from_row(cls, row) -> "Canvas3DModelBankRecord":
        known = {"_PW_ID", "MainId", "BankId", "FirstLoaderIndex"}
        check_unknown(row, known, frozenset(), "Canvas3DModelBankRecord")
        return cls(
            pw_id=as_int(row, "_PW_ID"),
            main_id=as_int(row, "MainId"),
            bank_id=as_int(row, "BankId"),
            first_loader_index=as_int(row, "FirstLoaderIndex"),
        )


@dataclass
class DessinDollInfoRecord:
    pw_id: int
    main_id: int
    canvas_id: int
    layer_id: int
    layer_object_id: int
    dessindoll_uuid: Optional[str]
    dessindoll_gui_opened: int

    @classmethod
    def from_row(cls, row) -> "DessinDollInfoRecord":
        known = {
            "_PW_ID",
            "MainId",
            "CanvasId",
            "LayerId",
            "LayerObjectId",
            "DessindollUUID",
            "DessindollGUIOpened",
        }
        check_unknown(row, known, frozenset(), "DessinDollInfoRecord")
        return cls(
            pw_id=as_int(row, "_PW_ID"),
            main_id=as_int(row, "MainId"),
            canvas_id=as_int(row, "CanvasId"),
            layer_id=as_int(row, "LayerId"),
            layer_object_id=as_int(row, "LayerObjectId"),
            dessindoll_uuid=opt_str(row, "DessindollUUID"),
            dessindoll_gui_opened=as_int(row, "DessindollGUIOpened"),
        )


# ---------------------------------------------------------------------------
# Rulers
# ---------------------------------------------------------------------------


@dataclass
class RulerVanishPointRecord:
    pw_id: int
    main_id: int
    canvas_id: int
    layer_id: int
    next_index: int
    flag: int
    vanish_point_x: float
    vanish_point_y: float
    parallel_angle: float
    guide_number: int
    guide_data_size: int
    # Decoded guide endpoints: list of (x0, y0, x1, y1) tuples — one per guide line.
    guides: List[tuple]

    @classmethod
    def from_row(cls, row) -> "RulerVanishPointRecord":
        known = {
            "_PW_ID",
            "MainId",
            "CanvasId",
            "LayerId",
            "NextIndex",
            "Flag",
            "VanishPointX",
            "VanishPointY",
            "ParallelAngle",
            "GuideNumber",
            "GuideDataSize",
            "Guide",
        }
        check_unknown(row, known, frozenset(), "RulerVanishPointRecord")
        raw = opt_bytes(row, "Guide")
        n = as_int(row, "GuideNumber")
        guides = parse_vanish_point_guide(raw, n) if raw is not None else []
        return cls(
            pw_id=as_int(row, "_PW_ID"),
            main_id=as_int(row, "MainId"),
            canvas_id=as_int(row, "CanvasId"),
            layer_id=as_int(row, "LayerId"),
            next_index=as_int(row, "NextIndex"),
            flag=as_int(row, "Flag"),
            vanish_point_x=as_float(row, "VanishPointX"),
            vanish_point_y=as_float(row, "VanishPointY"),
            parallel_angle=as_float(row, "ParallelAngle"),
            guide_number=n,
            guide_data_size=as_int(row, "GuideDataSize"),
            guides=guides,
        )


@dataclass
class RulerPerspectiveRecord:
    """A row of `RulerPerspective` — perspective-ruler config + lens setup."""

    pw_id: int
    main_id: int
    canvas_id: int
    layer_id: int
    next_index: int
    flag: int
    perspective_type: int
    grid_flag: int
    grid_size: float
    grid_origin_x: float
    grid_origin_y: float
    eye_level_handle_x: float
    eye_level_handle_y: float
    move_handle_x: float
    move_handle_y: float
    distortion: float
    distortion_weight: float
    initialized_lens: int
    lens_center_x: float
    lens_center_y: float
    lens_radius: float
    lens_radius_edit_handle_angle: float
    camera_near: float
    first_vanish_index: int
    center_guide_number: int
    center_guide_data_size: int
    center_guide_position: Optional[bytes]

    @classmethod
    def from_row(cls, row) -> "RulerPerspectiveRecord":
        known = {
            "_PW_ID",
            "MainId",
            "CanvasId",
            "LayerId",
            "NextIndex",
            "Flag",
            "PerspectiveType",
            "GridFlag",
            "GridSize",
            "GridOriginX",
            "GridOriginY",
            "EyeLevelHandleX",
            "EyeLevelHandleY",
            "MoveHandleX",
            "MoveHandleY",
            "Distortion",
            "DistortionWeight",
            "InitializedLens",
            "LensCenterX",
            "LensCenterY",
            "LensRadius",
            "LensRadiusEditHandleAngle",
            "CameraNear",
            "FirstVanishIndex",
            "centerGuideNumber",
            "centerGuideDataSize",
            "centerGuidePosition",
        }
        check_unknown(row, known, frozenset(), "RulerPerspectiveRecord")
        return cls(
            pw_id=as_int(row, "_PW_ID"),
            main_id=as_int(row, "MainId"),
            canvas_id=as_int(row, "CanvasId"),
            layer_id=as_int(row, "LayerId"),
            next_index=as_int(row, "NextIndex"),
            flag=as_int(row, "Flag"),
            perspective_type=as_int(row, "PerspectiveType"),
            grid_flag=as_int(row, "GridFlag"),
            grid_size=as_float(row, "GridSize"),
            grid_origin_x=as_float(row, "GridOriginX"),
            grid_origin_y=as_float(row, "GridOriginY"),
            eye_level_handle_x=as_float(row, "EyeLevelHandleX"),
            eye_level_handle_y=as_float(row, "EyeLevelHandleY"),
            move_handle_x=as_float(row, "MoveHandleX"),
            move_handle_y=as_float(row, "MoveHandleY"),
            distortion=as_float(row, "Distortion"),
            distortion_weight=as_float(row, "DistortionWeight"),
            initialized_lens=as_int(row, "InitializedLens"),
            lens_center_x=as_float(row, "LensCenterX"),
            lens_center_y=as_float(row, "LensCenterY"),
            lens_radius=as_float(row, "LensRadius"),
            lens_radius_edit_handle_angle=as_float(row, "LensRadiusEditHandleAngle"),
            camera_near=as_float(row, "CameraNear"),
            first_vanish_index=as_int(row, "FirstVanishIndex"),
            center_guide_number=as_int(row, "centerGuideNumber"),
            center_guide_data_size=as_int(row, "centerGuideDataSize"),
            center_guide_position=opt_bytes(row, "centerGuidePosition"),
        )


@dataclass
class SpecialRulerManagerRecord:
    """A row of `SpecialRulerManager` — first-element pointers per ruler type."""

    pw_id: int
    main_id: int
    canvas_id: int
    layer_id: int
    first_perspective: int
    first_parallel: int
    first_emit: int
    first_curve_emit: int
    first_curve_parallel: int
    first_multi_curve: int
    first_concentric_circle: int
    first_symmetry: int
    first_guide: int

    @classmethod
    def from_row(cls, row) -> "SpecialRulerManagerRecord":
        known = {
            "_PW_ID",
            "MainId",
            "CanvasId",
            "LayerId",
            "FirstPerspective",
            "FirstParallel",
            "FirstEmit",
            "FirstCurveEmit",
            "FirstCurveParallel",
            "FirstMultiCurve",
            "FirstConcentricCircle",
            "FirstSymmetry",
            "FirstGuide",
        }
        check_unknown(row, known, frozenset(), "SpecialRulerManagerRecord")
        return cls(
            pw_id=as_int(row, "_PW_ID"),
            main_id=as_int(row, "MainId"),
            canvas_id=as_int(row, "CanvasId"),
            layer_id=as_int(row, "LayerId"),
            first_perspective=as_int(row, "FirstPerspective"),
            first_parallel=as_int(row, "FirstParallel"),
            first_emit=as_int(row, "FirstEmit"),
            first_curve_emit=as_int(row, "FirstCurveEmit"),
            first_curve_parallel=as_int(row, "FirstCurveParallel"),
            first_multi_curve=as_int(row, "FirstMultiCurve"),
            first_concentric_circle=as_int(row, "FirstConcentricCircle"),
            first_symmetry=as_int(row, "FirstSymmetry"),
            first_guide=as_int(row, "FirstGuide"),
        )


# ---------------------------------------------------------------------------
# Small / item objects
# ---------------------------------------------------------------------------


@dataclass
class SmallObjectInfoRecord:
    pw_id: int
    main_id: int
    canvas_id: int
    layer_id: int
    layer_object_id: int
    small_object_uuid: Optional[str]
    tree_node_gui_open_close: Optional[SmallObjectFlag]
    last_applied_movable_frame: Optional[SmallObjectFlag]
    tree_node_gui_name: Optional[SmallObjectFlag]
    canvas_item_source_is_made_by_canvas_item_maker: int
    canvas_item_source_file_name: Optional[str]
    canvas_item_source_material_owner_user_id: int
    canvas_item_source_material_assets_content_id: int
    canvas_item_source_material_revision: Optional[str]
    canvas_item_source_material_is_uploadable: int

    @classmethod
    def from_row(cls, row) -> "SmallObjectInfoRecord":
        known = {
            "_PW_ID",
            "MainId",
            "CanvasId",
            "LayerId",
            "LayerObjectId",
            "SmallObjectUUID",
            "SmallObjectTreeNodeGUIOpenClose",
            "SmallObjectLastAppliedMovableFrame",
            "SmallObjectTreeNodeGUIName",
            "CanvasItemSourceIsMadeByCanvasItemMaker",
            "CanvasItemSourceFileName",
            "CanvasItemSourceMaterialOwnerUserId",
            "CanvasItemSourceMaterialAssetsContentId",
            "CanvasItemSourceMaterialRevision",
            "CanvasItemSourceMaterialIsUploadable",
        }
        check_unknown(row, known, frozenset(), "SmallObjectInfoRecord")
        return cls(
            pw_id=as_int(row, "_PW_ID"),
            main_id=as_int(row, "MainId"),
            canvas_id=as_int(row, "CanvasId"),
            layer_id=as_int(row, "LayerId"),
            layer_object_id=as_int(row, "LayerObjectId"),
            small_object_uuid=opt_str(row, "SmallObjectUUID"),
            tree_node_gui_open_close=parse_small_object_flag(
                opt_bytes(row, "SmallObjectTreeNodeGUIOpenClose")
            ),
            last_applied_movable_frame=parse_small_object_flag(
                opt_bytes(row, "SmallObjectLastAppliedMovableFrame")
            ),
            tree_node_gui_name=parse_small_object_flag(
                opt_bytes(row, "SmallObjectTreeNodeGUIName")
            ),
            canvas_item_source_is_made_by_canvas_item_maker=as_int(
                row, "CanvasItemSourceIsMadeByCanvasItemMaker"
            ),
            canvas_item_source_file_name=opt_str(row, "CanvasItemSourceFileName"),
            canvas_item_source_material_owner_user_id=as_int(
                row, "CanvasItemSourceMaterialOwnerUserId"
            ),
            canvas_item_source_material_assets_content_id=as_int(
                row, "CanvasItemSourceMaterialAssetsContentId"
            ),
            canvas_item_source_material_revision=opt_str(
                row, "CanvasItemSourceMaterialRevision"
            ),
            canvas_item_source_material_is_uploadable=as_int(
                row, "CanvasItemSourceMaterialIsUploadable"
            ),
        )


@dataclass
class CanvasItemBankRecord:
    pw_id: int
    main_id: int
    bank_root_item_main_index: int
    model_bank_main_index: int

    @classmethod
    def from_row(cls, row) -> "CanvasItemBankRecord":
        known = {
            "_PW_ID",
            "MainId",
            "BankRootItemMainIndex",
            "ModelBankMainIndex",
        }
        check_unknown(row, known, frozenset(), "CanvasItemBankRecord")
        return cls(
            pw_id=as_int(row, "_PW_ID"),
            main_id=as_int(row, "MainId"),
            bank_root_item_main_index=as_int(row, "BankRootItemMainIndex"),
            model_bank_main_index=as_int(row, "ModelBankMainIndex"),
        )


# ---------------------------------------------------------------------------
# Canvas preview thumbnail
# ---------------------------------------------------------------------------


@dataclass
class CanvasPreviewRecord:
    pw_id: int
    main_id: int
    canvas_id: int
    image_type: int  # observed always 0 (= PNG); enum unconfirmed
    image_width: int
    image_height: int
    image_data: Optional[bytes]  # PNG bytes — feed to PIL.Image.open(BytesIO(...))

    @classmethod
    def from_row(cls, row) -> "CanvasPreviewRecord":
        known = {
            "_PW_ID",
            "MainId",
            "CanvasId",
            "ImageType",
            "ImageWidth",
            "ImageHeight",
            "ImageData",
        }
        check_unknown(row, known, frozenset(), "CanvasPreviewRecord")
        return cls(
            pw_id=as_int(row, "_PW_ID"),
            main_id=as_int(row, "MainId"),
            canvas_id=as_int(row, "CanvasId"),
            image_type=as_int(row, "ImageType"),
            image_width=as_int(row, "ImageWidth"),
            image_height=as_int(row, "ImageHeight"),
            image_data=opt_bytes(row, "ImageData"),
        )


# ---------------------------------------------------------------------------
# External chunk index
# ---------------------------------------------------------------------------


@dataclass
class ExternalTableAndColumnNameRecord:
    pw_id: int
    main_id: int
    table_name: str
    column_name: str

    @classmethod
    def from_row(cls, row) -> "ExternalTableAndColumnNameRecord":
        known = {"_PW_ID", "MainId", "TableName", "ColumnName"}
        check_unknown(row, known, frozenset(), "ExternalTableAndColumnNameRecord")
        return cls(
            pw_id=as_int(row, "_PW_ID"),
            main_id=as_int(row, "MainId"),
            table_name=as_str(row, "TableName"),
            column_name=as_str(row, "ColumnName"),
        )


@dataclass
class ExternalChunkRecord:
    """A row of the `ExternalChunk` table — one external-id ↔ binary-offset
    mapping that the chunk parser uses to look up where each external lives
    inside the .clip binary blob."""

    external_id: bytes  # 32-byte ASCII id like b"extrnlid<32 hex>"
    offset: int

    @classmethod
    def from_row(cls, row) -> "ExternalChunkRecord":
        known = {"ExternalID", "Offset"}
        check_unknown(row, known, frozenset(), "ExternalChunkRecord")
        eid = row["ExternalID"]
        if not isinstance(eid, (bytes, bytearray)):
            eid = str(eid).encode("ascii")
        return cls(
            external_id=bytes(eid),
            offset=as_int(row, "Offset"),
        )


# ---------------------------------------------------------------------------
# Parameter / element schemas (large UI dictionaries)
# ---------------------------------------------------------------------------


@dataclass
class ParamSchemeRecord:
    """A row of `ParamScheme` — UI parameter schema (description of one
    column in another table for the parameter-editor UI). 1000+ rows in
    real files; not used at render time."""

    pw_id: int
    table_name: str
    label_name: str
    data_type: int
    flag: int
    owner_type: int
    link_table: str
    lock_specified: int
    lock_type: int
    alternative_lock_specified: int
    alternative_lock_type: int

    @classmethod
    def from_row(cls, row) -> "ParamSchemeRecord":
        known = {
            "_PW_ID",
            "TableName",
            "LabelName",
            "DataType",
            "Flag",
            "OwnerType",
            "LinkTable",
            "LockSpecified",
            "LockType",
            "AlternativeLockSpecified",
            "AlternativeLockType",
        }
        check_unknown(row, known, frozenset(), "ParamSchemeRecord")
        return cls(
            pw_id=as_int(row, "_PW_ID"),
            table_name=as_str(row, "TableName"),
            label_name=as_str(row, "LabelName"),
            data_type=as_int(row, "DataType"),
            flag=as_int(row, "Flag"),
            owner_type=as_int(row, "OwnerType"),
            link_table=as_str(row, "LinkTable"),
            lock_specified=as_int(row, "LockSpecified"),
            lock_type=as_int(row, "LockType"),
            alternative_lock_specified=as_int(row, "AlternativeLockSpecified"),
            alternative_lock_type=as_int(row, "AlternativeLockType"),
        )


@dataclass
class ElemSchemeRecord:
    """A row of `ElemScheme` — table-element schema (max-index registry per
    SQLite table). Used by CSP's UI to allocate fresh row IDs."""

    pw_id: int
    table_name: str
    elem_type: int
    max_index: int

    @classmethod
    def from_row(cls, row) -> "ElemSchemeRecord":
        known = {"_PW_ID", "TableName", "ElemType", "MaxIndex"}
        check_unknown(row, known, frozenset(), "ElemSchemeRecord")
        return cls(
            pw_id=as_int(row, "_PW_ID"),
            table_name=as_str(row, "TableName"),
            elem_type=as_int(row, "ElemType"),
            max_index=as_int(row, "MaxIndex"),
        )


# ---------------------------------------------------------------------------
# Bulk loader
# ---------------------------------------------------------------------------


TABLE_RECORD_CLASSES = {
    "Canvas": CanvasRecord,
    "Project": ProjectRecord,
    "Offscreen": OffscreenRecord,
    "Mipmap": MipmapRecord,
    "MipmapInfo": MipmapInfoRecord,
    "LayerThumbnail": LayerThumbnailRecord,
    "VectorObjectList": VectorObjectRecord,
    "BrushPatternImage": BrushPatternImageRecord,
    "BrushPatternStyle": BrushPatternStyleRecord,
    "BrushEffectorGraphData": BrushEffectorGraphDataRecord,
    "BrushStyleManager": BrushStyleManagerRecord,
    "Track": TrackRecord,
    "TimeLine": TimeLineRecord,
    "AnimationCutBank": AnimationCutBankRecord,
    "LayerComp": LayerCompRecord,
    "LayerCompManager": LayerCompManagerRecord,
    "LightInfo": LightInfoRecord,
    "CameraInfo": CameraInfoRecord,
    "Canvas3DModelBank": Canvas3DModelBankRecord,
    "DessinDollInfo": DessinDollInfoRecord,
    "RulerVanishPoint": RulerVanishPointRecord,
    "RulerPerspective": RulerPerspectiveRecord,
    "SpecialRulerManager": SpecialRulerManagerRecord,
    "SmallObjectInfo": SmallObjectInfoRecord,
    "CanvasItemBank": CanvasItemBankRecord,
    "CanvasPreview": CanvasPreviewRecord,
    "ExternalChunk": ExternalChunkRecord,
    "ExternalTableAndColumnName": ExternalTableAndColumnNameRecord,
    "ParamScheme": ParamSchemeRecord,
    "ElemScheme": ElemSchemeRecord,
}
"""Map from SQLite table name to its record dataclass.

Note: `Layer` and `BrushStyle` already have records in `clip_tools.types`
(`LayerRecord`, `BrushStyle`); their entries are added by
`build_table_records` so consumers can use a single dispatcher.

`sqlite_sequence` is intentionally absent (SQLite internal bookkeeping).
"""


def build_table_records(dfs):
    """Convert every supported table's DataFrame to a list of typed records.

    Returns `Dict[str, List[Record]]`. Tables without a registered class are
    silently skipped (e.g. `sqlite_sequence`).
    """
    classes = dict(TABLE_RECORD_CLASSES)
    classes["Layer"] = LayerRecord
    classes["BrushStyle"] = BrushStyle

    out = {}
    for name, df in dfs.items():
        cls = classes.get(name)
        if cls is None or len(df) == 0:
            continue
        out[name] = [cls.from_row(row) for _, row in df.iterrows()]
    return out
