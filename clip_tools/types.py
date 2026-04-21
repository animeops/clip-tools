from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Point:
    x: int
    y: int
    opacity: float
    thickness: float


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
    """One control point of a parsed vector stroke."""

    x: float
    y: float
    pressure: float
    width_factor: float
    opacity_factor: float
    # Per-point size modulation stored alongside each control point. Varies
    # across a stroke as a bell/taper profile (peak in the middle, zero at
    # endpoints), encoding pressure / velocity response. Multiplies
    # brush_size to get the rendered diameter.
    size_modulation: float = 1.0
    curve: Optional[tuple] = None  # (cx, cy) for CURVE-type quadratic handles


@dataclass
class VectorStroke:
    """A single stroke inside a vector object — geometry + brush reference."""

    vtype: "object"  # VectorType enum, forward-ref to avoid circular import
    color: tuple  # (r, g, b) ints
    stroke_opacity: float
    brush_size: float
    brush_id: int
    points: list  # list[VectorPoint]


@dataclass
class VectorSample:
    """One densely-interpolated sample along a stroke's path."""

    x: float
    y: float
    pressure: float
    width_factor: float
    opacity_factor: float
    size_modulation: float = 1.0


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

    # Rotation (per-stamp)
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

    # Raw pressure-curve effector blobs — not parsed yet, kept as bytes.
    size_effector: bytes = b""
    opacity_effector: bytes = b""
    flow_effector: bytes = b""
    thickness_effector: bytes = b""
    interval_effector: bytes = b""

    @classmethod
    def from_row(cls, row) -> "BrushStyle":
        """Build from a pandas Series (one row of the BrushStyle DataFrame).

        Raises KeyError if a required column is missing — use this as a loud
        signal when CLIP adds/removes BrushStyle fields between versions.
        """

        def _f(key, default=0.0):
            v = row[key]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return default
            return float(v)

        def _i(key, default=0):
            v = row[key]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return default
            return int(v)

        def _b(key):
            v = row[key]
            return v if isinstance(v, (bytes, bytearray)) else b""

        return cls(
            main_id=_i("MainId"),
            pattern_style=_i("PatternStyle"),
            texture_pattern=_i("TexturePattern"),
            hardness=_f("Hardness", 1.0),
            thickness_base=_f("ThicknessBase", 1.0),
            anti_alias=_i("AntiAlias", 2),
            flow_base=_f("FlowBase", 1.0),
            interval_base=_f("IntervalBase", 1.0),
            auto_interval_type=_i("AutoIntervalType"),
            rotation_base=_f("RotationBase"),
            rotation_random=_f("RotationRandom"),
            rotation_effector=_i("RotationEffector"),
            texture_scale=_f("TextureScale", 1.0),
            texture_rotate=_f("TextureRotate"),
            texture_offset_x=_f("TextureOffsetX"),
            texture_offset_y=_f("TextureOffsetY"),
            texture_density_base=_f("TextureDensityBase", 1.0),
            composite_mode=_i("CompositeMode"),
            spray_flag=_i("SprayFlag"),
            spray_size_base=_f("SpraySizeBase"),
            spray_density_base=_f("SprayDensityBase"),
            spray_bias=_f("SprayBias"),
            size_effector=_b("SizeEffector"),
            opacity_effector=_b("OpacityEffector"),
            flow_effector=_b("FlowEffector"),
            thickness_effector=_b("ThicknessEffector"),
            interval_effector=_b("IntervalEffector"),
        )
