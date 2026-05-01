from dataclasses import dataclass, field
from typing import Optional

import numpy as np


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
            size_effector=as_bytes("SizeEffector"),
            opacity_effector=as_bytes("OpacityEffector"),
            flow_effector=as_bytes("FlowEffector"),
            thickness_effector=as_bytes("ThicknessEffector"),
            interval_effector=as_bytes("IntervalEffector"),
        )
