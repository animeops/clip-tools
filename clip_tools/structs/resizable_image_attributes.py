"""Parsers for the `ResizableImageInfo` and `Camera2DResizableImageInfo`
columns. Both use the same wire format — a 120-byte transform header
followed by 4 polygon-corner doubles."""

import struct
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from clip_tools.constants import DEBUG
from clip_tools.utils import read_binary_spec


@dataclass
class ResizableImageInfo:
    """Decoded `ResizableImageInfo` / `Camera2DResizableImageInfo` blob.

    The layer's pixel content (`original_width × original_height`) is mapped
    onto `polygon_coords` via the homography built from `source_coords`.
    `clip_layer.py` only consumes `source_coords` + `polygon_coords`.

    `unknown_a/b/c/d/e` remain unlabelled — values are stable across the
    samples we have but their semantics aren't pinned down.
    """

    header: int
    unknown_a: int
    unknown_b: Tuple[int, int, int, int]
    unknown_c: Tuple[int, int]
    original_width: int
    original_height: int
    zoom: Tuple[float, float]
    rotation: float
    offset_x: float
    offset_y: float
    origin_x: float  # matches the `Camera2DOriginalFrameCenterX` SQL column
    origin_y: float  # matches the `Camera2DOriginalFrameCenterY` SQL column
    unknown_d: Tuple[int, int, int, int]
    unknown_e: Tuple[int, int]
    top_left: Tuple[float, float]
    top_right: Tuple[float, float]
    bottom_left: Tuple[float, float]
    bottom_right: Tuple[float, float]
    source_coords: np.ndarray
    polygon_coords: np.ndarray
    trailing_bytes: Optional[bytes] = None


def process_resizable_image_attributes(attributes: bytes) -> ResizableImageInfo:
    """Decode a `ResizableImageInfo` (or `Camera2DResizableImageInfo`) blob.

    Layout (all big-endian)::

        u32       header (=120)
        u32       unknown_a
        u32[4]    unknown_b
        u32[2]    unknown_c
        u32[2]    original_width, original_height
        f64[2]    zoom_x, zoom_y
        f64       rotation
        f64[2]    offset_x, offset_y
        f64[2]    origin_x, origin_y
        u32[4]    unknown_d
        u32[2]    unknown_e
        f64[2]    top_left_x, top_left_y
        f64[2]    top_right_x, top_right_y
        f64[2]    bottom_left_x, bottom_left_y
        f64[2]    bottom_right_x, bottom_right_y
    """
    uint_be = struct.Struct(">I")
    uint2_be = struct.Struct(">II")
    uint4_be = struct.Struct(">IIII")
    double_be = struct.Struct(">d")
    double2_be = struct.Struct(">dd")

    pos = 0
    (header,), pos = read_binary_spec(attributes, uint_be, pos)
    if header != 120:
        raise ValueError(
            f"Invalid resizable-image attributes header: expected 120, got {header}"
        )

    (unknown_a,), pos = read_binary_spec(attributes, uint_be, pos)
    unknown_b, pos = read_binary_spec(attributes, uint4_be, pos)
    unknown_c, pos = read_binary_spec(attributes, uint2_be, pos)
    (original_width, original_height), pos = read_binary_spec(attributes, uint2_be, pos)
    zoom, pos = read_binary_spec(attributes, double2_be, pos)
    (rotation,), pos = read_binary_spec(attributes, double_be, pos)
    (offset_x, offset_y), pos = read_binary_spec(attributes, double2_be, pos)
    (origin_x, origin_y), pos = read_binary_spec(attributes, double2_be, pos)
    unknown_d, pos = read_binary_spec(attributes, uint4_be, pos)
    unknown_e, pos = read_binary_spec(attributes, uint2_be, pos)
    top_left, pos = read_binary_spec(attributes, double2_be, pos)
    top_right, pos = read_binary_spec(attributes, double2_be, pos)
    bottom_left, pos = read_binary_spec(attributes, double2_be, pos)
    bottom_right, pos = read_binary_spec(attributes, double2_be, pos)

    polygon_coords = np.stack(
        [list(top_left), list(top_right), list(bottom_right), list(bottom_left)],
        axis=0,
    )
    source_coords = np.stack(
        [
            [0, 0],
            [original_width, 0],
            [original_width, original_height],
            [0, original_height],
        ],
        axis=0,
    )

    trailing = attributes[pos:] if pos != len(attributes) else None

    info = ResizableImageInfo(
        header=header,
        unknown_a=unknown_a,
        unknown_b=unknown_b,
        unknown_c=unknown_c,
        original_width=original_width,
        original_height=original_height,
        zoom=zoom,
        rotation=rotation,
        offset_x=offset_x,
        offset_y=offset_y,
        origin_x=origin_x,
        origin_y=origin_y,
        unknown_d=unknown_d,
        unknown_e=unknown_e,
        top_left=top_left,
        top_right=top_right,
        bottom_left=bottom_left,
        bottom_right=bottom_right,
        source_coords=source_coords,
        polygon_coords=polygon_coords,
        trailing_bytes=trailing,
    )

    if DEBUG:
        for k, v in info.__dict__.items():
            print(f"{k}: {v}")

    return info
