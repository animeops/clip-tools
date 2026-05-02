import struct
from typing import Any, Dict

import numpy as np

from clip_tools.constants import DEBUG
from clip_tools.utils import read_binary_spec


def process_resizable_image_attributes(attributes: bytes) -> Dict[str, Any]:
    """Decode a ``ResizableImageInfo`` blob.

    Used by ClipStudio's "transform" feature: a layer's pixel content
    (``original_width × original_height``) is mapped into a four-corner
    polygon on the canvas via a homography. ``clip_layer.py`` only consumes
    ``source_coords`` and ``polygon_coords``; everything else is exposed for
    completeness.

    No sample exercises this code path in the current corpus
    (``tests/test_data/test000–003.clip`` and ``wn_*.clip``), so the field
    layout below is taken as-is from the legacy parser and the unknowns are
    left as raw integers. See ``unknowns.md``.

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

    out: Dict[str, Any] = {}
    pos = 0

    (header,), pos = read_binary_spec(attributes, uint_be, pos)
    if header != 120:
        raise ValueError(
            f"Invalid resizable-image attributes header: expected 120, got {header}"
        )
    out["header"] = header

    (out["unknown_a"],), pos = read_binary_spec(attributes, uint_be, pos)
    out["unknown_b"], pos = read_binary_spec(attributes, uint4_be, pos)
    out["unknown_c"], pos = read_binary_spec(attributes, uint2_be, pos)

    (out["original_width"], out["original_height"]), pos = read_binary_spec(
        attributes, uint2_be, pos
    )

    (zoom_x, zoom_y), pos = read_binary_spec(attributes, double2_be, pos)
    out["zoom"] = (zoom_x, zoom_y)

    (out["rotation"],), pos = read_binary_spec(attributes, double_be, pos)

    (out["offset_x"], out["offset_y"]), pos = read_binary_spec(
        attributes, double2_be, pos
    )
    (out["origin_x"], out["origin_y"]), pos = read_binary_spec(
        attributes, double2_be, pos
    )

    out["unknown_d"], pos = read_binary_spec(attributes, uint4_be, pos)
    out["unknown_e"], pos = read_binary_spec(attributes, uint2_be, pos)

    out["top_left"], pos = read_binary_spec(attributes, double2_be, pos)
    out["top_right"], pos = read_binary_spec(attributes, double2_be, pos)
    out["bottom_left"], pos = read_binary_spec(attributes, double2_be, pos)
    out["bottom_right"], pos = read_binary_spec(attributes, double2_be, pos)

    out["polygon_coords"] = np.stack(
        [
            out["top_left"],
            out["top_right"],
            out["bottom_right"],
            out["bottom_left"],
        ],
        axis=0,
    )
    out["source_coords"] = np.stack(
        [
            [0, 0],
            [out["original_width"], 0],
            [out["original_width"], out["original_height"]],
            [0, out["original_height"]],
        ],
        axis=0,
    )

    if pos != len(attributes):
        out["trailing_bytes"] = attributes[pos:]

    if DEBUG:
        for k, v in out.items():
            print(f"{k}: {v}")

    return out
