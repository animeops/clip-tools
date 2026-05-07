import struct
from typing import List, Tuple

import numpy as np
import pandas as pd
from skimage.draw import line

from clip_tools.constants import DEBUG, VectorType
from clip_tools.types import VectorPoint, VectorStroke
from clip_tools.utils import read_binary_spec


def fix_bbox_coords(bbox: List[int]) -> List[int]:
    """
    Order: top_left_x, top_left_y, bottom_right_x, bottom_right_y
    """
    mask = (1 << bbox[0].bit_length()) - 1
    bbox_fixed = [0, 0, 0, 0]
    for i in range(4):
        if bbox[i] >= (2 << 16):
            bbox_fixed[i] = -(bbox[i] ^ mask)
        else:
            bbox_fixed[i] = bbox[i]
    return bbox_fixed


def parse_vector_binary(vb: bytes) -> List[VectorStroke]:
    """Parse a vector blob into a list of stroke records.

    Pure: bytes in, structured strokes out. No DataFrame access, no
    rasterization. Renderers consume the returned list.

    Per-stroke disk layout:

    - 88-byte stroke header: magic(16) + n_ctrl/header_id(8) + bbox(16) +
      primary_color(12) + secondary_color(12) + stroke_opacity_f64(8) +
      brush_id_u32(4) + base_brush_size_f64(8) + random_seed_u32(4).
    - Per control (88 bytes main, plus extras for CURVE/BEZIER):
        +0x00 f64  point.x
        +0x08 f64  point.y
        +0x10..+0x1f i32×4 bbox
        +0x20 u32  flags (bit 12 = size locked, bit 13 = flow locked)
        +0x24 f32  pressure
        +0x28 f32  velocity
        +0x2c f32  smooth
        +0x30 f32  angle_deg (tilt azimuth / stroke-tangent direction)
        +0x34 f32  tilt_x
        +0x38 f32  size (per-control SIZE multiplier; lock bit 12 protects)
        +0x3c f32  flow (per-control FLOW multiplier; lock bit 13 protects)
        +0x40 f32  stroke_opacity (outline broadcast scale)
        +0x44 f32  outline_dx (selection-preview only)
        +0x48 f32  outline_dy (selection-preview only)
        +0x4c f32  pattern_cache_lo (runtime scratch)
        +0x50 u32 / +0x54 i32  pattern_cache_hi (runtime scratch, 8-byte qword)
      Then for CURVE: 16 extra bytes = one (f64, f64) curve handle.
      For BEZIER: 32 extra bytes = two (f64, f64) handles (in, out).
    """
    header_spec = struct.Struct(">IIII")
    uint_spec = struct.Struct(">I")
    uint2_spec = struct.Struct(">II")
    color3_spec = struct.Struct(">III")
    double_spec = struct.Struct(">d")
    double2_spec = struct.Struct(">dd")
    ctrl_main_spec = struct.Struct(">dd iiii I fffffffffff I i")

    strokes: List[VectorStroke] = []
    pos = 0
    n = len(vb)

    while pos < n - 8:
        data, pos = read_binary_spec(vb, header_spec, pos)

        if data == (88, 72, 120, 88):
            vtype = VectorType.BEZIER
        elif data == (88, 72, 104, 88):
            vtype = VectorType.CURVE
        elif data == (88, 72, 88, 88):
            vtype = VectorType.STANDARD
        else:
            break

        data, pos = read_binary_spec(vb, uint2_spec, pos)
        num_control_points = data[0]
        # data[1] is the per-stroke header id (observed values: 8321, 33).

        # 4× u32 stroke bbox (left, top, right, bottom) — read but unused
        # for rendering.
        pos += 16

        data, pos = read_binary_spec(vb, color3_spec, pos)
        cr, cg, cb = data
        color = (
            int((cr & 0xFF00) >> 8),
            int((cg & 0xFF00) >> 8),
            int((cb & 0xFF00) >> 8),
        )

        # 3 × u32 secondary / variant color slots (not used by renderer).
        pos += 12

        data, pos = read_binary_spec(vb, double_spec, pos)
        stroke_opacity = data[0]

        data, pos = read_binary_spec(vb, uint_spec, pos)
        brush_id = data[0]

        data, pos = read_binary_spec(vb, double_spec, pos)
        brush_size = data[0] * 2.0

        # Per-stroke random seed used by Random-flag effectors (spray /
        # ribbon paths) and per-stamp jitter. Stored in the blob right
        # after ``base_brush_size``. The line-stamp size/flow paths run
        # with random gating off and ignore it.
        data, pos = read_binary_spec(vb, uint_spec, pos)
        random_seed = data[0]

        points: List[VectorPoint] = []

        for _ in range(num_control_points):
            (
                (
                    px,
                    py,
                    _bbox0,
                    _bbox1,
                    _bbox2,
                    _bbox3,
                    flags,
                    pressure,
                    velocity,
                    smooth,
                    angle_deg,
                    tilt_x,
                    size,
                    flow,
                    p_stroke_opacity,
                    outline_dx,
                    outline_dy,
                    pattern_cache_lo,
                    pattern_cache_u32,
                    _pattern_cache_i32,
                ),
                pos,
            ) = read_binary_spec(vb, ctrl_main_spec, pos)

            curve = None
            if vtype == VectorType.CURVE:
                data, pos = read_binary_spec(vb, double2_spec, pos)
                curve = (data[0], data[1])
            elif vtype == VectorType.BEZIER:
                # Two (f64, f64) handles: in-handle and out-handle. Stored
                # but not yet consumed by the cubic-Bezier sampler.
                pos += 32

            points.append(
                VectorPoint(
                    x=px,
                    y=py,
                    pressure=pressure,
                    velocity=velocity,
                    smooth=smooth,
                    angle_deg=angle_deg,
                    tilt_x=tilt_x,
                    size=size,
                    flow=flow,
                    stroke_opacity=p_stroke_opacity,
                    outline_dx=outline_dx,
                    outline_dy=outline_dy,
                    pattern_cache_lo=pattern_cache_lo,
                    pattern_cache_hi=pattern_cache_u32,
                    flags=flags,
                    curve=curve,
                )
            )

        strokes.append(
            VectorStroke(
                vtype=vtype,
                color=color,
                stroke_opacity=stroke_opacity,
                brush_size=brush_size,
                brush_id=brush_id,
                points=points,
                random_seed=random_seed,
            )
        )

        if DEBUG:
            print(
                f"stroke: vtype={vtype} color={color} opacity={stroke_opacity} "
                f"brush_id={brush_id} brush_size={brush_size} npoints={len(points)}"
            )

    return strokes


def rasterize_polylines(
    strokes: List[VectorStroke],
    canvas_shape: Tuple[int, int],
    brush_styles: pd.DataFrame,
) -> np.ndarray:
    """Legacy fallback rasterizer: 1-pixel Bresenham per segment.

    Used when the line-stamp renderer can't handle a stroke (spray brushes,
    unresolvable patterns).
    """
    arr = np.zeros((canvas_shape[0], canvas_shape[1], 4), dtype=np.uint8)
    h, w = arr.shape[:2]

    for st in strokes:
        stroke_op = st.stroke_opacity
        if brush_styles is not None and len(brush_styles) > 0:
            match = brush_styles[brush_styles["MainId"] == st.brush_id]
            if len(match) and match.iloc[0]["CompositeMode"] in [27]:
                stroke_op = 0

        flat: List[Tuple[float, float]] = []
        for p in st.points:
            flat.append((p.x, p.y))
            if p.curve is not None:
                flat.append((p.curve[0], p.curve[1]))

        r, g, b = st.color
        alpha = int(stroke_op * 255)
        for i in range(len(flat) - 1):
            x0, y0 = flat[i]
            x1, y1 = flat[i + 1]
            rr, cc = line(int(y0), int(x0), int(y1), int(x1))
            in_bounds = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
            arr[rr[in_bounds], cc[in_bounds]] = (r, g, b, alpha)

    return arr
