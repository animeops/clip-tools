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
    """
    header_spec = struct.Struct(">IIII")
    uint_spec = struct.Struct(">I")
    uint2_spec = struct.Struct(">II")
    color3_spec = struct.Struct(">III")
    float_spec = struct.Struct(">f")
    float2_spec = struct.Struct(">ff")
    double_spec = struct.Struct(">d")
    double2_spec = struct.Struct(">dd")
    byte4_spec = struct.Struct(">BBBB")

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

        # bbox (top_left, bottom_right) — read but unused for rendering.
        pos += 16

        data, pos = read_binary_spec(vb, color3_spec, pos)
        cr, cg, cb = data
        color = (
            int((cr & 0xFF00) >> 8),
            int((cg & 0xFF00) >> 8),
            int((cb & 0xFF00) >> 8),
        )

        # 3 × u32 color variants — see clip_tools/unknowns.md.
        pos += 12

        data, pos = read_binary_spec(vb, double_spec, pos)
        stroke_opacity = data[0]

        data, pos = read_binary_spec(vb, uint_spec, pos)
        brush_id = data[0]

        data, pos = read_binary_spec(vb, double_spec, pos)
        brush_size = data[0] * 2.0

        ctrl = (
            num_control_points + 1 if vtype == VectorType.BEZIER else num_control_points
        )
        first = True
        points: List[VectorPoint] = []

        for i in range(ctrl):
            # Per-point stroke_id (4 bytes) and cumulative_param (uint32 except first).
            pos += 4
            if not first:
                pos += 4

            data, pos = read_binary_spec(vb, double2_spec, pos)
            px, py = data

            if vtype == VectorType.BEZIER:
                # Bezier control handles — captured but not yet wired into
                # the sampler. See clip_tools/unknowns.md.
                if i == 1 or i == ctrl - 2:
                    pos += 32
                if i == ctrl - 1:
                    pos += 16
                    if first:
                        first = False
                    continue

            curve = None
            if vtype == VectorType.CURVE and not first:
                data, pos = read_binary_spec(vb, double2_spec, pos)
                curve = (data[0], data[1])

            # Per-point bbox + flags.
            pos += 16
            pos += 4

            data, pos = read_binary_spec(vb, float2_spec, pos)
            pressure = data[0]

            data, pos = read_binary_spec(vb, float_spec, pos)
            size_modulation = data[0]

            # pressure_range — always (0.0, 1.0); see clip_tools/unknowns.md.
            pos += 8

            data, pos = read_binary_spec(vb, float2_spec, pos)
            width_factor, opacity_factor = data

            # tilt_xy + rotation + texture_seed — see clip_tools/unknowns.md.
            pos += 8 + 4 + 4

            points.append(
                VectorPoint(
                    x=px,
                    y=py,
                    pressure=pressure,
                    width_factor=width_factor,
                    opacity_factor=opacity_factor,
                    size_modulation=size_modulation,
                    curve=curve,
                )
            )

            if first:
                first = False

        # Stroke trailers — see clip_tools/unknowns.md.
        if vtype == VectorType.STANDARD or vtype == VectorType.CURVE:
            pos += 8  # tail_id (4) + tail_param (4)
        if vtype == VectorType.CURVE:
            pos += 16  # curve_trailer

        strokes.append(
            VectorStroke(
                vtype=vtype,
                color=color,
                stroke_opacity=stroke_opacity,
                brush_size=brush_size,
                brush_id=brush_id,
                points=points,
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

        flat: List[Tuple[float, float, float]] = []
        for p in st.points:
            flat.append((p.x, p.y, p.opacity_factor))
            if p.curve is not None:
                flat.append((p.curve[0], p.curve[1], p.opacity_factor))

        r, g, b = st.color
        for i in range(len(flat) - 1):
            x0, y0, op0 = flat[i]
            x1, y1, _ = flat[i + 1]
            alpha = int(stroke_op * op0 * 255)
            rr, cc = line(int(y0), int(x0), int(y1), int(x1))
            in_bounds = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
            arr[rr[in_bounds], cc[in_bounds]] = (r, g, b, alpha)

    return arr
