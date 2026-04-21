import struct
from typing import Tuple, List
import numpy as np
from skimage.draw import line
import pandas as pd

from clip_tools.constants import DEBUG
from clip_tools.constants import VectorType
from clip_tools.types import Point
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


def process_vector_binary(
    vector_binary_str: bytes,
    binary_size: int,
    canvas_shape: Tuple[int, int],
    brush_styles: pd.DataFrame,
) -> np.ndarray:
    header_spec = struct.Struct(">IIII")
    uint_spec = struct.Struct(">I")
    uint2_spec = struct.Struct(">II")
    color3_spec = struct.Struct(">III")
    float_spec = struct.Struct(">f")
    float2_spec = struct.Struct(">ff")
    double_spec = struct.Struct(">d")
    double2_spec = struct.Struct(">dd")
    byte4_spec = struct.Struct(">BBBB")

    arr = np.zeros((canvas_shape[0], canvas_shape[1], 4), dtype=np.uint8)

    pos = 0

    while pos < binary_size - 8:
        vector_ds = {}

        data, pos = read_binary_spec(vector_binary_str, header_spec, pos)

        assert (
            data == (88, 72, 88, 88)
            or data == (88, 72, 104, 88)
            or data == (88, 72, 120, 88)
        )

        if data == (88, 72, 120, 88):
            vector_type = VectorType.BEZIER

        elif data == (88, 72, 104, 88):
            vector_type = VectorType.CURVE

        elif data == (88, 72, 88, 88):
            vector_type = VectorType.STANDARD

        vector_ds["vector_type"] = vector_type

        data, pos = read_binary_spec(vector_binary_str, uint2_spec, pos)
        num_control_points = data[0]
        vector_ds["num_control_points"] = num_control_points
        vector_ds["header_id"] = data[1]

        try:
            assert data[1] in [8321, 33]
        except AssertionError:
            pass

        data, pos = read_binary_spec(vector_binary_str, uint2_spec, pos)
        top_left = data
        data, pos = read_binary_spec(vector_binary_str, uint2_spec, pos)
        bottom_right = data
        global_bbox = [top_left[0], top_left[1], bottom_right[0], bottom_right[1]]
        global_bbox = fix_bbox_coords(global_bbox)
        vector_ds["global_bbox"] = global_bbox

        data, pos = read_binary_spec(vector_binary_str, color3_spec, pos)
        color_r, color_g, color_b = data
        color_r = (color_r & (255 << 8)) >> 8
        color_g = (color_g & (255 << 8)) >> 8
        color_b = (color_b & (255 << 8)) >> 8
        vector_ds["color"] = [color_r, color_g, color_b, 255]

        # 3 x u32 following the primary color. For custom/colored brushes, these
        # byte-for-byte duplicate the three color uint32s (a redundant color
        # copy). For default black-ink strokes they become a hardcoded
        # (0xFFFFFFFF, 0xAFAFAFAF, 0x00000000) — likely a pressure-falloff /
        # tint curve (255, 175, 0). Kept as raw bytes for now.
        data, pos = read_binary_spec(vector_binary_str, byte4_spec, pos)
        vector_ds["color_variant_r"] = data
        data, pos = read_binary_spec(vector_binary_str, byte4_spec, pos)
        vector_ds["color_variant_g"] = data
        data, pos = read_binary_spec(vector_binary_str, byte4_spec, pos)
        vector_ds["color_variant_b"] = data

        data, pos = read_binary_spec(vector_binary_str, double_spec, pos)
        vector_ds["stroke_opacity"] = data[0]

        data, pos = read_binary_spec(vector_binary_str, uint_spec, pos)
        vector_ds["brush_id"] = data[0]

        if vector_ds["brush_id"] in brush_styles["MainId"].values:
            brush = brush_styles[brush_styles["MainId"] == vector_ds["brush_id"]].iloc[
                0
            ]

            if brush["CompositeMode"] in [27]:
                vector_ds["stroke_opacity"] = 0

        data, pos = read_binary_spec(vector_binary_str, double_spec, pos)
        vector_ds["stroke_width"] = data[0]
        vector_ds["brush_size"] = data[0] * 2.0

        first_point = True
        vector_id = None
        if vector_type == VectorType.BEZIER:
            num_control_points += 1

        vector_ds["strokes"] = [{} for _ in range(num_control_points)]
        vector_ds["points"] = []

        for i in range(num_control_points):
            # Start the loop
            data, pos = read_binary_spec(vector_binary_str, byte4_spec, pos)
            vector_ds["strokes"][i]["stroke_id"] = data

            if first_point:
                vector_id = data
            else:
                # Monotonically increasing u32 across control points.
                # Inferred meaning: cumulative arc-length or sample-time along
                # the stroke (e.g. 0, 4, 18, 47, 100, 131, ..., 475 across 22
                # points on a single stroke).
                data, pos = read_binary_spec(vector_binary_str, uint_spec, pos)
                vector_ds["strokes"][i]["cumulative_param"] = data

            data, pos = read_binary_spec(vector_binary_str, double2_spec, pos)
            point_x, point_y = data
            vector_ds["strokes"][i]["point"] = [point_x, point_y]

            if vector_type == VectorType.BEZIER:
                # Bezier control-handle bytes are consumed but not yet wired
                # into the sampler.
                if i == num_control_points - 2 or i == 1:
                    data, pos = read_binary_spec(vector_binary_str, double2_spec, pos)
                    point_x, point_y = data

                    data, pos = read_binary_spec(vector_binary_str, double2_spec, pos)
                    point_x, point_y = data

                if i == num_control_points - 1:
                    data, pos = read_binary_spec(vector_binary_str, double2_spec, pos)
                    point_x, point_y = data
                    continue

            if vector_type == VectorType.CURVE and not first_point:
                data, pos = read_binary_spec(vector_binary_str, double2_spec, pos)
                point_x, point_y = data
                vector_ds["strokes"][i]["curve"] = [point_x, point_y]

            data, pos = read_binary_spec(vector_binary_str, uint2_spec, pos)
            top_left = data
            data, pos = read_binary_spec(vector_binary_str, uint2_spec, pos)
            bottom_right = data
            stroke_bbox = [top_left[0], top_left[1], bottom_right[0], bottom_right[1]]
            stroke_bbox = fix_bbox_coords(stroke_bbox)
            vector_ds["strokes"][i]["stroke_bbox"] = stroke_bbox

            data, pos = read_binary_spec(vector_binary_str, uint_spec, pos)

            rounded_corners = not (data[0] & 1)
            data = data[0] >> 1

            vector_ds["strokes"][i]["rounded_corners"] = rounded_corners
            vector_ds["strokes"][i]["num_controls_enabled"] = bin(data).count("1")
            # 2 floats in [0, 1]. First float matches a classic pen-pressure
            # curve (0.1 → 1.0 → 0.0 over a stroke's length). Second float
            # drifts slowly; likely tilt or velocity. Names inferred, not
            # confirmed from file spec.
            data, pos = read_binary_spec(vector_binary_str, float2_spec, pos)
            vector_ds["strokes"][i]["pressure_and_tilt"] = data

            # float32 in [0, 1]. Small positives that roughly track pressure:
            # ~0.19 during steady drawing, drops to ~0.02 at stroke ends.
            # Best guess: brush step / spacing per sample. (Was previously
            # mis-typed as 4 raw bytes.)
            data, pos = read_binary_spec(vector_binary_str, float_spec, pos)
            vector_ds["strokes"][i]["brush_step"] = data[0]

            # Always (0.0, 1.0) across every observed stroke in every file —
            # looks like (pressure_min, pressure_max) normalization bounds.
            data, pos = read_binary_spec(vector_binary_str, float2_spec, pos)
            vector_ds["strokes"][i]["pressure_range"] = data
            # vector_ds.append(data)
            data, pos = read_binary_spec(vector_binary_str, float2_spec, pos)
            vector_ds["strokes"][i]["stroke_width"] = data[0]
            vector_ds["strokes"][i]["stroke_opacity"] = data[1]

            vector_ds["points"].append(
                Point(
                    x=vector_ds["strokes"][i]["point"][0],
                    y=vector_ds["strokes"][i]["point"][1],
                    opacity=vector_ds["strokes"][i]["stroke_opacity"],
                    thickness=vector_ds["strokes"][i]["stroke_width"],
                )
            )
            if "curve" in vector_ds["strokes"][i]:
                vector_ds["points"].append(
                    Point(
                        x=vector_ds["strokes"][i]["curve"][0],
                        y=vector_ds["strokes"][i]["curve"][1],
                        opacity=vector_ds["strokes"][i]["stroke_opacity"],
                        thickness=vector_ds["strokes"][i]["stroke_width"],
                    )
                )

            # Always (0.0, 0.0) in every observed sample — default tilt vector.
            data, pos = read_binary_spec(vector_binary_str, float2_spec, pos)
            vector_ds["strokes"][i]["tilt_xy"] = data
            # Always 0.0 — default stroke rotation.
            data, pos = read_binary_spec(vector_binary_str, float_spec, pos)
            vector_ds["strokes"][i]["rotation"] = data
            # float32 in [0, 1], uniformly distributed; always 0.0 on first
            # point. Best guess: per-sample texture random seed / UV phase
            # offset used to de-tile brush stamps along the stroke. (Was
            # previously mis-typed as 4 raw bytes.)
            data, pos = read_binary_spec(vector_binary_str, float_spec, pos)
            vector_ds["strokes"][i]["texture_seed"] = data[0]

            if first_point:
                first_point = False

        if vector_type == VectorType.STANDARD or vector_type == VectorType.CURVE:
            data, pos = read_binary_spec(vector_binary_str, byte4_spec, pos)
            vector_ds["mystery_4"] = data
            data, pos = read_binary_spec(vector_binary_str, uint_spec, pos)
            vector_ds["mystery_5"] = data

        if vector_type == VectorType.CURVE:
            data, pos = read_binary_spec(vector_binary_str, header_spec, pos)
            vector_ds["mystery_6"] = data

        if DEBUG:
            for key, value in vector_ds.items():
                if key == "strokes":
                    for stroke in value:
                        for k, v in stroke.items():
                            print(f"  {k}: {v}")
                        print("")
                else:
                    print(f"{key}: {value}")

        for i in range(len(vector_ds["points"]) - 1):
            line_color = vector_ds["color"]
            line_color[3] = int(
                vector_ds["stroke_opacity"] * vector_ds["points"][i].opacity * 255
            )

            rr, cc = line(
                int(vector_ds["points"][i].y),
                int(vector_ds["points"][i].x),
                int(vector_ds["points"][i + 1].y),
                int(vector_ds["points"][i + 1].x),
            )
            h, w = arr.shape[:2]
            in_bounds = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
            arr[rr[in_bounds], cc[in_bounds]] = line_color

    return arr
