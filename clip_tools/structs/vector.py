import struct
from typing import Tuple, List
import numpy as np
from skimage.draw import line
import pandas as pd
from pydantic import BaseModel

from clip_tools.constants import DEBUG
from clip_tools.constants import VectorType
from clip_tools.utils import read_binary_spec


class Point(BaseModel):
    x: int
    y: int
    opacity: float
    thickness: float


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

        try:
            assert (
                data == (88, 72, 88, 88)
                or data == (88, 72, 104, 88)
                or data == (88, 72, 120, 88)
            )
        except AssertionError:
            import pdb

            pdb.set_trace()

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
            # print("WARNING: Something is weird")
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

        # No idea why these exist
        # color_ra = (color_r & (255 << 0)) >> 0
        # color_ga = (color_g & (255 << 0)) >> 0
        # color_ba = (color_b & (255 << 0)) >> 0

        color_r = (color_r & (255 << 8)) >> 8
        color_g = (color_g & (255 << 8)) >> 8
        color_b = (color_b & (255 << 8)) >> 8
        vector_ds["color"] = [color_r, color_g, color_b, 255]

        data, pos = read_binary_spec(vector_binary_str, byte4_spec, pos)
        vector_ds["mystery_1"] = data
        # vector_ds.append(data)
        data, pos = read_binary_spec(vector_binary_str, byte4_spec, pos)
        vector_ds["mystery_2"] = data
        # vector_ds.append(data)
        data, pos = read_binary_spec(vector_binary_str, byte4_spec, pos)
        vector_ds["mystery_3"] = data
        # vector_ds.append(data)
        # no idea wtf this is

        data, pos = read_binary_spec(vector_binary_str, double_spec, pos)
        vector_ds["stroke_opacity"] = data[0]

        data, pos = read_binary_spec(vector_binary_str, uint_spec, pos)
        # data, pos = read_binary_spec(vector_binary_str, byte2_spec, pos)
        vector_ds["brush_id"] = data[0]

        if vector_ds["brush_id"] in brush_styles["MainId"].values:
            brush = brush_styles[brush_styles["MainId"] == vector_ds["brush_id"]].iloc[
                0
            ]

            if brush["CompositeMode"] in [27]:
                # print(f"WARNING: Composite mode {brush['CompositeMode']} is not supported yet")
                vector_ds["stroke_opacity"] = 0

        # vector_ds.append(data)
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
                data, pos = read_binary_spec(vector_binary_str, uint_spec, pos)
                vector_ds["strokes"][i]["mystery_0"] = data

            data, pos = read_binary_spec(vector_binary_str, double2_spec, pos)
            point_x, point_y = data
            vector_ds["strokes"][i]["point"] = [point_x, point_y]

            if vector_type == VectorType.BEZIER:
                # TODO: Currently this doesn't really do anything
                if i == num_control_points - 2 or i == 1:
                    data, pos = read_binary_spec(vector_binary_str, double2_spec, pos)
                    point_x, point_y = data
                    # vector_ds.append(("Bezier:", [point_x, point_y]))

                    data, pos = read_binary_spec(vector_binary_str, double2_spec, pos)
                    point_x, point_y = data
                    # vector_ds.append(("Bezier:", [point_x, point_y]))

                if i == num_control_points - 1:
                    data, pos = read_binary_spec(vector_binary_str, double2_spec, pos)
                    point_x, point_y = data
                    # vector_ds.append(("Bezier:", [point_x, point_y]))
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

            # Print binary
            # print("{:0b}".format(data[0]))
            rounded_corners = not (data[0] & 1)
            data = data[0] >> 1

            vector_ds["strokes"][i]["rounded_corners"] = rounded_corners
            vector_ds["strokes"][i]["num_controls_enabled"] = bin(data).count("1")
            data, pos = read_binary_spec(vector_binary_str, float2_spec, pos)
            vector_ds["strokes"][i]["mystery_1"] = data

            # data, pos = read_binary_spec(vector_binary_str, uint_spec, pos)
            data, pos = read_binary_spec(vector_binary_str, byte4_spec, pos)
            vector_ds["strokes"][i]["mystery_2"] = data

            data, pos = read_binary_spec(vector_binary_str, float2_spec, pos)
            vector_ds["strokes"][i]["mystery_3"] = data
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

            data, pos = read_binary_spec(vector_binary_str, float2_spec, pos)
            vector_ds["strokes"][i]["mystery_4"] = data
            data, pos = read_binary_spec(vector_binary_str, float_spec, pos)
            vector_ds["strokes"][i]["mystery_5"] = data
            data, pos = read_binary_spec(vector_binary_str, byte4_spec, pos)
            vector_ds["strokes"][i]["mystery_6"] = data

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

        # Draw list of points as lines with cv2
        for i in range(len(vector_ds["points"]) - 1):
            """
            cv2.line(arr, 
                     (int(vector_ds["points"][i][0]), 
                      int(vector_ds["points"][i][1])), 
                     (int(vector_ds["points"][i + 1][0]), 
                      int(vector_ds["points"][i + 1][1])), 
                    # vector_ds["color"], int(np.floor(vector_ds["stroke_width"])), lineType=cv2.LINE_AA)
                    vector_ds["color"], 2, lineType=cv2.LINE_AA)
            """
            line_color = vector_ds["color"]
            line_color[3] = int(
                vector_ds["stroke_opacity"] * vector_ds["points"][i].opacity * 255
            )

            """
            cv2.line(arr,
                    (int(vector_ds["points"][i].x), 
                     int(vector_ds["points"][i].y)),
                    (int(vector_ds["points"][i + 1].x), 
                     int(vector_ds["points"][i + 1].y)),
                    line_color,
                    # max(int(np.round(vector_ds["points"][i].thickness)), 1), 
                    2,
                    lineType=cv2.LINE_AA)
            """

            rr, cc = line(
                int(vector_ds["points"][i].y),
                int(vector_ds["points"][i].x),
                int(vector_ds["points"][i + 1].y),
                int(vector_ds["points"][i + 1].x),
            )
            arr[rr, cc] = line_color

            # Replace with scipy calls

    # Undo pre-multiplied alphas
    # alphas = arr[..., 3] / 255.0
    # arr[arr[..., 3] > 0] = vector_ds["color"]
    # arr[..., :3] = (((arr.astype(np.float32) / 255.0) * alphas[..., np.newaxis]) * 255).astype(np.uint8)[..., :3]
    # arr[..., 3] = (alphas * 255).astype(np.uint8)

    return arr
