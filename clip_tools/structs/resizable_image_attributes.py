from typing import Dict, Any
import struct
import numpy as np

from clip_tools.utils import read_binary_spec
from clip_tools.constants import DEBUG


def process_resizable_image_attributes(attributes: bytes) -> Dict[str, Any]:
    uint_spec = struct.Struct(">I")
    uint2_spec = struct.Struct(">II")
    uint4_spec = struct.Struct(">IIII")
    float_spec = struct.Struct(">f")
    float2_spec = struct.Struct(">ff")
    float4_spec = struct.Struct(">ffff")
    double_spec = struct.Struct(">d")
    double2_spec = struct.Struct(">dd")

    # spec for byte (0, 255)
    byte2_spec = struct.Struct("<BB")
    byte4_spec = struct.Struct("<BBBB")
    byte16_spec = struct.Struct("<BBBBBBBBBBBBBBBB")

    ushort3_spec = struct.Struct("<HHH")

    # color3_spec = struct.Struct('>III')
    color3_spec = struct.Struct(">III")

    # enum to map int to text

    attr_arr = []
    attr_ds = {}

    pos = 0
    data, pos = read_binary_spec(attributes, uint_spec, pos)

    if data[0] == 120:
        attr_arr.append(("header", data[0]))
        attr_ds["header"] = data[0]
    else:
        raise Exception(f"Invalid header (expected 11, got {data})")

    data, pos = read_binary_spec(attributes, uint_spec, pos)
    attr_arr.append(("?", data))

    data, pos = read_binary_spec(attributes, uint4_spec, pos)
    attr_arr.append(("?", data))

    data, pos = read_binary_spec(attributes, uint2_spec, pos)
    attr_arr.append(("?", data))

    data, pos = read_binary_spec(attributes, uint2_spec, pos)
    attr_arr.append(("original_width, original_height", data))
    attr_ds["original_width"] = data[0]
    attr_ds["original_height"] = data[1]

    data, pos = read_binary_spec(attributes, double2_spec, pos)
    attr_arr.append(("zoom_x, zoom_y", data))
    attr_ds["zoom"] = data

    data, pos = read_binary_spec(attributes, double_spec, pos)
    attr_arr.append(("rotation", data))
    attr_ds["rotation"] = data[0]

    data, pos = read_binary_spec(attributes, double2_spec, pos)
    attr_arr.append(("offset", data))
    attr_ds["offset_x"] = data[0]
    attr_ds["offset_y"] = data[1]

    data, pos = read_binary_spec(attributes, double2_spec, pos)
    attr_arr.append(("origin", data))
    attr_ds["origin_x"] = data[0]
    attr_ds["origi_y"] = data[1]

    data, pos = read_binary_spec(attributes, uint4_spec, pos)
    attr_arr.append(("?", data))

    data, pos = read_binary_spec(attributes, uint2_spec, pos)
    attr_arr.append(("?", data))

    data, pos = read_binary_spec(attributes, double2_spec, pos)
    attr_arr.append(("top_left", data))
    attr_ds["top_left"] = data

    data, pos = read_binary_spec(attributes, double2_spec, pos)
    attr_arr.append(("top_right", data))
    attr_ds["top_right"] = data

    data, pos = read_binary_spec(attributes, double2_spec, pos)
    attr_arr.append(("bottom_left", data))
    attr_ds["bottom_left"] = data

    data, pos = read_binary_spec(attributes, double2_spec, pos)
    attr_arr.append(("bottom_right", data))
    attr_ds["bottom_right"] = data

    attr_ds["polygon_coords"] = np.stack(
        [
            attr_ds["top_left"],
            attr_ds["top_right"],
            attr_ds["bottom_right"],
            attr_ds["bottom_left"],
        ],
        axis=0,
    )

    attr_ds["source_coords"] = np.stack(
        [
            [0, 0],
            [attr_ds["original_width"], 0],
            [attr_ds["original_width"], attr_ds["original_height"]],
            [0, attr_ds["original_height"]],
        ],
        axis=0,
    )

    if DEBUG:
        for row in attr_arr:
            print(row)

    return attr_ds
