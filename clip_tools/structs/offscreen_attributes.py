import struct
from typing import Any, Dict
from clip_tools.utils import read_binary_spec


def process_offscreen_attributes(attribute: bytes) -> Dict[str, Any]:
    PARAMETER_HEADER = "Parameter".encode("utf-16be")
    INITCOLOR_HEADER = "InitColor".encode("utf-16be")
    # TODO: Maybe this InitColor is what I need to follow for alphas...
    BLOCKSIZE_HEADER = "BlockSize".encode("utf-16be")
    parameter_header_spec = struct.Struct(">IIII")

    uint_spec = struct.Struct(">I")
    uint2_spec = struct.Struct(">II")
    uint4_spec = struct.Struct(">IIII")
    float_spec = struct.Struct(">f")
    double_spec = struct.Struct(">d")

    attr_arr = []
    attr_ds = {}

    pos = 0
    while pos < len(attribute):
        data, pos = read_binary_spec(attribute, uint4_spec, pos)
        attr_arr.append(("?", data))

        data, pos = read_binary_spec(attribute, uint_spec, pos)
        attr_arr.append(("Boundary", data[0]))

        if attribute[pos : pos + len(PARAMETER_HEADER)] == PARAMETER_HEADER:
            pos += len(PARAMETER_HEADER)
        else:
            raise Exception("Invalid attribute")

        data, pos = read_binary_spec(attribute, uint2_spec, pos)
        attr_arr.append(("Width, Height", data))
        attr_ds["width"] = data[0]
        attr_ds["height"] = data[1]

        data, pos = read_binary_spec(attribute, uint2_spec, pos)
        attr_arr.append(("Cols, Rows", data))
        attr_ds["cols"] = data[0]
        attr_ds["rows"] = data[1]

        data, pos = read_binary_spec(attribute, uint4_spec, pos)
        attr_arr.append(("?, ?, num_channels, ?", data))
        attr_ds["num_channels"] = data[2]

        data, pos = read_binary_spec(attribute, uint4_spec, pos)
        attr_arr.append(("?", data))
        data, pos = read_binary_spec(attribute, uint4_spec, pos)
        attr_arr.append(("block_width, ?, block_height, ?", data))
        attr_ds["block_width"] = data[0]
        attr_ds["block_height"] = data[2]
        data, pos = read_binary_spec(attribute, uint4_spec, pos)
        attr_arr.append(("?", data))

        data, pos = read_binary_spec(attribute, uint_spec, pos)
        attr_arr.append(("Boundary", data[0]))

        if attribute[pos : pos + len(INITCOLOR_HEADER)] == INITCOLOR_HEADER:
            pos += len(INITCOLOR_HEADER)
        else:
            raise Exception("Invalid attribute")

        data, pos = read_binary_spec(attribute, uint_spec, pos)
        attr_arr.append(("?", data))
        data, pos = read_binary_spec(attribute, uint4_spec, pos)
        attr_arr.append(("?", data))

        data, pos = read_binary_spec(attribute, uint_spec, pos)
        attr_arr.append(("Boundary", data[0]))

        if attribute[pos : pos + len(BLOCKSIZE_HEADER)] == BLOCKSIZE_HEADER:
            pos += len(BLOCKSIZE_HEADER)
        else:
            raise Exception("Invalid attribute")

        # if DEBUG:
        #     for row in attr_arr:
        #         print(row)
        #     print("")
        return attr_ds

    return attr_ds

    # return num_cols, num_rows
