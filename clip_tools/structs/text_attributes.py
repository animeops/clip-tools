import struct
from typing import Dict, Any
from clip_tools.utils import read_binary_spec
from clip_tools.constants import DEBUG
from clip_tools.constants import (
    TextStylingType,
    TextWindowType,
    TextHollowType,
    TextAlignmentType,
)


def process_text_attributes(attributes: bytes) -> Dict[str, Any]:
    uint_spec = struct.Struct("<I")
    uint2_spec = struct.Struct("<II")
    uint4_spec = struct.Struct("<IIII")
    float_spec = struct.Struct("<f")
    float2_spec = struct.Struct("<ff")
    float4_spec = struct.Struct("<ffff")
    double_spec = struct.Struct("<d")
    double2_spec = struct.Struct("<dd")

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

    if data[0] == 11:
        attr_arr.append(("header", data[0]))
        attr_ds["header"] = data[0]
    else:
        raise Exception(f"Invalid header (expected 11, got {data})")

    data, pos = read_binary_spec(attributes, uint_spec, pos)
    attr_arr.append(("? length", data[0]))
    attr_ds["some_length"] = data[0]

    data, pos = read_binary_spec(attributes, uint_spec, pos)
    attr_arr.append(("num_font_styles", data[0]))
    attr_ds["num_font_styles"] = data[0]

    data, pos = read_binary_spec(attributes, uint_spec, pos)
    attr_arr.append(("?", data))

    for i in range(attr_ds["num_font_styles"]):
        data, pos = read_binary_spec(attributes, uint_spec, pos)
        attr_arr.append(("num_chars_0", data[0]))
        attr_ds["num_chars_0"] = data[0]

        data, pos = read_binary_spec(attributes, uint_spec, pos)
        attr_arr.append((f"num_bytes_{i}", data[0]))
        attr_ds[f"num_bytes_{i}"] = data[0]

        data, pos = read_binary_spec(attributes, byte2_spec, pos)

        try:
            text_styling_type = TextStylingType(data[0])
            attr_arr.append(("text_styling_type", text_styling_type))
        except Exception:
            attr_arr.append(("text_styling_type", data[0]))

        try:
            text_window_type = TextWindowType(data[1])
            attr_arr.append(("text_window_type", text_window_type))
        except Exception:
            text_window_type = data[1]
            attr_arr.append(("text_window_type", data[1]))

        byte_string = attributes[pos : pos + (attr_ds[f"num_bytes_{i}"] - 6)]
        attr_arr.append(("byte_string", byte_string))
        pos += attr_ds[f"num_bytes_{i}"] - 6

    data, pos = read_binary_spec(attributes, uint_spec, pos)
    attr_arr.append(("?", data[0]))

    data, pos = read_binary_spec(attributes, uint2_spec, pos)
    attr_arr.append(("num_chunks_0", data[0]))
    attr_ds["num_chunks_0"] = data[0]
    attr_arr.append(("?", data[1]))

    for i in range(attr_ds["num_chunks_0"]):
        data, pos = read_binary_spec(attributes, uint_spec, pos)
        attr_arr.append((f"chunk_num_chars_{i}", data[0]))

        data, pos = read_binary_spec(attributes, uint_spec, pos)
        attr_arr.append(("?", data))

        data, pos = read_binary_spec(attributes, double2_spec, pos)
        attr_arr.append(("text_box_scale_x", data[0]))
        attr_arr.append(("text_box_scale_y", data[1]))

        data, pos = read_binary_spec(attributes, uint_spec, pos)
        attr_arr.append(("?", data))

    data, pos = read_binary_spec(attributes, uint_spec, pos)
    attr_arr.append(("?", data))
    data, pos = read_binary_spec(attributes, uint2_spec, pos)
    attr_arr.append(("?", data))

    data, pos = read_binary_spec(attributes, uint2_spec, pos)
    try:
        assert data == (1, 0)
        # attr_arr.append(("chunk_boundary", data))
    except Exception:
        attr_arr.append(("chunk_boundary", data))
        if DEBUG:
            for row in attr_arr:
                print(row)

    data, pos = read_binary_spec(attributes, uint_spec, pos)
    attr_arr.append(("num_chars_1", data[0]))

    data, pos = read_binary_spec(attributes, uint_spec, pos)
    attr_arr.append(("?", data))

    data, pos = read_binary_spec(attributes, byte2_spec, pos)
    attr_arr.append(("? (byte2)", data))

    data, pos = read_binary_spec(attributes, uint2_spec, pos)
    attr_arr.append(("?", data))
    data, pos = read_binary_spec(attributes, uint2_spec, pos)
    try:
        assert data == (1, 0)
        # attr_arr.append(("chunk_boundary", data))
    except Exception:
        attr_arr.append(("chunk_boundary", data))
        if DEBUG:
            for row in attr_arr:
                print(row)

    # data, pos = read_binary_spec(attributes, uint4_spec, pos)
    # attr_arr.append(("?", data))

    data, pos = read_binary_spec(attributes, uint_spec, pos)
    attr_arr.append(("num_chars_2", data[0]))

    data, pos = read_binary_spec(attributes, uint_spec, pos)
    attr_arr.append(("?", data))

    data, pos = read_binary_spec(attributes, byte2_spec, pos)
    attr_arr.append(("? (byte2)", data))

    data, pos = read_binary_spec(attributes, double2_spec, pos)
    attr_arr.append(("?", data[0]))
    attr_arr.append(("line_spacing", data[1]))

    for i in range(attr_ds["num_chunks_0"]):
        data, pos = read_binary_spec(attributes, uint4_spec, pos)
        attr_arr.append(("?_chunk?", data))

        data, pos = read_binary_spec(attributes, uint4_spec, pos)
        attr_arr.append(("?_chunk?", data))

    if data[3] == 18:
        data, pos = read_binary_spec(attributes, uint2_spec, pos)
        try:
            assert data == (1, 0)
        except Exception:
            attr_arr.append(("chunk_boundary", data))
            if DEBUG:
                pass

        data, pos = read_binary_spec(attributes, uint_spec, pos)
        attr_arr.append(("num_chars_4", data[0]))
        attr_ds["num_chars"] = data[0]

        data, pos = read_binary_spec(attributes, uint_spec, pos)
        attr_arr.append(("?", data))

        data, pos = read_binary_spec(attributes, byte2_spec, pos)
        attr_arr.append(("? (byte2)", data))

        data, pos = read_binary_spec(attributes, uint4_spec, pos)
        attr_arr.append(("?", data))

        data, pos = read_binary_spec(attributes, uint_spec, pos)
        attr_arr.append(("?", data))

        data, pos = read_binary_spec(attributes, uint_spec, pos)
        attr_arr.append(("some_font_length", data[0]))
        attr_ds["some_font_length"] = data[0]

        some_font_name = attributes[pos : pos + attr_ds["some_font_length"]]
        attr_arr.append(("some_font_name", some_font_name))
        pos += attr_ds["some_font_length"]

        data, pos = read_binary_spec(attributes, uint2_spec, pos)
        attr_arr.append(("?", data))
        data, pos = read_binary_spec(attributes, uint2_spec, pos)
        attr_arr.append(("?", data))
        data, pos = read_binary_spec(attributes, uint2_spec, pos)
        try:
            assert data == (1, 0)
            # attr_arr.append(("chunk_boundary", data))
        except Exception:
            attr_arr.append(("chunk_boundary", data))
            if DEBUG:
                for row in attr_arr:
                    print(row)
    else:
        try:
            # while data[:3] != (18, 18, 1):
            while data[1:3] != (18, 1):
                data, pos = read_binary_spec(attributes, uint4_spec, pos)
                attr_arr.append(("?", data))
        except Exception:
            if DEBUG:
                for row in attr_arr:
                    print(row)

    data, pos = read_binary_spec(attributes, uint_spec, pos)
    attr_arr.append(("num_chars_5", data[0]))

    data, pos = read_binary_spec(attributes, uint_spec, pos)
    attr_arr.append(("?", data))

    data, pos = read_binary_spec(attributes, byte2_spec, pos)

    try:
        text_hollow_type = TextHollowType(data[0])
        attr_arr.append(("text_hollow_type", text_hollow_type))
    except Exception:
        attr_arr.append(("text_hollow_type", data[0]))

    attr_arr.append(("?", data[1]))

    data, pos = read_binary_spec(attributes, uint4_spec, pos)
    attr_arr.append(("?", data))

    data, pos = read_binary_spec(attributes, uint_spec, pos)
    attr_arr.append(("?", data))

    data, pos = read_binary_spec(attributes, uint_spec, pos)
    attr_arr.append(("font_name_length", data[0]))
    attr_ds["font_name_length"] = data[0]

    try:
        font_name = attributes[pos : pos + attr_ds["font_name_length"]].decode("utf-8")
        attr_arr.append(("font_name", font_name))
        pos += attr_ds["font_name_length"]
    except Exception:
        if DEBUG:
            for row in attr_arr:
                print(row)

    data, pos = read_binary_spec(attributes, uint2_spec, pos)
    attr_arr.append(("?", data))

    data, pos = read_binary_spec(attributes, uint_spec, pos)
    attr_arr.append(("font_size", data[0]))

    data, pos = read_binary_spec(attributes, uint_spec, pos)
    attr_arr.append(("?", data))

    data, pos = read_binary_spec(attributes, uint4_spec, pos)
    attr_arr.append(("?", data))

    data, pos = read_binary_spec(attributes, color3_spec, pos)
    color_r, color_g, color_b = data

    # No idea why these exist
    # color_ra = (color_r & (255 << 0)) >> 0
    # color_ga = (color_g & (255 << 0)) >> 0
    # color_ba = (color_b & (255 << 0)) >> 0

    color_r = (color_r & (255 << 8)) >> 8
    color_g = (color_g & (255 << 8)) >> 8
    color_b = (color_b & (255 << 8)) >> 8
    attr_arr.append(("color", [color_r, color_g, color_b, 255]))

    # No idea why these exist
    # color_ra = (color_r & (255 << 0)) >> 0
    # color_ga = (color_g & (255 << 0)) >> 0
    # color_ba = (color_b & (255 << 0)) >> 0

    data, pos = read_binary_spec(attributes, uint2_spec, pos)
    attr_arr.append(("?", data))

    data, pos = read_binary_spec(attributes, byte4_spec, pos)

    try:
        text_alignment_type = TextAlignmentType(data[0])
        attr_arr.append(("text_alignment_type", text_alignment_type))
    except Exception:
        attr_arr.append(("text_alignment_type", data[0]))

    attr_arr.append(("?", data[1]))
    attr_arr.append(("?", data[2]))
    attr_arr.append(("?", data[3]))

    data, pos = read_binary_spec(attributes, uint_spec, pos)
    attr_arr.append(("?", data))

    data, pos = read_binary_spec(attributes, uint4_spec, pos)
    attr_arr.append(("?", data))

    data, pos = read_binary_spec(attributes, uint4_spec, pos)
    attr_arr.append(("?", data))
    data, pos = read_binary_spec(attributes, uint4_spec, pos)
    attr_arr.append(("?", data))

    data, pos = read_binary_spec(attributes, uint4_spec, pos)
    attr_arr.append(("?", data))
    data, pos = read_binary_spec(attributes, uint4_spec, pos)
    attr_arr.append(("?", data))

    data, pos = read_binary_spec(attributes, uint_spec, pos)
    attr_arr.append(("?", data))
    data, pos = read_binary_spec(attributes, uint2_spec, pos)
    attr_ds["general_offset_x"] = data[0]
    attr_ds["general_offset_y"] = data[1]
    attr_arr.append(("general_offset_x", data[0]))
    attr_arr.append(("general_offset_y", data[1]))

    data, pos = read_binary_spec(attributes, uint4_spec, pos)
    attr_arr.append(("?", data))
    data, pos = read_binary_spec(attributes, uint4_spec, pos)
    attr_arr.append(("?", data))

    if DEBUG:
        for row in attr_arr:
            print(row)

    return attr_ds
