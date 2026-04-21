import struct
from typing import Any, Dict

from clip_tools.utils import read_binary_spec


def process_offscreen_attributes(attribute: bytes) -> Dict[str, Any]:
    """Parse an Offscreen.Attribute blob.

    The blob is a self-describing TLV-like structure with three sections
    ("Parameter", "InitColor", "BlockSize") preceded by a 16-byte section-size
    table and separated by `9`-valued boundary markers.

    Layout (bytes):
        section_sizes   : 4 x u32  (self_size=16, param_len=102, initcolor_len, blocksize_len)
        boundary        : u32 = 9
        "Parameter" (utf-16be, 18 bytes)
        width, height   : 2 x u32
        cols, rows      : 2 x u32   # block grid; nblocks = cols * rows
        color_mode      : 4 x u32 = (33, 1, num_channels, 5)    # color_mode, alpha_flag, nchan, bit_depth
        block_geom      : 4 x u32 = (65536, 4, 1024, 1)         # block_bytes, nchan, subblocks/block, flag
        block_dims      : 4 x u32 = (block_w, 65536, block_h, stride)
        subblock_dims   : 4 x u32 = (subblock_w, subblock_h, 0, 0)
        boundary        : u32 = 9
        "InitColor" (utf-16be, 18 bytes)
        initcolor_magic : u32 = 20
        init_color      : 4 x u32 = (has_color, packed_rgba, nchan, nchan)
        [if has_color==1: 16 bytes extra color payload, zeros in all observed samples]
        boundary        : u32 = 9
        "BlockSize" (utf-16be, 18 bytes)
        blocksize_hdr   : 3 x u32 = (magic=12, nblocks, nchan)
        block_sizes     : nblocks x u32  # compressed byte-size of each block's data

    The `section_sizes` table allows skipping whole sections without knowing
    internal layout:
        section_sizes[0] = 16                       # size of this table itself
        section_sizes[1] = 102                      # length of Parameter section (incl. B1)
        section_sizes[2] = 42 or 58                 # length of InitColor section (42 base, +16 if has_color)
        section_sizes[3] = 34 + 4 * nblocks         # length of BlockSize section
    """
    PARAMETER_HEADER = "Parameter".encode("utf-16be")
    INITCOLOR_HEADER = "InitColor".encode("utf-16be")
    BLOCKSIZE_HEADER = "BlockSize".encode("utf-16be")

    uint_spec = struct.Struct(">I")
    uint2_spec = struct.Struct(">II")
    uint4_spec = struct.Struct(">IIII")
    uint3_spec = struct.Struct(">III")

    attr_ds: Dict[str, Any] = {}

    pos = 0
    data, pos = read_binary_spec(attribute, uint4_spec, pos)
    section_sizes = data  # (self_size=16, param_len, initcolor_len, blocksize_len)
    attr_ds["section_sizes"] = section_sizes

    # --- Parameter section ---
    data, pos = read_binary_spec(attribute, uint_spec, pos)
    # boundary marker (9)

    if attribute[pos : pos + len(PARAMETER_HEADER)] == PARAMETER_HEADER:
        pos += len(PARAMETER_HEADER)
    else:
        raise Exception("Invalid attribute: missing Parameter header")

    data, pos = read_binary_spec(attribute, uint2_spec, pos)
    attr_ds["width"] = data[0]
    attr_ds["height"] = data[1]

    data, pos = read_binary_spec(attribute, uint2_spec, pos)
    attr_ds["cols"] = data[0]
    attr_ds["rows"] = data[1]

    data, pos = read_binary_spec(attribute, uint4_spec, pos)
    # (color_mode=33, alpha_flag=1, num_channels, bit_depth_enum=5)
    attr_ds["color_mode"] = data[0]
    attr_ds["alpha_flag"] = data[1]
    attr_ds["num_channels"] = data[2]
    attr_ds["bit_depth_enum"] = data[3]

    data, pos = read_binary_spec(attribute, uint4_spec, pos)
    # (block_bytes=65536, num_channels=4, subblocks_per_block=1024, flag=1)
    attr_ds["block_bytes"] = data[0]
    attr_ds["subblocks_per_block"] = data[2]

    data, pos = read_binary_spec(attribute, uint4_spec, pos)
    # (block_width, block_bytes, block_height, block_stride)
    attr_ds["block_width"] = data[0]
    attr_ds["block_height"] = data[2]
    attr_ds["block_stride"] = data[3]

    data, pos = read_binary_spec(attribute, uint4_spec, pos)
    # (subblock_width=8, subblock_height=8, 0, 0)
    attr_ds["subblock_width"] = data[0]
    attr_ds["subblock_height"] = data[1]

    # --- InitColor section ---
    data, pos = read_binary_spec(attribute, uint_spec, pos)
    # boundary marker (9)

    if attribute[pos : pos + len(INITCOLOR_HEADER)] == INITCOLOR_HEADER:
        pos += len(INITCOLOR_HEADER)
    else:
        raise Exception("Invalid attribute: missing InitColor header")

    data, pos = read_binary_spec(attribute, uint_spec, pos)
    attr_ds["initcolor_magic"] = data[0]  # always 20 in observed samples

    data, pos = read_binary_spec(attribute, uint4_spec, pos)
    # (has_color, packed_rgba, nchan, nchan)
    attr_ds["has_init_color"] = bool(data[0])
    attr_ds["init_color"] = data[1]  # packed RGBA u32, 0xFFFFFFFF = opaque white

    if attr_ds["has_init_color"]:
        data, pos = read_binary_spec(attribute, uint4_spec, pos)
        attr_ds["init_color_extra"] = (
            data  # purpose unclear; zeros in all observed samples
        )

    # --- BlockSize section ---
    data, pos = read_binary_spec(attribute, uint_spec, pos)
    # boundary marker (9)

    if attribute[pos : pos + len(BLOCKSIZE_HEADER)] == BLOCKSIZE_HEADER:
        pos += len(BLOCKSIZE_HEADER)
    else:
        raise Exception("Invalid attribute: missing BlockSize header")

    data, pos = read_binary_spec(attribute, uint3_spec, pos)
    # (magic=12, nblocks, nchan) — nblocks == cols * rows
    attr_ds["blocksize_magic"] = data[0]
    nblocks = data[1]

    rest = attribute[pos:]
    expected_tail_bytes = 4 * nblocks
    if len(rest) != expected_tail_bytes:
        raise Exception(
            f"BlockSize tail length mismatch: got {len(rest)}, expected {expected_tail_bytes}"
        )
    attr_ds["block_sizes"] = list(struct.unpack(f">{nblocks}I", rest))

    return attr_ds
