"""Parser for the per-Offscreen `Attribute` blob (the section-sized,
named-section container that holds canvas/block geometry, init-color, and
block-size tables for a raster Offscreen)."""

import struct
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from clip_tools.utils import read_binary_spec


@dataclass
class OffscreenAttributes:
    """Decoded Offscreen.Attribute blob.

    See `process_offscreen_attributes` for layout details. Names like
    `color_mode`, `alpha_flag`, `bit_depth_enum`, `initcolor_magic` are
    positional inferences: their values are constant across every observed
    sample, so the labels are educated guesses and could be confirmed only
    by samples with different color modes / bit depths.
    """

    section_sizes: Tuple[
        int, int, int, int
    ]  # (self_size=16, param_len, initcolor_len, blocksize_len)

    # --- Parameter section ---
    width: int
    height: int
    cols: int
    rows: int
    color_mode: int  # observed always 33
    alpha_flag: int  # observed always 1
    num_channels: int
    bit_depth_enum: int  # observed always 5
    block_bytes: int  # observed always 256*256
    subblocks_per_block: int  # observed always 32*32
    block_width: int
    block_height: int
    block_stride: int
    subblock_width: int
    subblock_height: int

    # --- InitColor section ---
    initcolor_magic: int  # observed always 20
    has_init_color: bool
    init_color: int  # packed RGBA u32; 0xFFFFFFFF = opaque white
    init_color_extra: Optional[Tuple[int, int, int, int]] = (
        None  # 4 channels (high byte of each u32)
    )
    init_color_extra_raw: Optional[Tuple[int, int, int, int]] = None  # raw u32s

    # --- BlockSize section ---
    blocksize_magic: int = 0
    block_sizes: List[int] = field(default_factory=list)


def process_offscreen_attributes(attribute: bytes) -> OffscreenAttributes:
    """Parse an Offscreen.Attribute blob.

    The blob is a self-describing TLV-like structure with three sections
    ("Parameter", "InitColor", "BlockSize") preceded by a 16-byte section-size
    table and separated by `9`-valued boundary markers.

    Layout (bytes)::

        section_sizes   : 4 x u32  (self_size=16, param_len=102, initcolor_len, blocksize_len)
        boundary        : u32 = 9
        "Parameter" (utf-16be, 18 bytes)
        width, height   : 2 x u32
        cols, rows      : 2 x u32   # block grid; nblocks = cols * rows
        color_mode      : 4 x u32 = (33, 1, num_channels, 5)
        block_geom      : 4 x u32 = (65536, 4, 1024, 1)
        block_dims      : 4 x u32 = (block_w, 65536, block_h, stride)
        subblock_dims   : 4 x u32 = (subblock_w, subblock_h, 0, 0)
        boundary        : u32 = 9
        "InitColor" (utf-16be, 18 bytes)
        initcolor_magic : u32 = 20
        init_color      : 4 x u32 = (has_color, packed_rgba, nchan, nchan)
        [if has_color==1: 16-byte extra color payload]
        boundary        : u32 = 9
        "BlockSize" (utf-16be, 18 bytes)
        blocksize_hdr   : 3 x u32 = (magic=12, nblocks, nchan)
        block_sizes     : nblocks x u32

    See `unknowns.md` for fields whose semantics are inferred but not
    confirmed (color_mode/alpha_flag/bit_depth_enum/initcolor_magic).
    """
    PARAMETER_HEADER = "Parameter".encode("utf-16be")
    INITCOLOR_HEADER = "InitColor".encode("utf-16be")
    BLOCKSIZE_HEADER = "BlockSize".encode("utf-16be")

    uint_spec = struct.Struct(">I")
    uint2_spec = struct.Struct(">II")
    uint3_spec = struct.Struct(">III")
    uint4_spec = struct.Struct(">IIII")

    pos = 0
    section_sizes, pos = read_binary_spec(attribute, uint4_spec, pos)

    # --- Parameter section ---
    _boundary, pos = read_binary_spec(attribute, uint_spec, pos)
    if attribute[pos : pos + len(PARAMETER_HEADER)] != PARAMETER_HEADER:
        raise ValueError("Invalid attribute: missing Parameter header")
    pos += len(PARAMETER_HEADER)

    (width, height), pos = read_binary_spec(attribute, uint2_spec, pos)
    (cols, rows), pos = read_binary_spec(attribute, uint2_spec, pos)
    (color_mode, alpha_flag, num_channels, bit_depth_enum), pos = read_binary_spec(
        attribute, uint4_spec, pos
    )
    block_geom, pos = read_binary_spec(attribute, uint4_spec, pos)
    block_bytes = block_geom[0]
    subblocks_per_block = block_geom[2]
    block_dims, pos = read_binary_spec(attribute, uint4_spec, pos)
    block_width, _, block_height, block_stride = block_dims
    subblock_dims, pos = read_binary_spec(attribute, uint4_spec, pos)
    subblock_width, subblock_height = subblock_dims[0], subblock_dims[1]

    # --- InitColor section ---
    _boundary, pos = read_binary_spec(attribute, uint_spec, pos)
    if attribute[pos : pos + len(INITCOLOR_HEADER)] != INITCOLOR_HEADER:
        raise ValueError("Invalid attribute: missing InitColor header")
    pos += len(INITCOLOR_HEADER)

    (initcolor_magic,), pos = read_binary_spec(attribute, uint_spec, pos)
    init_color_quad, pos = read_binary_spec(attribute, uint4_spec, pos)
    has_init_color = bool(init_color_quad[0])
    init_color = init_color_quad[1]

    init_color_extra: Optional[Tuple[int, int, int, int]] = None
    init_color_extra_raw: Optional[Tuple[int, int, int, int]] = None
    if has_init_color:
        extra, pos = read_binary_spec(attribute, uint4_spec, pos)
        # 4× u32, each shifted right by 24 to extract the high byte = an
        # 8-bit RGBA channel.
        init_color_extra = tuple(min(255, v >> 24) for v in extra)
        init_color_extra_raw = extra

    # --- BlockSize section ---
    _boundary, pos = read_binary_spec(attribute, uint_spec, pos)
    if attribute[pos : pos + len(BLOCKSIZE_HEADER)] != BLOCKSIZE_HEADER:
        raise ValueError("Invalid attribute: missing BlockSize header")
    pos += len(BLOCKSIZE_HEADER)

    (blocksize_magic, nblocks, _nchan), pos = read_binary_spec(
        attribute, uint3_spec, pos
    )

    rest = attribute[pos:]
    expected_tail_bytes = 4 * nblocks
    if len(rest) != expected_tail_bytes:
        raise ValueError(
            f"BlockSize tail length mismatch: got {len(rest)}, expected {expected_tail_bytes}"
        )
    block_sizes = list(struct.unpack(f">{nblocks}I", rest))

    return OffscreenAttributes(
        section_sizes=section_sizes,
        width=width,
        height=height,
        cols=cols,
        rows=rows,
        color_mode=color_mode,
        alpha_flag=alpha_flag,
        num_channels=num_channels,
        bit_depth_enum=bit_depth_enum,
        block_bytes=block_bytes,
        subblocks_per_block=subblocks_per_block,
        block_width=block_width,
        block_height=block_height,
        block_stride=block_stride,
        subblock_width=subblock_width,
        subblock_height=subblock_height,
        initcolor_magic=initcolor_magic,
        has_init_color=has_init_color,
        init_color=init_color,
        init_color_extra=init_color_extra,
        init_color_extra_raw=init_color_extra_raw,
        blocksize_magic=blocksize_magic,
        block_sizes=block_sizes,
    )
