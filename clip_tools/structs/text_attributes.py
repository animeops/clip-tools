import struct
from typing import Any, Dict, List

from clip_tools.constants import (
    DEBUG,
    TextStylingType,
    TextWindowType,
)
from clip_tools.utils import read_binary_spec


def _parse_font_style_block(blob: bytes, pos: int) -> Dict[str, Any]:
    """Parse one font-style record.

    Layout::

        u32  num_chars_in_style   # how many characters of TextLayerString this style spans
        u32  num_bytes            # body is num_bytes - 6 bytes (after the 6 fields below)
        u8   text_styling_type
        u8   text_window_type
        body (num_bytes - 6 bytes):
            u16 [LE] color_r   # full 16-bit value; 8-bit channel = (v >> 8) & 0xFF
            u16 [LE] color_g
            u16 [LE] color_b
            byte[6]  color_padding   # observed all zero (alpha + reserved?)
            u16 [LE] style_flag      # 0x4059 if any styling present, 0 otherwise
            u16 [LE] font_name_length    # in UTF-16 LE chars
            byte[font_name_length*2] font_name (UTF-16 LE)
            u32 [LE] tail_field      # unknown — not the same as num_bytes
    """
    uint_le = struct.Struct("<I")
    u16_le = struct.Struct("<HHH")
    byte2 = struct.Struct("<BB")
    u16_pair_le = struct.Struct("<HH")

    out: Dict[str, Any] = {}

    out["num_chars_in_style"] = uint_le.unpack_from(blob, pos)[0]
    pos += 4
    num_bytes = uint_le.unpack_from(blob, pos)[0]
    pos += 4
    out["num_bytes"] = num_bytes
    styling, window = byte2.unpack_from(blob, pos)
    pos += 2
    try:
        out["styling"] = TextStylingType(styling)
    except ValueError:
        out["styling"] = styling
    try:
        out["window"] = TextWindowType(window)
    except ValueError:
        out["window"] = window

    body_end = pos + (num_bytes - 6)

    r, g, b = u16_le.unpack_from(blob, pos)
    pos += 6
    out["color"] = ((r >> 8) & 0xFF, (g >> 8) & 0xFF, (b >> 8) & 0xFF)
    out["_color_padding"] = blob[pos : pos + 6]
    pos += 6

    style_flag, name_len = u16_pair_le.unpack_from(blob, pos)
    pos += 4
    out["style_flag"] = style_flag
    out["font_name_length"] = name_len

    if name_len > 0:
        out["font_name"] = blob[pos : pos + name_len * 2].decode("utf-16-le")
        pos += name_len * 2
    else:
        out["font_name"] = ""

    out["tail_field"] = uint_le.unpack_from(blob, pos)[0]
    pos += 4

    if pos != body_end:
        out["_unconsumed"] = blob[pos:body_end]
        pos = body_end

    return out, pos


def _parse_chunk_block(blob: bytes, pos: int, is_last: bool) -> Dict[str, Any]:
    """Parse one chunk record.

    A "chunk" mirrors a font-style range — there is one chunk per font_style
    in the same order, covering the same character span.

    Layout (28 bytes for non-last chunks, 24 for the last)::

        u32  num_chars_in_chunk  # equals the matching font_style.num_chars_in_style
        u32  marker              # observed always 16 — a record-type code?
        f64  text_box_scale_x    # percent
        f64  text_box_scale_y    # percent
        u32  separator           # only on non-last chunks; observed always 2
    """
    uint_le = struct.Struct("<I")
    d2 = struct.Struct("<dd")
    out: Dict[str, Any] = {}
    out["num_chars_in_chunk"] = uint_le.unpack_from(blob, pos)[0]
    pos += 4
    out["marker"] = uint_le.unpack_from(blob, pos)[0]
    pos += 4
    sx, sy = d2.unpack_from(blob, pos)
    pos += 16
    out["text_box_scale_x"] = sx
    out["text_box_scale_y"] = sy
    if not is_last:
        out["separator"] = uint_le.unpack_from(blob, pos)[0]
        pos += 4
    return out, pos


def process_text_attributes(attributes: bytes) -> Dict[str, Any]:
    """Decode the per-layer ``TextLayerAttributes`` blob.

    The blob is style metadata only; the actual text content lives in the
    sibling ``TextLayerString`` column (UTF-8).

    Layout (the section past the chunks block is still under investigation —
    see ``unknowns.md``)::

        u32 header (=11)
        u32 font_styles_section_length     # bytes from the field below to
                                           # the end of the last per-style block
        u32 num_font_styles
        u32 mystery_a                      # observed values 0/3/16; mirrors
                                           # the same field after the chunks
                                           # block
        FontStyleBlock × num_font_styles
        u32 chunks_section_length          # bytes of the next 8 bytes + per-chunk records
        u32 num_chunks
        u32 mystery_a (mirror)
        ChunkBlock × num_chunks
        ... (rest currently undecoded)
    """
    uint_le = struct.Struct("<I")

    out: Dict[str, Any] = {}
    pos = 0

    header = uint_le.unpack_from(attributes, pos)[0]
    pos += 4
    if header != 11:
        raise ValueError(f"Invalid text attributes header: expected 11, got {header}")
    out["header"] = header

    fs_section_len = uint_le.unpack_from(attributes, pos)[0]
    pos += 4
    out["font_styles_section_length"] = fs_section_len

    # The font-styles length field measures bytes starting from *after*
    # num_font_styles (mystery_a + the per-style records). Read
    # num_font_styles first so we can anchor the section end correctly.
    out["num_font_styles"] = uint_le.unpack_from(attributes, pos)[0]
    pos += 4
    fs_section_end = pos + fs_section_len
    out["mystery_a"] = uint_le.unpack_from(attributes, pos)[0]
    pos += 4

    out["font_styles"] = []
    for _ in range(out["num_font_styles"]):
        fs, pos = _parse_font_style_block(attributes, pos)
        out["font_styles"].append(fs)

    if pos != fs_section_end:
        out["_font_styles_section_unconsumed"] = attributes[pos:fs_section_end]
        pos = fs_section_end

    chunks_section_len = uint_le.unpack_from(attributes, pos)[0]
    pos += 4
    out["chunks_section_length"] = chunks_section_len
    chunks_section_end = pos + chunks_section_len

    out["num_chunks"] = uint_le.unpack_from(attributes, pos)[0]
    pos += 4
    out["mystery_a_mirror"] = uint_le.unpack_from(attributes, pos)[0]
    pos += 4

    out["chunks"] = []
    for i in range(out["num_chunks"]):
        ck, pos = _parse_chunk_block(
            attributes, pos, is_last=(i == out["num_chunks"] - 1)
        )
        out["chunks"].append(ck)

    if pos != chunks_section_end:
        out["_chunks_section_unconsumed"] = attributes[pos:chunks_section_end]
        pos = chunks_section_end

    out["_undecoded_tail"] = attributes[pos:]
    out["_undecoded_tail_offset"] = pos

    if DEBUG:
        for k, v in out.items():
            if k.startswith("_"):
                continue
            if isinstance(v, list):
                print(f"{k}:")
                for i, item in enumerate(v):
                    print(f"  [{i}] {item}")
            else:
                print(f"{k}: {v}")
        if out["_undecoded_tail"]:
            print(
                f"undecoded tail: {len(out['_undecoded_tail'])} bytes from offset "
                f"{out['_undecoded_tail_offset']}"
            )

    return out
