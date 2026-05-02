import struct
from typing import Any, Dict, List

from clip_tools.constants import (
    DEBUG,
    TextStylingType,
    TextWindowType,
)
from clip_tools.utils import read_binary_spec


def parse_font_style_block(blob: bytes, pos: int) -> Dict[str, Any]:
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


def parse_chunk_block(blob: bytes, pos: int, is_last: bool) -> Dict[str, Any]:
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


# TLV tag IDs.
TLV_PARAGRAPH_ALIGN = 12  # paragraph-run array (start, length, align byte)
TLV_PARAGRAPH_UNK_18 = 18  # same shape as 12/16/20; specific property unknown
TLV_PARAGRAPH_UNDERLINE = 16
TLV_PARAGRAPH_STRIKE = 20
TLV_ASPECT_RATIO = 26  # f64 at offset 16 of the value
TLV_DEFAULT_FONT = 31  # UTF-8 font name
TLV_FONT_SIZE = 32  # u32 in 1/100 pt
TLV_UNIT = 33  # u32; expected 1, observed 0 in our samples
TLV_OUTLINE_COLOR = 34  # 3× i32 normalized to [0, 1]
TLV_TEXT_BBOX = 42  # 4× i32 (x0, y0, x1, y1)
TLV_SECONDARY_FONT = 47  # font reference with i16 prefix + (50, 0) markers
TLV_FONT_ALIASES = 57  # i16 N + N × (display, postscript) UTF-8 pairs + i32
TLV_SKEW_ANGLE_1 = 59
TLV_SKEW_ANGLE_2 = 60
TLV_BOX_SIZE = 63  # 2× i32
TLV_QUAD_VERTS = 64  # 8× i32, divided by 100 → 4 (x, y) corners


def parse_tlv_records(blob: bytes, start: int, end: int) -> List[Dict[str, Any]]:
    """Walk a tag-length-value stream. Each record is `u32 tag + u32 length
    + length bytes value`. Returns a list of {"tag", "value"} dicts."""
    records: List[Dict[str, Any]] = []
    uint_le = struct.Struct("<I")
    pos = start
    while pos + 8 <= end:
        tag = uint_le.unpack_from(blob, pos)[0]
        pos += 4
        length = uint_le.unpack_from(blob, pos)[0]
        pos += 4
        if pos + length > end:
            break
        records.append({"tag": tag, "value": blob[pos : pos + length]})
        pos += length
    return records


def parse_paragraph_runs(val: bytes) -> List[Dict[str, Any]]:
    """Decode a paragraph-runs TLV value (tags 12, 16, 18, 20).

    Layout::

        i32      n
        per run (14 bytes):
            i32  start
            i32  length
            i32  unk1
            i8   value     # the actual property (alignment for tag 12, etc.)
            i8   unk2

    A zero-length value (no leading count) means "no runs / feature off" —
    seen on tags 16 and 20 in our current samples.
    """
    if len(val) < 4:
        return []
    n = struct.unpack_from("<i", val, 0)[0]
    out = []
    pos = 4
    for _ in range(n):
        if pos + 14 > len(val):
            break
        start = struct.unpack_from("<i", val, pos)[0]
        length = struct.unpack_from("<i", val, pos + 4)[0]
        unk1 = struct.unpack_from("<i", val, pos + 8)[0]
        value = struct.unpack_from("<b", val, pos + 12)[0]
        unk2 = struct.unpack_from("<b", val, pos + 13)[0]
        out.append(
            {
                "start": start,
                "length": length,
                "unk1": unk1,
                "value": value,
                "unk2": unk2,
            }
        )
        pos += 14
    return out


def parse_font_aliases(val: bytes) -> Dict[str, Any]:
    """Decode TLV tag 57: a list of (display_name, postscript_name) pairs.

    Layout::

        i16              n (number of font pairs)
        per pair:
            i16          display_name_length (UTF-8 bytes)
            byte[*]      display_name        (UTF-8)
            i16          ps_name_length      (UTF-8 bytes)
            byte[*]      postscript_name     (UTF-8)
        i32              trailer (observed 0x00000708 = 1800)
    """
    if len(val) < 2:
        return {"font_list": [], "trailer": 0}
    n = struct.unpack_from("<h", val, 0)[0]
    pos = 2
    font_list = []
    for _ in range(n):
        if pos + 2 > len(val):
            break
        n1 = struct.unpack_from("<h", val, pos)[0]
        pos += 2
        display = val[pos : pos + n1].decode("utf-8", errors="replace")
        pos += n1
        if pos + 2 > len(val):
            break
        n2 = struct.unpack_from("<h", val, pos)[0]
        pos += 2
        ps = val[pos : pos + n2].decode("utf-8", errors="replace")
        pos += n2
        font_list.append({"display_name": display, "postscript_name": ps})
    trailer = struct.unpack_from("<i", val, pos)[0] if pos + 4 <= len(val) else None
    return {"font_list": font_list, "trailer": trailer}


def parse_secondary_font(val: bytes) -> Dict[str, Any]:
    """Decode TLV tag 47: a secondary font reference with a structured prefix.

    Layout::

        i16              flag      (varies; 0 when empty, non-zero otherwise)
        i32              marker_a  (always 50)
        i32              marker_b  (always 0)
        i16              name_length
        byte[*]          font_name (UTF-8)
    """
    if len(val) < 12:
        return {"empty": True, "raw": val}
    flag = struct.unpack_from("<h", val, 0)[0]
    marker_a = struct.unpack_from("<i", val, 2)[0]
    marker_b = struct.unpack_from("<i", val, 6)[0]
    name_len = struct.unpack_from("<h", val, 10)[0]
    name = val[12 : 12 + name_len].decode("utf-8", errors="replace") if name_len else ""
    out = {
        "flag": flag,
        "marker_a": marker_a,
        "marker_b": marker_b,
        "font_name": name,
    }
    if marker_a != 50 or marker_b != 0:
        out["unexpected_markers"] = True
    return out


def process_text_attributes(attributes: bytes) -> Dict[str, Any]:
    """Decode the per-layer ``TextLayerAttributes`` blob.

    The blob is style metadata only; the actual text content lives in the
    sibling ``TextLayerString`` column (UTF-8).

    Top-level layout::

        u32 header (=11)
        u32 font_styles_section_length     # bytes from after num_font_styles
        u32 num_font_styles
        u32 mystery_a                      # observed values 0/3/16
        FontStyleBlock × num_font_styles

        u32 chunks_section_length          # bytes from after this field
        u32 num_chunks
        u32 mystery_a (mirror)
        ChunkBlock × num_chunks

        TLV stream (until end of blob):
            u32 tag, u32 length, byte[length] value

    Known TLV tags:

    - 12  paragraph_align             (paragraph runs)
    - 16  paragraph_underline         (paragraph runs)
    - 18  paragraph_unk_18            (paragraph runs; specific property unknown)
    - 20  paragraph_strike            (paragraph runs)
    - 26  aspect_ratio                (f64 at offset 16 of the value)
    - 31  default_font_name           (UTF-8)
    - 32  font_size                   (i32 in 1/100 pt)
    - 33  unit                        (i32; expected 1)
    - 34  outline_color               (3× i32 normalized to [0, 1])
    - 42  text_bbox                   (4× i32: x0, y0, x1, y1) — origin doubles
                                       as the layer's general offset
    - 47  secondary_font              (font reference with prefix flags)
    - 57  font_aliases                (list of (display, PostScript) pairs)
    - 59  skew_angle_1
    - 60  skew_angle_2
    - 63  box_size                    (2× i32)
    - 64  quad_verts                  (8× i32 / 100 → 4 (x, y) corners)

    Other tags are exposed as raw bytes via ``tlv_records``; see
    ``unknowns.md`` for the still-unidentified set.
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
        fs, pos = parse_font_style_block(attributes, pos)
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
        ck, pos = parse_chunk_block(
            attributes, pos, is_last=(i == out["num_chunks"] - 1)
        )
        out["chunks"].append(ck)

    if pos != chunks_section_end:
        out["_chunks_section_unconsumed"] = attributes[pos:chunks_section_end]
        pos = chunks_section_end

    # Everything after the chunks section is a flat TLV stream.
    tlv_records = parse_tlv_records(attributes, pos, len(attributes))
    out["tlv_records"] = tlv_records

    # Surface known tags as named fields for ergonomic access.
    by_tag: Dict[int, bytes] = {r["tag"]: r["value"] for r in tlv_records}

    if TLV_DEFAULT_FONT in by_tag:
        try:
            out["default_font_name"] = by_tag[TLV_DEFAULT_FONT].decode("utf-8")
        except UnicodeDecodeError:
            out["default_font_name"] = ""
    if TLV_FONT_SIZE in by_tag and len(by_tag[TLV_FONT_SIZE]) == 4:
        out["font_size"] = struct.unpack_from("<i", by_tag[TLV_FONT_SIZE], 0)[0]
    if TLV_UNIT in by_tag and len(by_tag[TLV_UNIT]) == 4:
        out["unit"] = struct.unpack_from("<i", by_tag[TLV_UNIT], 0)[0]
    if TLV_OUTLINE_COLOR in by_tag and len(by_tag[TLV_OUTLINE_COLOR]) == 12:
        chans = struct.unpack_from("<iii", by_tag[TLV_OUTLINE_COLOR], 0)
        out["outline_color"] = tuple(c / (2**32 - 1) for c in chans)
    if TLV_TEXT_BBOX in by_tag and len(by_tag[TLV_TEXT_BBOX]) == 16:
        out["text_bbox"] = struct.unpack_from("<iiii", by_tag[TLV_TEXT_BBOX], 0)
        out["general_offset_x"] = out["text_bbox"][0]
        out["general_offset_y"] = out["text_bbox"][1]
    if TLV_ASPECT_RATIO in by_tag and len(by_tag[TLV_ASPECT_RATIO]) >= 32:
        out["aspect_ratio"] = struct.unpack_from("<d", by_tag[TLV_ASPECT_RATIO], 16)[0]
    if TLV_SECONDARY_FONT in by_tag:
        out["secondary_font"] = parse_secondary_font(by_tag[TLV_SECONDARY_FONT])
    if TLV_FONT_ALIASES in by_tag:
        out["font_aliases"] = parse_font_aliases(by_tag[TLV_FONT_ALIASES])
    if TLV_SKEW_ANGLE_1 in by_tag and len(by_tag[TLV_SKEW_ANGLE_1]) == 4:
        out["skew_angle_1"] = struct.unpack_from("<i", by_tag[TLV_SKEW_ANGLE_1], 0)[0]
    if TLV_SKEW_ANGLE_2 in by_tag and len(by_tag[TLV_SKEW_ANGLE_2]) == 4:
        out["skew_angle_2"] = struct.unpack_from("<i", by_tag[TLV_SKEW_ANGLE_2], 0)[0]
    if TLV_BOX_SIZE in by_tag and len(by_tag[TLV_BOX_SIZE]) == 8:
        out["box_size"] = struct.unpack_from("<ii", by_tag[TLV_BOX_SIZE], 0)
    if TLV_QUAD_VERTS in by_tag and len(by_tag[TLV_QUAD_VERTS]) == 32:
        verts = struct.unpack_from("<8i", by_tag[TLV_QUAD_VERTS], 0)
        out["quad_verts"] = tuple(v / 100 for v in verts)
    for tag, key in (
        (TLV_PARAGRAPH_ALIGN, "paragraph_align"),
        (TLV_PARAGRAPH_UNDERLINE, "paragraph_underline"),
        (TLV_PARAGRAPH_STRIKE, "paragraph_strike"),
        (TLV_PARAGRAPH_UNK_18, "paragraph_unk_18"),
    ):
        if tag in by_tag:
            out[key] = parse_paragraph_runs(by_tag[tag])

    out["_undecoded_tail"] = b""
    out["_undecoded_tail_offset"] = len(attributes)

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
