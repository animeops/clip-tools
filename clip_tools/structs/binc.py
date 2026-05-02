"""Parser for the Celsys MoTion "binc" typed-serialization format.

Used by Track.TrackActionMixer{,2} (Action Mixer keyframe animation),
TimeLapseBlob.BlobData, Canvas3DModelBank.BankData, and other
non-raster external chunks. The acronym is "CMT" = Celsys MoTion;
the format also carries skeletal animation, FCurves, and 3D pose data
in CSP's Odyssey 3D module.

Wire format (bytes):

    "cmt " + ascii_version (4) + "binc"   # 12 bytes magic
    u32 LE body_crc32                       # CRC32 of all bytes after this field
    u32 LE num_strings                      # combined type + identifier table
    [string_table]:
        u8  length
        bytes name                          # UTF-8
    [root_node]                             # see parse_node

Two versions seen so far: "0100" and "0110". The only difference:
0110 prepends a 12-byte forward-jump table to every node containing
the byte offsets of the type/num_attrs/num_children fields relative to
the prefix start. We skip the prefix; the actual record layout is the
same in both versions.

Node layout:

    u32 name_idx          # into string table
    u32 type_idx          # into string table; type_idx == 0 ("null") = container
    [value]               # only if type != "null"; encoding depends on type
    u32 num_attrs
    num_attrs × (u32 attr_name_idx, u32 attr_value_string_idx)
    u32 num_children
    num_children × node   # recursive

Attr values are always treated as string-table indices (no per-attr type),
which matches every observed sample. String index 0xFFFFFFFF is a sentinel
for the empty string.
"""

import struct
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

EMPTY_STRING_SENTINEL = 0xFFFFFFFF


@dataclass
class BincNode:
    name: str
    type: str
    value: Any = None
    attrs: Dict[str, str] = field(default_factory=dict)
    children: List["BincNode"] = field(default_factory=list)


@dataclass
class BincDocument:
    version: str  # "0100" or "0110"
    body_crc32: int  # CRC32 of all bytes after the magic+crc header
    strings: List[str]  # combined type + identifier table
    root: BincNode


def is_binc(buf: bytes) -> bool:
    return len(buf) >= 12 and buf[:4] == b"cmt " and buf[8:12] == b"binc"


def parse_binc(buf: bytes) -> BincDocument:
    if not is_binc(buf):
        raise ValueError("Not a binc blob (missing 'cmt ...binc' magic)")

    version = buf[4:8].decode("ascii")
    stored_crc = struct.unpack_from("<I", buf, 12)[0]
    actual_crc = zlib.crc32(buf[16:]) & 0xFFFFFFFF
    if stored_crc != actual_crc:
        raise ValueError(
            f"binc CRC32 mismatch: header says {stored_crc:08x}, "
            f"body computes {actual_crc:08x}"
        )
    num_strings = struct.unpack_from("<I", buf, 16)[0]

    strings: List[str] = []
    pos = 20
    for _ in range(num_strings):
        slen = buf[pos]
        pos += 1
        strings.append(buf[pos : pos + slen].decode("utf-8", errors="replace"))
        pos += slen

    has_offset_prefix = version == "0110"

    def lookup_string(idx: int) -> str:
        if idx == EMPTY_STRING_SENTINEL:
            return ""
        return strings[idx]

    def read_value(b: bytes, p: int, type_name: str) -> Tuple[Any, int]:
        if type_name == "null":
            return None, p
        if type_name == "Byte":
            return b[p], p + 1
        if type_name == "SByte":
            return struct.unpack_from("<b", b, p)[0], p + 1
        if type_name == "UInt16":
            return struct.unpack_from("<H", b, p)[0], p + 2
        if type_name == "Int16":
            return struct.unpack_from("<h", b, p)[0], p + 2
        if type_name == "UInt32":
            return struct.unpack_from("<I", b, p)[0], p + 4
        if type_name == "Int32":
            return struct.unpack_from("<i", b, p)[0], p + 4
        if type_name == "Single":
            return struct.unpack_from("<f", b, p)[0], p + 4
        if type_name == "Double":
            return struct.unpack_from("<d", b, p)[0], p + 8
        if type_name == "String":
            return lookup_string(struct.unpack_from("<I", b, p)[0]), p + 4
        if type_name == "Float2":
            return struct.unpack_from("<2f", b, p), p + 8
        if type_name == "Float3":
            return struct.unpack_from("<3f", b, p), p + 12
        if type_name == "Double2":
            return struct.unpack_from("<2d", b, p), p + 16
        if type_name == "Double3":
            return struct.unpack_from("<3d", b, p), p + 24
        if type_name == "Quat":
            return struct.unpack_from("<4f", b, p), p + 16
        if type_name == "Matrix44":
            return struct.unpack_from("<16f", b, p), p + 64
        if type_name == "Byte[]":
            n = struct.unpack_from("<I", b, p)[0]
            return bytes(b[p + 4 : p + 4 + n]), p + 4 + n
        if type_name == "String[]":
            n = struct.unpack_from("<I", b, p)[0]
            arr = [
                lookup_string(struct.unpack_from("<I", b, p + 4 + 4 * i)[0])
                for i in range(n)
            ]
            return arr, p + 4 + 4 * n

        array_specs = {
            "Single[]": ("f", 1),
            "Int32[]": ("i", 1),
            "Double[]": ("d", 1),
            "Double2[]": ("d", 2),
            "Double3[]": ("d", 3),
            "Float2[]": ("f", 2),
            "Float3[]": ("f", 3),
            "Quat[]": ("f", 4),
            "Matrix44[]": ("f", 16),
        }
        if type_name in array_specs:
            n = struct.unpack_from("<I", b, p)[0]
            ch, mul = array_specs[type_name]
            total = n * mul
            byte_size = total * struct.calcsize(ch)
            arr = list(struct.unpack_from(f"<{total}{ch}", b, p + 4))
            return arr, p + 4 + byte_size

        raise ValueError(f"Unhandled binc type: {type_name!r}")

    def parse_node(b: bytes, p: int) -> Tuple[BincNode, int]:
        if has_offset_prefix:
            # Forward-jump table: type_off, num_attrs_off, num_children_off
            # (byte offsets within this record). Used for fast skip-ahead;
            # we don't need it because we walk the body anyway.
            p += 12
        name_idx = struct.unpack_from("<I", b, p)[0]
        p += 4
        type_idx = struct.unpack_from("<I", b, p)[0]
        p += 4
        name = lookup_string(name_idx)
        type_name = lookup_string(type_idx)
        value, p = read_value(b, p, type_name)

        num_attrs = struct.unpack_from("<I", b, p)[0]
        p += 4
        attrs: Dict[str, str] = {}
        for _ in range(num_attrs):
            an = struct.unpack_from("<I", b, p)[0]
            p += 4
            av = struct.unpack_from("<I", b, p)[0]
            p += 4
            attrs[lookup_string(an)] = lookup_string(av)

        num_children = struct.unpack_from("<I", b, p)[0]
        p += 4
        children: List[BincNode] = []
        for _ in range(num_children):
            child, p = parse_node(b, p)
            children.append(child)

        return BincNode(name, type_name, value, attrs, children), p

    root, _ = parse_node(buf, pos)
    return BincDocument(version, stored_crc, strings, root)


def find_child(node: BincNode, name: str) -> Optional[BincNode]:
    for c in node.children:
        if c.name == name:
            return c
    return None
