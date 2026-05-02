"""Parsers for small SQLite-cell binary blobs.

Each parser is pure: bytes in, structured data out. Larger / more complex
blobs (Offscreen.Attribute, TextLayerAttributes, vector data, binc) live in
their own modules.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import List, Optional, Tuple


def format_uuid(b: Optional[bytes]) -> Optional[str]:
    """Standard 8-4-4-4-12 hex formatting for a 16-byte UUID."""
    if b is None or len(b) != 16:
        return None
    h = b.hex()
    return f"{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


@dataclass
class SmallObjectFlag:
    """Decoded 12B blob: `(size=12, magic=0, value)` triplet from
    `SmallObjectInfo.SmallObject*` columns."""

    size: int
    magic: int
    value: int


def parse_vanish_point_guide(
    blob: bytes, guide_number: int
) -> List[Tuple[float, float, float, float]]:
    """Decode `RulerVanishPoint.Guide`: `guide_number` x (x0,y0,x1,y1) f64 BE.

    Each guide is a line segment given by two endpoints in canvas coords.
    The blob is `guide_number * 32` bytes; we skip parsing if the size
    doesn't match (returns an empty list).
    """
    if guide_number <= 0:
        return []
    expected = guide_number * 32
    if len(blob) < expected:
        return []
    out: List[Tuple[float, float, float, float]] = []
    for i in range(guide_number):
        off = i * 32
        x0, y0, x1, y1 = struct.unpack(">4d", blob[off : off + 32])
        out.append((x0, y0, x1, y1))
    return out


def parse_effector_control_points(
    blob: bytes, control_number: int
) -> List[Tuple[float, float]]:
    """Decode `BrushEffectorGraphData.ControlPoints`: `control_number` x
    (x, y) f64 BE pairs, monotonically rising x in [0, 1] (curve domain)
    with y in [0, 1] (curve range)."""
    if control_number <= 0:
        return []
    expected = control_number * 16
    if len(blob) < expected:
        return []
    return [
        struct.unpack(">2d", blob[i * 16 : i * 16 + 16]) for i in range(control_number)
    ]


def parse_small_object_flag(blob: Optional[bytes]) -> Optional[SmallObjectFlag]:
    """Decode `SmallObjectInfo.SmallObject*` 12-byte flag triplets:
    `(size_u32=12, magic_u32=0, value_u32)`. Returns a `SmallObjectFlag`."""
    if blob is None or len(blob) != 12:
        return None
    size, magic, value = struct.unpack(">3I", blob)
    return SmallObjectFlag(size=size, magic=magic, value=value)


# ---------------------------------------------------------------------------
# BrushStyle.*Effector — pressure / dynamics curves
# ---------------------------------------------------------------------------


@dataclass
class BrushEffector:
    """A `BrushStyle.*Effector` blob (`SizeEffector`, `OpacityEffector`, ...).

    Layout (observed across `wn_04_009_LOSA.clip` brushes):

    - `effector_id : u32`  flags / type. `0` (alone) means "no curve".
    - `unknown_u32 : u32`  always 0 in samples.
    - `param_count : u32`  number of `(input_index, value)` pairs that follow.
    - For each param: `(input_index : u32, value)` where `value` may be a
      ref into `BrushEffectorGraphData.MainId` (when stored as u32) or a
      f32 scalar — the discriminator is encoded inside `effector_id` and
      not yet fully unpicked.

    For now the dataclass holds the decoded header + the raw param tail.
    Renderers needing the curve resolve `param_count` and the byte tail
    against `BrushEffectorGraphData` themselves.
    """

    effector_id: int
    unknown_u32: int
    param_count: int
    param_tail: bytes
    raw: bytes

    @property
    def is_no_op(self) -> bool:
        return self.effector_id == 0 and len(self.raw) <= 4


def parse_brush_effector(blob: Optional[bytes]) -> Optional[BrushEffector]:
    if blob is None:
        return None
    if len(blob) < 4:
        return None
    if len(blob) == 4:
        # Stub: just an effector_id.
        (eid,) = struct.unpack(">I", blob)
        return BrushEffector(
            effector_id=eid,
            unknown_u32=0,
            param_count=0,
            param_tail=b"",
            raw=bytes(blob),
        )
    if len(blob) < 12:
        # Truncated header; treat conservatively as a stub.
        return BrushEffector(
            effector_id=struct.unpack(">I", blob[:4])[0],
            unknown_u32=0,
            param_count=0,
            param_tail=bytes(blob[4:]),
            raw=bytes(blob),
        )
    eid, unk, pcount = struct.unpack(">3I", blob[:12])
    return BrushEffector(
        effector_id=eid,
        unknown_u32=unk,
        param_count=pcount,
        param_tail=bytes(blob[12:]),
        raw=bytes(blob),
    )


# ---------------------------------------------------------------------------
# Track.TrackValueMap — typed property map keyed by UTF-16BE strings
# ---------------------------------------------------------------------------


@dataclass
class TrackValueEntry:
    """One entry of a `TrackValueMap` typed property dict."""

    key: str  # UTF-16BE-decoded name
    value_type: int  # 1=u32, 2=f64, 3=string, ... (observed; not exhaustive)
    raw_value: bytes  # bytes of the value field (post-key)


@dataclass
class TrackValueMap:
    magic: int  # observed always 8
    entries: List[TrackValueEntry]


def parse_track_value_map(blob: Optional[bytes]) -> Optional[TrackValueMap]:
    """Skeleton parser for `Track.TrackValueMap`.

    Format (TLV map):
      `magic : u32` (=8)
      `count : u32`
      For each of `count` entries:
        `entry_size : u32` (length of the rest of this entry, in bytes)
        `key_chars : u32`
        UTF-16BE key (`key_chars * 2` bytes)
        `value_type : u32`
        `raw_value : entry_size - 4 - 4 - key_chars*2 - 4` bytes

    The exact `value_type` codes (1, 2, 3, …) aren't fully pinned down —
    the dataclass keeps the value bytes raw so consumers can decode per
    type once those are known.
    """
    if blob is None or len(blob) < 8:
        return None
    magic, count = struct.unpack(">2I", blob[:8])
    pos = 8
    entries: List[TrackValueEntry] = []
    for _ in range(count):
        if pos + 8 > len(blob):
            break
        entry_size, key_chars = struct.unpack(">2I", blob[pos : pos + 8])
        pos += 8
        key_bytes = key_chars * 2
        if pos + key_bytes + 4 > len(blob):
            break
        key = blob[pos : pos + key_bytes].decode("utf-16-be", errors="replace")
        pos += key_bytes
        (value_type,) = struct.unpack(">I", blob[pos : pos + 4])
        pos += 4
        # Remaining bytes inside this entry: entry_size accounts for the
        # whole entry minus its own size prefix (= entry_size - 4 already
        # consumed before the key_chars field). Bound conservatively.
        consumed_in_entry = 4 + key_bytes + 4  # key_chars + key + value_type
        value_len = max(0, entry_size - consumed_in_entry)
        if pos + value_len > len(blob):
            value_len = len(blob) - pos
        raw_value = bytes(blob[pos : pos + value_len])
        pos += value_len
        entries.append(
            TrackValueEntry(key=key, value_type=value_type, raw_value=raw_value)
        )
    return TrackValueMap(magic=magic, entries=entries)


# ---------------------------------------------------------------------------
# LayerComp.CompLayerInfo — saved per-layer state snapshots
# ---------------------------------------------------------------------------


@dataclass
class CompLayerInfoEntry:
    layer_uuid: Optional[str]
    raw_payload: bytes  # tail bytes within this entry after the UUID


@dataclass
class CompLayerInfo:
    magic: int  # observed always 8
    entry_count: int
    entries: List[CompLayerInfoEntry]


def parse_comp_layer_info(blob: Optional[bytes]) -> Optional[CompLayerInfo]:
    """Skeleton parser for `LayerComp.CompLayerInfo`.

    Format (observed on the one wn sample):
      `magic : u32` (=8)
      `entry_count : u32`
      For each entry:
        `entry_size : u32` (=32 in the sample)
        `entry_type : u32` (=16, = a 16-byte UUID payload follows)
        `layer_uuid : 16 bytes`
        `tail : entry_size - 16 - 8` bytes (8B payload in the sample)

    Variable per-entry payload shapes are passed through as `raw_payload`.
    """
    if blob is None or len(blob) < 8:
        return None
    magic, count = struct.unpack(">2I", blob[:8])
    pos = 8
    entries: List[CompLayerInfoEntry] = []
    for _ in range(count):
        if pos + 8 > len(blob):
            break
        entry_size, entry_type = struct.unpack(">2I", blob[pos : pos + 8])
        pos += 8
        # entry_size covers everything after itself (entry_type + payload).
        payload_len = max(0, entry_size - 4)
        if pos + payload_len > len(blob):
            break
        body = blob[pos : pos + payload_len]
        pos += payload_len
        if entry_type == 16 and len(body) >= 16:
            uuid = format_uuid(bytes(body[:16]))
            tail = bytes(body[16:])
        else:
            uuid = None
            tail = bytes(body)
        entries.append(CompLayerInfoEntry(layer_uuid=uuid, raw_payload=tail))
    return CompLayerInfo(magic=magic, entry_count=count, entries=entries)
