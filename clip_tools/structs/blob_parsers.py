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
# BrushStyle.*Effector — per-stamp parameter modulator blobs
# ---------------------------------------------------------------------------
#
# Layout: 4-byte flags + variable-size modulator blocks in flag-bit order.
# Each set bit pulls in its modulator block at a fixed size:
#
#   Bit 0x10  Pressure : f32 min + u32 curve_id      (8B)
#   Bit 0x20  Velocity : f32 min + u32 curve_id + f32 max  (12B)
#   Bit 0x40  Smooth   : f32 min                     (4B)
#   Bit 0x80  Random   : f32 min                     (4B)
#   Bit 0x100 Tilt     : f32 min                     (4B)
#   Bit 0x01  master "effector enabled" flag (always set in observed blobs)
#
# `curve_id` is a foreign key into `BrushEffectorGraphData.MainId` whose
# `control_points` define the input→output curve for that modulator. NULL /
# missing curve falls back to identity linear `[(0,0), (1,1)]`.


EFFECTOR_FLAG_ENABLED = 0x01
EFFECTOR_FLAG_PRESSURE = 0x10
EFFECTOR_FLAG_VELOCITY = 0x20
EFFECTOR_FLAG_SMOOTH = 0x40
EFFECTOR_FLAG_RANDOM = 0x80
EFFECTOR_FLAG_TILT = 0x100


@dataclass
class EffectorPressure:
    min: float  # output multiplier at pressure=0  (curve_y * (1-min) + min)
    curve_id: int  # FK into BrushEffectorGraphData.MainId; 0 = identity


@dataclass
class EffectorVelocity:
    min: float
    curve_id: int
    # `max` lives after the trailing version field in the blob, gated by a
    # version check inside `LoadEffector`. Defaults to 1.0 when the version
    # check fails or the blob is too short to include it.
    max: float = 1.0


@dataclass
class EffectorSmooth:
    min: float


@dataclass
class EffectorRandom:
    min: float


@dataclass
class EffectorTilt:
    min: float


@dataclass
class BrushEffector:
    """A decoded `BrushStyle.*Effector` blob.

    Each modulator is `None` if its flag bit is unset. Render-time evaluation
    chains the present modulators in flag-bit order; missing ones contribute
    an identity multiplier (1.0).
    """

    flags: int  # raw flag bits (per-modulator + low version/init bits)
    pressure: Optional[EffectorPressure] = None
    velocity: Optional[EffectorVelocity] = None
    smooth: Optional[EffectorSmooth] = None
    random: Optional[EffectorRandom] = None
    tilt: Optional[EffectorTilt] = None
    version: int = 0  # trailing `version_or_size` field; gates velocity_max
    raw: bytes = b""

    @property
    def is_no_op(self) -> bool:
        """True if the effector has no enabled modulators (just a stub)."""
        return (
            self.pressure is None
            and self.velocity is None
            and self.smooth is None
            and self.random is None
            and self.tilt is None
        )


def parse_brush_effector(blob: Optional[bytes]) -> Optional[BrushEffector]:
    """Decode a ``BrushStyle.*Effector`` blob to a ``BrushEffector`` dataclass.

    Layout::

        flags                  : BE u32
        if flags & 0x10:       # Pressure
            pressure_min       : BE f32
            pressure_curve_id  : BE u32
        if flags & 0x20:       # Velocity
            velocity_min       : BE f32
            velocity_curve_id  : BE u32
        if flags & 0x40:       # Smooth
            smooth_min         : BE f32
        if flags & 0x80:       # Random
            random_min         : BE f32
        if flags & 0x100:      # Tilt
            tilt_min           : BE f32
        version_or_size        : BE u32 (optional, present in observed samples)
        if (version_check) and (flags & 0x20):
            velocity_max       : BE f32 (else defaults to 1.0)

    Returns `None` on empty / too-short input. A 4-byte stub (just flags)
    decodes to an effector with all modulators `None`.
    """
    if blob is None or len(blob) < 4:
        return None
    raw = bytes(blob)
    flags = struct.unpack(">I", raw[:4])[0]
    pos = 4

    pressure = None
    velocity = None
    smooth = None
    random_ = None
    tilt = None
    version = 0

    if flags & EFFECTOR_FLAG_PRESSURE and pos + 8 <= len(raw):
        p_min, p_curve = struct.unpack(">fI", raw[pos : pos + 8])
        pressure = EffectorPressure(min=p_min, curve_id=p_curve)
        pos += 8

    if flags & EFFECTOR_FLAG_VELOCITY and pos + 8 <= len(raw):
        v_min, v_curve = struct.unpack(">fI", raw[pos : pos + 8])
        velocity = EffectorVelocity(min=v_min, curve_id=v_curve)
        pos += 8

    if flags & EFFECTOR_FLAG_SMOOTH and pos + 4 <= len(raw):
        (s_min,) = struct.unpack(">f", raw[pos : pos + 4])
        smooth = EffectorSmooth(min=s_min)
        pos += 4

    if flags & EFFECTOR_FLAG_RANDOM and pos + 4 <= len(raw):
        (r_min,) = struct.unpack(">f", raw[pos : pos + 4])
        random_ = EffectorRandom(min=r_min)
        pos += 4

    if flags & EFFECTOR_FLAG_TILT and pos + 4 <= len(raw):
        (t_min,) = struct.unpack(">f", raw[pos : pos + 4])
        tilt = EffectorTilt(min=t_min)
        pos += 4

    # ``velocity_max`` is read directly after ``tilt_min`` (no separate
    # version field; for short blobs this read short-circuits via the
    # bounds check).
    version = 0
    if velocity is not None and pos + 4 <= len(raw):
        (v_max,) = struct.unpack(">f", raw[pos : pos + 4])
        velocity.max = v_max
        pos += 4

    return BrushEffector(
        flags=flags,
        pressure=pressure,
        velocity=velocity,
        smooth=smooth,
        random=random_,
        tilt=tilt,
        version=version,
        raw=raw,
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
