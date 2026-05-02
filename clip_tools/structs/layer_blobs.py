"""Parsers for the smaller per-layer SQL blob columns.

`MonochromeFillInfo` and `LightTableInfo` both use a "named record" form
common across the CSP format: a length-prefixed UTF-16BE name followed by
an opaque payload. Their payload semantics aren't decoded yet — what we
expose is the structural framing (size, name, payload bytes) so callers
can identify and round-trip them.
"""

import struct
from dataclasses import dataclass
from typing import Optional


@dataclass
class MonochromeFillInfo:
    """`Layer.MonochromeFillInfo` — paper-layer monochrome settings.

    Layout::

        u32 BE total_size
        u32 BE record_count   (=1 in observed samples)
        u32 BE name_length    (chars; observed = 17)
        UTF-16BE name         (observed = "MonochromeSetting")
        bytes payload         (observed: 12 zero bytes + u32 BE = 1)
    """

    total_size: int
    record_count: int
    name: str
    payload: bytes


def parse_monochrome_fill_info(blob: bytes) -> MonochromeFillInfo:
    if len(blob) < 12:
        raise ValueError(f"MonochromeFillInfo too short: {len(blob)}B")
    total_size, record_count, name_length = struct.unpack_from(">III", blob, 0)
    if total_size != len(blob):
        raise ValueError(
            f"MonochromeFillInfo total_size mismatch: header={total_size}, blob={len(blob)}"
        )
    name_bytes = blob[12 : 12 + name_length * 2]
    name = name_bytes.decode("utf-16-be", errors="replace")
    payload = bytes(blob[12 + name_length * 2 :])
    return MonochromeFillInfo(
        total_size=total_size,
        record_count=record_count,
        name=name,
        payload=payload,
    )


@dataclass
class LightTableInfo:
    """`Layer.LightTableInfo` — onion-skin / light-table per-layer config.

    Wire shape across observed samples (15B total)::

        bytes[3]   prefix       (observed: 01 01 01)
        u8         key_length
        bytes      key (ASCII)  (observed: "typename")
        bytes      tail         (observed: 06 00 00)

    The exact split between "key" and "value" is provisional — across
    samples the entire blob is byte-identical (only the one observed
    payload `01 01 01 08 typename 06 00 00`), so a deeper structural
    decode needs more diversity.
    """

    raw: bytes
    prefix: bytes
    key: str
    tail: bytes


def parse_light_table_info(blob: bytes) -> LightTableInfo:
    if len(blob) < 4:
        raise ValueError(f"LightTableInfo too short: {len(blob)}B")
    prefix = bytes(blob[:3])
    key_length = blob[3]
    key_bytes = blob[4 : 4 + key_length]
    key = key_bytes.decode("ascii", errors="replace")
    tail = bytes(blob[4 + key_length :])
    return LightTableInfo(raw=bytes(blob), prefix=prefix, key=key, tail=tail)
