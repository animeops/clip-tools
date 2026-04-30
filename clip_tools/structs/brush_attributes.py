import struct
from typing import List


def parse_brush_pattern_image_index(blob: bytes) -> List[int]:
    """Decode ``BrushPatternStyle.ImageIndex`` — a packed big-endian uint32
    array of ``BrushPatternImage.MainId`` values."""
    n = len(blob) // 4
    if n == 0:
        return []
    return list(struct.unpack(f">{n}I", blob))
