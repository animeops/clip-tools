from dataclasses import dataclass

import numpy as np


@dataclass
class Point:
    x: int
    y: int
    opacity: float
    thickness: float


@dataclass
class LayerEntry:
    """A parsed layer image and its classification.

    type is "raster" or "vector" for renderable entries; auxiliary entries
    use values like "invalid", "clipped", "group", "mipmap", "other".
    """

    type: str
    image: np.ndarray


@dataclass
class ExternalIdEntry:
    """Describes where an external-id chunk belongs in the SQLite schema."""

    table_name: str
    column_name: str
    found: bool = False
