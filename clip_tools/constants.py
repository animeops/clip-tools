import enum

DEBUG = False


class TextStylingType(enum.Enum):
    NONE = 0
    BOLD = 1
    ITALIC = 2
    BOLD_ITALIC = 3


class TextWindowType(enum.Enum):
    NONE = 0
    STRETCHED = 2
    SPECIAL = 3


class TextAlignmentType(enum.Enum):
    LEFT = 0
    RIGHT = 1
    CENTER = 2


class TextHollowType(enum.Enum):
    NONE = 0
    THIN = 1
    THICK = 2


class VectorType(enum.Enum):
    STANDARD = 1
    BEZIER = 2
    CURVE = 3


class LayerComposite(enum.IntEnum):
    """`Layer.LayerComposite` (per-layer blend mode).

    Values 27, 28, 29, 31..35 unused/unknown. 30 (PASS_THROUGH) is folder-only.
    """

    NORMAL = 0
    DARKEN = 1
    MULTIPLY = 2
    COLOR_BURN = 3
    LINEAR_BURN = 4
    SUBTRACT = 5
    DARKER_COLOR = 6
    LIGHTEN = 7
    SCREEN = 8
    COLOR_DODGE = 9
    GLOW_DODGE = 10
    ADD = 11
    ADD_GLOW = 12
    LIGHTER_COLOR = 13
    OVERLAY = 14
    SOFT_LIGHT = 15
    HARD_LIGHT = 16
    VIVID_LIGHT = 17
    LINEAR_LIGHT = 18
    PIN_LIGHT = 19
    HARD_MIX = 20
    DIFFERENCE = 21
    EXCLUSION = 22
    HUE = 23
    SATURATION = 24
    COLOR = 25
    LUMINOSITY = 26
    PASS_THROUGH = 30
    DIVIDE = 36


class LayerKind(enum.IntEnum):
    """`Layer.LayerType`.

    Bit 1 (`value & 2`) is the "has layer mask" bit. Pure raster vs masked
    raster differ only in that bit. Smaller values may exist (the table
    below picks the canonical observed value per kind).
    """

    OTHER = 0  # vector, regular folder, 3d, frame folder, gradient, fill, tone
    RASTER = 1
    OTHER_MASKED = 2
    RASTER_MASKED = 3
    DUMMY = 256  # root folder
    PAPER = 1584  # paper layer (one per canvas)
    FILTER = 4096  # adjustment / filter layer
    FILTER_MASKED = 4098


class FilterLayerKind(enum.IntEnum):
    """First u32 of a `FilterLayerInfo` BLOB — the adjustment-layer type."""

    BRIGHTNESS_CONTRAST = 1
    LEVEL_CORRECTION = 2
    TONE_CURVE = 3
    HSL = 4
    COLOR_BALANCE = 5
    REVERSE_GRADIENT = 6
    POSTERIZATION = 7
    BINARIZATION = 8
    GRADIENT_MAP = 9


class CanvasUnit(enum.IntEnum):
    """`Canvas.CanvasUnit`. Value 4 is unused."""

    PIXELS = 0
    CENTIMETRES = 1
    MILLIMETRES = 2
    INCHES = 3
    POINTS = 5


class LayerLockBit(enum.IntFlag):
    """Bit flags packed into the `Layer.LayerLock` u32 column."""

    EDIT = 1 << 0  # whole-layer edit lock
    ALPHA = 1 << 4  # transparency lock


class LayerFolderBit(enum.IntFlag):
    """Bit flags packed into the `Layer.LayerFolder` u32 column."""

    CLOSED = 1 << 4  # folder is collapsed in the UI


class ChunkMagic:
    """Top-level chunk-type tags found in the binary section."""

    FILE = b"CSFCHUNK"
    HEADER = b"CHNKHead"
    EXTERNAL = b"CHNKExta"
    SQLITE = b"CHNKSQLi"
    FOOTER = b"CHNKFoot"


class BlockMarker:
    """UTF-16 BE name tags inside an Offscreen.BlockData stream."""

    BLOCK_DATA_BEGIN = "BlockDataBeginChunk".encode("utf-16be")
    BLOCK_DATA_END = "BlockDataEndChunk".encode("utf-16be")
    BLOCK_STATUS = "BlockStatus".encode("utf-16be")
    BLOCK_CHECK_SUM = "BlockCheckSum".encode("utf-16be")
