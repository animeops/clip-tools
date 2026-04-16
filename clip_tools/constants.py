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
