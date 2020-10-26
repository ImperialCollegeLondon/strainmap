from enum import Enum, auto


class Region(Enum):
    GLOBAL = 1
    ANGULAR_x6 = 6
    ANGULAR_x24 = 24
    RADIAL_x3 = 3


class Comp(Enum):
    MAG = "mag"
    X = "x"
    Y = "y"
    Z = "z"
    RAD = "Radial"
    CIRC = "Circumferential"
    LONG = "Longitudinal"


class Mark(Enum):
    PS = auto()
    PD = auto()
    PAS = auto()
    ES = auto()
    PC1 = auto()
    PC2 = auto()
    PC3 = auto()
    PSS = auto()
    ESS = auto()
