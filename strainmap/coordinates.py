from enum import Enum, auto
from typing import Tuple


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
    PS: Tuple[int, int, bool] = (1, 15, True)
    PD: Tuple[int, int, bool] = (15, 35, False)
    PAS: Tuple[int, int, bool] = (35, 47, False)
    PC1: Tuple[int, int, bool] = (1, 5, False)
    PC2: Tuple[int, int, bool] = (6, 12, True)
    PC3: auto()
    ES: Tuple[int, int] = (14, 21)
    PSS = auto()
    ESS = auto()