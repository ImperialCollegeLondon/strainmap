import numpy as np
from itertools import product

from typing import Tuple, Union


def cartcoords(
    shape: tuple, zsize: Union[float, np.ndarray], xsize: float, ysize: float
) -> Tuple:
    """Create cartesian coordinates of the given shape based on the pixel sizes.

    As slices can be separated by a different number, zsize can also be an array of
    positions in the Z direction rather than an scalar.
    """
    if isinstance(zsize, np.ndarray):
        z = zsize - zsize[0]
    else:
        z = np.linspace(0, zsize, shape[0])
    x = np.linspace(0, xsize, shape[1])
    y = np.linspace(0, ysize, shape[2])

    return z, x, y


def cylcoords(
    z: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    origin: np.ndarray,
    theta0: Union[float, np.ndarray],
) -> Tuple:
    """Calculate the cylindrical coordinates out of X, Y Z, an origin and theta0."""

    org = validate_origin(origin, len(z))
    th0 = validate_theta0(theta0, len(z))

    zz, xx, yy = np.meshgrid(z, x, y, indexing="ij")
    xx -= org[:, -2:-1, None]
    yy -= org[:, -1:, None]
    r = np.sqrt(xx ** 2 + yy ** 2)
    theta = np.arctan2(yy, xx) - th0[:, None, None]
    return zz, r, theta


def validate_origin(origin: np.ndarray, lenz: int):
    """Validates the shape of the origin array."""

    msg = "Origin must be a 1D array of length 2 or a 2D array of shape (len(z), 2)"
    if len(origin.shape) == 1:
        assert len(origin) == 2, msg
        return np.tile(origin, (lenz, 1))
    elif len(origin.shape) == 2:
        assert origin.shape == (lenz, 2), msg
        return origin
    else:
        raise ValueError(msg)


def validate_theta0(theta0: Union[float, np.ndarray], lenz: int):
    """Validates the shape of the theta0 array."""
    if isinstance(theta0, np.ndarray):
        assert theta0.shape == (lenz,), f"If an array, theta0 must have shape {(lenz,)}"
        return theta0
    else:
        return np.array([theta0] * lenz)


def reduce_array(
    data: np.ndarray, angular: np.ndarray, radial: np.ndarray
) -> np.ndarray:
    from numpy.ma import MaskedArray

    assert data.shape == angular.shape == radial.shape
    assert len(data.shape) == 3

    aidx = set(angular.flatten()) - {0}
    ridx = set(radial.flatten()) - {0}

    unfolded = np.zeros((data.shape[0], len(ridx), len(aidx)))
    for i, (r, a) in enumerate(product(ridx, aidx)):
        unfolded[:, r - 1, a - 1] = (
            MaskedArray(data, ~np.logical_and(radial == r, angular == a))
            .mean(axis=(1, 2))
            .data
        )

    return unfolded
