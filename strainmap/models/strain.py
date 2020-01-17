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
    """Calculate the cylindrical coordinates out of X, Y, Z, an origin and theta0.

    Notice that this is different from the velocity calculations where origin and theta0
    might depend on time (different frames/cines), not on Z (different heart slices).

    In other words, these are the spacial coordinates for a specific time frame and all
    relevant short axis slices (typically there will be 8 of them)."""

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


def reduce_array(data: np.ndarray, *masks: np.ndarray, axis: tuple) -> np.ndarray:
    """Reduces array in cylindrical coordinates to the non-zero elements in the masks.

    The masks must have the same shape than the input array.

    In the case of interest of having two masks, the radial and angular masks,
    these define a region of interest in 2D space in the shape of
    a torus, with Na angular segments and Nr radial segments. The rest of the
    space is not relevant. This means that a large 2D array with data can be reduced to
    a much smaller and easy to handle (Nr, Na) array, where the value of each entry
    is the mean values of the pixels in the regions defined by both masks.

    Examples:
        This example reduces the (8, 8) angular array (serving also as input data) to an
        array of shape (Nr=2, Na=4) where each element is the average of the input
        pixels in the 8 regions (2x4) defined by the angular and radial masks. There is
        no radial dependency in the reduced array (all rows are the same) because there
        is no radial dependency in the input array either.
        >>> radial = np.array([
        ...     [0, 0, 0, 0, 0, 0, 0, 0],
        ...     [0, 2, 2, 2, 2, 2, 2, 0],
        ...     [0, 2, 1, 1, 1, 1, 2, 0],
        ...     [0, 2, 1, 0, 0, 1, 2, 0],
        ...     [0, 2, 1, 0, 0, 1, 2, 0],
        ...     [0, 2, 1, 1, 1, 1, 2, 0],
        ...     [0, 2, 2, 2, 2, 2, 2, 0],
        ...     [0, 0, 0, 0, 0, 0, 0, 0],
        ... ])
        >>> angular = np.array([
        ...     [1, 1, 1, 1, 4, 4, 4, 4],
        ...     [1, 1, 1, 1, 4, 4, 4, 4],
        ...     [1, 1, 1, 1, 4, 4, 4, 4],
        ...     [1, 1, 1, 1, 4, 4, 4, 4],
        ...     [2, 2, 2, 2, 3, 3, 3, 3],
        ...     [2, 2, 2, 2, 3, 3, 3, 3],
        ...     [2, 2, 2, 2, 3, 3, 3, 3],
        ...     [2, 2, 2, 2, 3, 3, 3, 3],
        ... ])
        >>> reduced = reduce_array(angular, radial, angular, axis=(0, 1))
        >>> print(reduced)
        [[1 2 3 4]
         [1 2 3 4]]

        We can repeat this using the radial mask as input. In this case, there is no
        angular dependency, as expected.
        >>> reduced = reduce_array(radial, radial, angular, axis=(0, 1))
        >>> print(reduced)
        [[1 1 1 1]
         [2 2 2 2]]

        In general, if there are no symmetries in the input array, all elements of the
        reduced array will be different.
        >>> np.random.seed(12345)
        >>> reduced = reduce_array(np.random.rand(*radial.shape), radial, angular,
        ...     axis=(0, 1))
        >>> print(reduced)
        [[0.89411584 0.46596842 0.17654222 0.51028107]
         [0.79289128 0.28042882 0.73393468 0.18159693]]

    The reduced array has the dimensions defined in axis removed and the extra
    dimensions (one per mask) added to the end. So if data shape is (N0, N1, N2, N3, N4)
    and axis is (1, 2), then the reduced array in the case of having the above radial
    and angular masks will have shape (N0, N3, N4, Nr, Na)
    """
    from numpy.ma import MaskedArray

    assert len(masks) > 0
    assert all([data.shape == m.shape for m in masks])
    assert len(data.shape) > max(axis)

    indices = [set(m.flatten()) - {0} for m in masks]
    shape = [s for i, s in enumerate(data.shape) if i not in axis] + [
        len(idx) for idx in indices
    ]
    reduced = np.zeros(shape, dtype=data.dtype)
    for idx in product(*indices):
        elements = tuple([...] + [k - 1 for k in idx])
        condition = ~np.all([masks[i] == k for i, k in enumerate(idx)], axis=0)
        reduced[elements] = MaskedArray(data, condition).mean(axis=axis).data
    return reduced


def expand_array(reduced: np.ndarray, *masks: np.ndarray, axis: tuple) -> np.ndarray:
    """Transforms a reduced array into a full array with the same shape as the masks.

    This function, partially opposite to `reduce_array`, will recover a full size array
    with the same shape as the masks and with the masked elements equal to the
    corresponding entries of the reduced array. All other elements are masked.

        >>> radial = np.array([
        ...     [0, 0, 0, 0, 0, 0, 0, 0],
        ...     [0, 2, 2, 2, 2, 2, 2, 0],
        ...     [0, 2, 1, 1, 1, 1, 2, 0],
        ...     [0, 2, 1, 0, 0, 1, 2, 0],
        ...     [0, 2, 1, 0, 0, 1, 2, 0],
        ...     [0, 2, 1, 1, 1, 1, 2, 0],
        ...     [0, 2, 2, 2, 2, 2, 2, 0],
        ...     [0, 0, 0, 0, 0, 0, 0, 0],
        ... ])
        >>> angular = np.array([
        ...     [1, 1, 1, 1, 4, 4, 4, 4],
        ...     [1, 1, 1, 1, 4, 4, 4, 4],
        ...     [1, 1, 1, 1, 4, 4, 4, 4],
        ...     [1, 1, 1, 1, 4, 4, 4, 4],
        ...     [2, 2, 2, 2, 3, 3, 3, 3],
        ...     [2, 2, 2, 2, 3, 3, 3, 3],
        ...     [2, 2, 2, 2, 3, 3, 3, 3],
        ...     [2, 2, 2, 2, 3, 3, 3, 3],
        ... ])
        >>> reduced = reduce_array(angular, radial, angular, axis=(0, 1))
        >>> print(reduced)
        [[1 2 3 4]
         [1 2 3 4]]

        Now we "recover" the original full size array:
        >>> data = expand_array(reduced, radial, angular, axis=(0, 1))
        >>> print(data)
        [[-- -- -- -- -- -- -- --]
         [-- 1 1 1 4 4 4 --]
         [-- 1 1 1 4 4 4 --]
         [-- 1 1 -- -- 4 4 --]
         [-- 2 2 -- -- 3 3 --]
         [-- 2 2 2 3 3 3 --]
         [-- 2 2 2 3 3 3 --]
         [-- -- -- -- -- -- -- --]]
    """
    from numpy.ma import MaskedArray

    assert len(masks) > 0
    assert all([masks[0].shape == m.shape for m in masks[1:]])
    assert len(masks[0].shape) > max(axis)

    indices = reduced.shape[-len(masks) :]
    data = np.zeros_like(masks[0], dtype=reduced.dtype)

    for idx in product(*[range(i) for i in indices]):
        condition = np.all([masks[i] == k + 1 for i, k in enumerate(idx)], axis=0)
        values = reduced[(...,) + idx]
        for ax in axis:
            values = np.expand_dims(values, ax)
        data[condition] = (values * condition)[condition]

    return MaskedArray(data, ~np.all(masks, axis=0))
