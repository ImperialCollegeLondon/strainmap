from itertools import product
from typing import Dict, Sequence, Text, Tuple, Union

import numpy as np

from .readers import DICOMReaderBase
from .strainmap_data_model import StrainMapData
from .velocities import find_theta0


def cartcoords(shape: tuple, *sizes: Union[float, np.ndarray]) -> Tuple:
    """Create cartesian coordinates of the given shape based on the pixel sizes.

    sizes can be an scalar indicating the step value of the coordinate in the given
    dimension, or an array already indicating the positions. In the later case, these
    are shifted so the coordinate starts at 0.
    """
    assert len(shape) == len(sizes)

    def build_coordinate(length, value):
        if isinstance(value, np.ndarray):
            return value - value[0]
        else:
            return np.linspace(0, value * length, length, endpoint=False)

    return tuple(map(build_coordinate, shape, sizes))


def cylcoords(
    z: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    origin: np.ndarray,
    theta0: Union[float, np.ndarray],
    lent: int,
) -> Tuple:
    """Calculate the cylindrical coordinates out of X, Y, Z, an origin and theta0.

    Notice that this is different from the velocity calculations where origin and theta0
    might depend on time (different frames/cines), not on Z (different heart slices).

    In other words, these are the spacial coordinates for a specific time frame and all
    relevant short axis slices (typically there will be 8 of them).
    """

    org = validate_origin(origin, len(z), lent)
    th0 = validate_theta0(theta0, len(z), lent)

    coords = np.meshgrid(z, x, y, indexing="ij")
    zz, xx, yy = (np.tile(c, (lent, 1, 1, 1)) for c in coords)
    xx -= org[:, :, -2:-1, None]
    yy -= org[:, :, -1:, None]
    r = np.sqrt(xx ** 2 + yy ** 2)
    theta = np.arctan2(yy, xx) - th0[:, :, None, None]
    return zz, r, theta


def validate_origin(origin: np.ndarray, lenz: int, lent: int):
    """Validates the shape of the origin array."""

    msg = (
        f"Origin must be a 1D array of length 2, a 2D array of shape ({lenz}, 2) or"
        f"({lent}, 2) or a 3D array of shape ({lent}, {lenz}, 2)"
    )
    if origin.shape == (2,):
        return np.tile(origin, (lent, lenz, 1))
    elif origin.shape == (lenz, 2):
        return np.tile(origin, (lent, 1, 1))
    elif origin.shape == (lent, 2):
        return np.tile(origin, (lenz, 1, 1)).transpose((1, 0, 2))
    elif origin.shape == (lent, lenz, 2):
        return origin
    else:
        raise ValueError(msg)


def validate_theta0(theta0: Union[float, np.ndarray], lenz: int, lent: int):
    """Validates the shape of the theta0 array."""
    if isinstance(theta0, np.ndarray):
        msg = (
            f"If an array, theta0 must have shape ({lenz},), ({lent},) or "
            f"({lent}, {lenz})"
        )
        if theta0.shape == (lenz,):
            return np.tile(theta0, (lent, 1))
        elif theta0.shape == (lent,):
            return np.tile(theta0, (lenz, 1)).T
        elif theta0.shape == (lent, lenz):
            return theta0
        else:
            raise ValueError(msg)
    else:
        return np.full((lent, lenz), theta0)


def masked_reduction(data: np.ndarray, *masks: np.ndarray, axis: tuple) -> np.ndarray:
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
        >>> reduced = masked_reduction(angular, radial, angular, axis=(0, 1))
        >>> print(reduced)
        [[1 2 3 4]
         [1 2 3 4]]

        We can repeat this using the radial mask as input. In this case, there is no
        angular dependency, as expected.
        >>> reduced = masked_reduction(radial, radial, angular, axis=(0, 1))
        >>> print(reduced)
        [[1 1 1 1]
         [2 2 2 2]]

        In general, if there are no symmetries in the input array, all elements of the
        reduced array will be different.
        >>> np.random.seed(12345)
        >>> reduced = masked_reduction(np.random.rand(*radial.shape), radial, angular,
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


def masked_expansion(
    reduced: np.ndarray, *masks: np.ndarray, axis: tuple
) -> np.ndarray:
    """Transforms a reduced array into a full array with the same shape as the masks.

    This function, partially opposite to `masked_reduction`, will recover a full size
    array with the same shape as the masks and with the masked elements equal to the
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
        >>> reduced = masked_reduction(angular, radial, angular, axis=(0, 1))
        >>> print(reduced)
        [[1 2 3 4]
         [1 2 3 4]]

        Now we "recover" the original full size array:
        >>> data = masked_expansion(reduced, radial, angular, axis=(0, 1))
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


def prepare_coordinates(
    data: DICOMReaderBase, zero_angle: Dict[str, np.ndarray], datasets: Tuple[str, ...]
):
    """Prepares the coordinate arrays to calculate the strain.

    Returns:
        - time - Array with shape (time, z)
        - space - Array with shape (component, z, r, theta)
    """
    lenz = len(datasets)
    lent, lenx, leny = data.mag(datasets[0]).shape

    z_location = np.zeros(lenz)
    t_interval = np.zeros(lenz)
    theta0 = np.zeros((lent, lenz))
    origin = np.zeros((lent, lenz, 2))

    for i, d in enumerate(datasets):
        z_location[i] = data.slice_loc(d)
        t_interval[i] = data.time_interval(d)

        theta0[:, i] = find_theta0(zero_angle[d])
        origin[:, i, :] = zero_angle[d][:, :, 1]

    px_size = data.pixel_size(datasets[0])
    z = z_location - z_location[0]
    x = np.linspace(0, px_size * lenx, lenx, endpoint=False)
    y = np.linspace(0, px_size * leny, leny, endpoint=False)
    zz, r, theta = cylcoords(z, x, y, origin, theta0, lent)
    time = np.linspace(0, t_interval * lent, lent, endpoint=False)

    return time, np.array([zz, r, theta])


def prepare_masks_and_velocities(
    masks: Dict[str, Dict[str, np.ndarray]],
    datasets: Tuple[str, ...],
    nrad: int = 3,
    nang: int = 24,
    background: str = "Estimated",
):
    """Prepare the masks and cylindrical velocities to be used in strain calculation.

    Returns:
        - velocities - Array with shape (component, time, z, r, theta)
        - radial masks - Array with shape (time, z, r, theta)
        - angular masks - Array with shape (time, z, r, theta)
    """
    vkey = f"cylindrical - {background}"
    rkey = f"radial x{nrad} - {background}"
    akey = f"angular x{nang} - {background}"

    vel = []
    radial = []
    angular = []

    for i, d in enumerate(datasets):
        vel.append(masks[d][vkey])
        radial.append(masks[d][rkey])
        angular.append(masks[d][akey])

    vel = np.array(vel).transpose((1, 2, 0, 3, 4))
    radial = np.array(radial).transpose((1, 0, 2, 3))
    angular = np.array(angular).transpose((1, 0, 2, 3))

    return vel, radial, angular


def calculate_strain(data: StrainMapData, datasets: Tuple[str, ...]):
    """Calculates the strain and updates the Data object with the result."""
    time, space = prepare_coordinates(data.data_files, data.zero_angle, datasets)
    vel, radial, angular = prepare_masks_and_velocities(data.masks, datasets)
    reduced_vel = masked_reduction(vel, radial, angular, axis=vel.shape[-2:])
    reduced_space = masked_reduction(space, radial, angular, axis=space.shape[-2:])
    reduced_strain = differentiate(reduced_vel, reduced_space, time)
    strain = masked_expansion(reduced_strain, radial, angular, axis=vel.shape[-2:])
    data.strain = calculate_regional_strain(strain, data.masks, datasets)


def gridded_ring(
    radii: Union[Tuple[float, float], Sequence[float]],
    ntheta: int = 50,
    nr: int = 20,
    origin: Sequence[float] = (0, 0),
    theta0: float = 0,
    mode: Text = "cartesian",
) -> np.ndarray:
    """Regular grid of a ring.

    Returns a (n, 2) array of coordinates corresponding to the points of a cylindrical
    grid within the given radii. The angular part of the grid starts at ``theta0``, so
    that different slices with different ``theta0`` are aligned (e.g. the angle of point
    0 on one slice corresponds
    to theta0).

    Example:

        >>> import numpy as np
        >>> from pytest import approx
        >>> from strainmap.models.strain import gridded_ring

        if ``mode=="cartesian"`` (default), then the points are returned in cartesian
        coordinates.

        >>> coords = gridded_ring((3, 4), ntheta=4, nr=3)

        The coordinates first iterate over the rods.

        >>> assert coords[0] == approx([3, 0])
        >>> assert coords[1] == approx([3.5, 0])
        >>> assert coords[2] == approx([4, 0])

        Once a rod is done, the next one starts.

        >>> coords[3] == approx([3 * np.cos(2 * np.pi / 4), 3 * np.sin(2 * np.pi / 4)])
        True

        If `mode=="cylindrical"`, then the points are returned in cylindrical
        coordinates, with the radius corresponding to the first component, and the angle
        to the second:

        >>> coords = gridded_ring((3, 4), ntheta=4, nr=3, mode="cylindrical")
        >>> assert coords[0] == approx([3, 0])
        >>> assert coords[1] == approx([3.5, 0])
        >>> assert coords[2] == approx([4, 0])
        >>> assert coords[3] == approx([3, 2 * np.pi / 4])
    """
    if mode.lower() not in {"cartesian", "cylindrical"}:
        raise ValueError("Expected either 'cartesian' or 'cylindrical'")
    rs = np.linspace(start=min(*radii), stop=max(*radii), num=nr, endpoint=True,)
    thetas = np.linspace(start=0, stop=2 * np.pi, num=ntheta, endpoint=False) + theta0
    if mode.lower() == "cartesian":
        x = (rs[None, :] * np.cos(thetas)[:, None]).reshape(-1, 1)
        y = (rs[None, :] * np.sin(thetas)[:, None]).reshape(-1, 1)
        return np.array(origin) + np.concatenate((x, y), axis=1)
    return np.concatenate(
        (
            np.repeat(rs[None, :, None], len(thetas), axis=0),
            np.repeat(thetas[:, None, None], len(rs), axis=1),
        ),
        axis=2,
    ).reshape(-1, 2)


def differentiate(reduced_vel, reduced_space, time) -> np.ndarray:
    """Calculate the strain out of the velocity data."""


def calculate_regional_strain(strain, masks, datasets) -> Dict:
    """Calculate the regional strains (1D curves)."""
