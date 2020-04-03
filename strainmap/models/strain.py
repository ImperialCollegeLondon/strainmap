from typing import Dict, Text, Tuple, Union, Callable
from itertools import product
from collections import defaultdict

import numpy as np

from .readers import DICOMReaderBase
from .strainmap_data_model import StrainMapData
from .velocities import find_theta0, regenerate
from .writers import terminal


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
    xx -= org[:, :, -1:, None]
    yy -= org[:, :, -2:-1, None]
    r = np.sqrt(xx ** 2 + yy ** 2)
    theta = np.mod(np.arctan2(yy, xx) + th0[:, :, None, None], 2 * np.pi)
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


def masked_reduction(data: np.ndarray, masks: np.ndarray, axis: tuple) -> np.ndarray:
    """Reduces array in cylindrical coordinates to the non-zero elements in the
    masks.

    The masks must have the same shape than the input array.

    In the case of interest of having two masks, the radial and angular masks,
    these define a region of interest in 2D space in the shape of
    a torus, with Na angular segments and Nr radial segments. The rest of the
    space is not relevant. This means that a large 2D array with data can be
    reduced to
    a much smaller and easy to handle (Nr, Na) array, where the value of each entry
    is the mean values of the pixels in the regions defined by both masks.

    Examples:
        This example reduces the (8, 8) angular array (serving also as input
        data) to an
        array of shape (Nr=2, Na=4) where each element is the average of the input
        pixels in the 8 regions (2x4) defined by the angular and radial masks.
        There is
        no radial dependency in the reduced array (all rows are the same) because
        there
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

        In general, if there are no symmetries in the input array, all elements
        of the
        reduced array will be different.
        >>> np.random.seed(12345)
        >>> reduced = masked_reduction(np.random.rand(*radial.shape), radial,
        angular,
        ...     axis=(0, 1))
        >>> print(reduced)
        [[0.89411584 0.46596842 0.17654222 0.51028107]
         [0.79289128 0.28042882 0.73393468 0.18159693]]

    The reduced array has the dimensions defined in axis removed and the extra
    dimensions (one per mask) added to the end. So if data shape is (N0, N1, N2,
    N3, N4)
    and axis is (1, 2), then the reduced array in the case of having the above
    radial
    and angular masks will have shape (N0, N3, N4, Nr, Na)
    """
    from numpy.ma import MaskedArray
    from functools import reduce

    assert data.shape[1:] == masks.shape

    mask_max = masks.max()
    nrad, nang = mask_max % 100, mask_max // 100
    nz = np.nonzero(masks)
    xmin, xmax, ymin, ymax = (
        nz[-2].min(),
        nz[-2].max() + 1,
        nz[-1].min(),
        nz[-1].max() + 1,
    )
    sdata = data[..., xmin:xmax, ymin:ymax]
    smasks = masks[..., xmin:xmax, ymin:ymax]

    shape = [s for i, s in enumerate(data.shape) if i not in axis] + [nrad, nang]
    reduced = np.zeros(shape, dtype=data.dtype)

    def reduction(red, idx):
        elements = tuple([...] + [k - 1 for k in idx])
        i = idx[0] + 100 * idx[1]
        red[elements] = (
            MaskedArray(
                sdata, np.tile(smasks != i, (data.shape[0],) + (1,) * len(masks.shape))
            )
            .mean(axis=axis)
            .data
        )
        return red

    return reduce(reduction, product(range(1, nrad + 1), range(1, nang + 1)), reduced)


def masked_expansion(data: np.ndarray, masks: np.ndarray, axis: tuple) -> np.ndarray:
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
    from functools import reduce
    from .velocities import remap_array

    assert len(data.shape) > max(axis)

    mask_max = masks.max()
    nrad, nang = mask_max % 100, mask_max // 100
    nz = np.nonzero(masks)
    xmin, xmax, ymin, ymax = (
        nz[-2].min(),
        nz[-2].max() + 1,
        nz[-1].min(),
        nz[-1].max() + 1,
    )

    shape = (data.shape[0],) + masks[..., xmin:xmax, ymin:ymax].shape
    expanded = np.zeros(shape, dtype=data.dtype)
    exmasks = np.tile(
        masks[..., xmin:xmax, ymin:ymax], (expanded.shape[0],) + (1,) * len(masks.shape)
    )

    def expansion(exp, idx):
        i = idx[0] + 100 * idx[1] + 101
        condition = exmasks == i
        values = data[(...,) + idx].reshape(data.shape[:-2] + (1, 1))
        exp[condition] = (values * condition)[condition]
        return exp

    return remap_array(
        reduce(expansion, product(range(nrad), range(nang)), expanded),
        masks.shape[-2:],
        (xmin, xmax, ymin, ymax),
    )


def prepare_coordinates(
    data: DICOMReaderBase, zero_angle: Dict[str, np.ndarray], datasets: Tuple[str, ...]
):
    """Prepares the coordinate arrays to calculate the strain.

    The time is the time interval between frames and it's defined per dataset.

    Returns:
        - time - Array with shape (z)
        - space - Array with shape (3, frames, z, r, theta)
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
    origin *= px_size
    z = z_location - z_location[0]
    x = np.linspace(0, px_size * lenx, lenx, endpoint=False)
    y = np.linspace(0, px_size * leny, leny, endpoint=False)
    zz, r, theta = cylcoords(z, x, y, origin, theta0, lent)

    return t_interval, np.array([zz, r, theta])


def prepare_masks_and_velocities(
    masks: Dict[str, Dict[str, np.ndarray]],
    datasets: Tuple[str, ...],
    nrad: int = 3,
    nang: int = 24,
    background: str = "Estimated",
):
    """Prepare the masks and cylindrical velocities to be used in strain calculation.

    Returns:
        - velocities - Array with shape (component, frames, z, r, theta)
        - radial masks - Array with shape (frames, z, r, theta)
        - angular masks - Array with shape (frames, z, r, theta)
    """
    vkey = f"cylindrical - {background}"
    rkey = f"radial x{nrad} - {background}"
    akey = f"angular x{nang} - {background}"

    vel = []
    msk = []

    for i, d in enumerate(datasets):
        vel.append(masks[d][vkey])
        msk.append(masks[d][rkey] + 100 * masks[d][akey])

    vel = np.array(vel).transpose((1, 2, 0, 3, 4))
    msk = np.array(msk).transpose((1, 0, 2, 3))

    return vel, msk


def calculate_strain(
    data: StrainMapData, datasets: Tuple[str, ...], callback: Callable = terminal
):
    """Calculates the strain and updates the Data object with the result."""

    # Do we need to calculate the strain?
    if all([d in data.strain.keys() for d in datasets]):
        return

    # Do we need to regenerate the velocities?
    to_regen = [d for d in datasets if list(data.velocities[d].values())[0] is None]
    if len(to_regen) > 0:
        regenerate(data, to_regen, callback=callback)

    sorted_datasets = tuple(sorted(datasets, key=data.data_files.slice_loc))

    callback("Preparing dependent variables", 1 / 5.0)
    vel, masks = prepare_masks_and_velocities(data.masks, sorted_datasets)
    img_axis = tuple(range(len(vel.shape)))[-2:]
    reduced_vel = masked_reduction(vel, masks, axis=img_axis)
    del vel

    callback("Preparing independent variables", 2 / 5.0)
    time, space = prepare_coordinates(data.data_files, data.zero_angle, sorted_datasets)
    reduced_space = masked_reduction(space, masks, axis=img_axis)
    del space

    callback("Calculating derivatives", 3 / 5.0)
    reduced_strain = differentiate(reduced_vel, reduced_space, time)
    strain = masked_expansion(reduced_strain, masks, axis=img_axis)
    del masks

    callback("Calculating the regional strains", 4 / 5.0)
    data.strain = calculate_regional_strain(strain, data.masks, sorted_datasets)

    callback("Done!", 1)


def differentiate(vel, space, time) -> np.ndarray:
    """ Calculate the strain out of the velocity data.

    Args:
        vel: Array of velocities with shape (3, frames, z, nrad, nang)
        space: Array of spatial locations with shape (3, frames, z, nrad, nang)
        time: Array with shape (z)

    Returns:
        Array with shape (3, frames, z, nrad, nang) containing the strain.
    """
    disp = np.cumsum(vel * time[None, None, :, None, None], axis=1)

    dvz = finite_differences(disp[0], space[0], axis=1)
    dvr = finite_differences(disp[1], space[1], axis=2)
    dvth = finite_differences(disp[2], space[2], axis=3, period=2 * np.pi)

    dvth = dvth / space[1] + disp[1] / space[1]

    return np.array([dvz, dvr, dvth])


def finite_differences(f, x, axis=0, period=None):
    """ Calculate the finite differences of a multidimensional array along an axis.

    `f` and `x` must have the same shape and the grid defined by `x` can be
    inhomogeneous. If period is None, central differences are in the interior points
    and forward/backward differences in the boundaries. Otherwise, central differences
    are used everywhere, considering the periodicity of the function.

    Args:
        f: Array with the values.
        x: Array with the positions of each of the values.
        axis: Axis along which to calculate the derivative.
        period: None or the period, in case of a periodic derivative.

    Returns:
        An array with the same shape as f and x with the derivatives along axis.
    """
    assert f.shape == x.shape

    f0 = np.moveaxis(f, axis, 0)
    x0 = np.moveaxis(x, axis, 0)
    result = np.zeros_like(f0)

    # Interior points
    b = abs(x0[2:] - x0[1:-1])
    c = abs(x0[:-2] - x0[1:-1])
    result[1:-1] = ((f0[2:] - f0[1:-1]) / c + (f0[1:-1] - f0[:-2]) / b) / 2

    # Boundaries
    if period:
        a = abs(x0[-1] - x0[0] - period)
        b = abs(x0[1] - x0[0])
        c = abs(x0[-2] - x0[-1])
        result[0] = ((f0[1] - f0[0]) / b + (f0[0] - f0[-1]) / a) / 2
        result[-1] = ((f0[-1] - f0[-2]) / c + (f0[0] - f0[-1]) / a) / 2
    else:
        b = abs(x0[1] - x0[0])
        c = abs(x0[-2] - x0[-1])
        result[0] = (f0[1] - f0[0]) / b
        result[-1] = (f0[-1] - f0[-2]) / c

    return np.moveaxis(result, 0, axis)


def calculate_regional_strain(strain, masks, datasets) -> Dict:
    """Calculate the regional strains (1D curves)."""
    from strainmap.models.contour_mask import masked_means

    result: Dict[Text, Dict[str, np.ndarray]] = defaultdict(dict)
    for i, d in enumerate(datasets):
        for k, m in masks[d].items():
            if "global" in k or "6" in k:
                result[d][k] = masked_means(
                    np.take(strain, indices=i, axis=2), m, axes=(2, 3)
                )
            elif "cylindrical" in k:
                result[d][k] = np.take(strain, indices=i, axis=2)

    return result
