from typing import Dict, Text, Tuple, Callable
from itertools import product
from functools import partial
from collections import defaultdict

import numpy as np
from scipy import interpolate

from .strainmap_data_model import StrainMapData
from .velocities import find_theta0, regenerate
from .writers import terminal


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
        >>> mask = angular * 100 + 1
        >>> reduced = masked_reduction(angular, mask, axis=(0, 1))
        >>> print(reduced)
        [[1 2 3 4]]

        We can repeat this using the radial mask as input. In this case, there is no
        angular dependency, as expected.
        >>> mask = radial + 100
        >>> reduced = masked_reduction(radial, mask, axis=(0, 1))
        >>> print(reduced)
        [[1]
         [2]]

        In general, if there are no symmetries in the input array, all elements
        of the
        reduced array will be different.
        >>> np.random.seed(12345)
        >>> mask = radial + 100 * angular
        >>> reduced = masked_reduction(np.random.rand(*radial.shape), mask, axis=(0, 1))
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

    assert data.shape[-len(masks.shape) :] == masks.shape

    mask_max = masks.max()
    nrad, nang = mask_max % 100, mask_max // 100
    nz = np.nonzero(masks)
    xmin, xmax, ymin, ymax = (
        nz[-2].min(),
        nz[-2].max() + 1,
        nz[-1].min(),
        nz[-1].max() + 1,
    )
    sdata = data[..., xmin : xmax + 1, ymin : ymax + 1]
    smasks = masks[..., xmin : xmax + 1, ymin : ymax + 1]

    shape = [s for i, s in enumerate(data.shape) if i not in axis] + [nrad, nang]
    reduced = np.zeros(shape, dtype=data.dtype)

    tile_shape = (
        (data.shape[0],) + (1,) * len(masks.shape)
        if data.shape != masks.shape
        else (1,) * len(masks.shape)
    )

    def reduction(red, idx):
        elements = tuple([...] + [k - 1 for k in idx])
        i = idx[0] + 100 * idx[1]
        red[elements] = (
            MaskedArray(sdata, np.tile(smasks != i, tile_shape)).mean(axis=axis).data
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
        >>> mask = angular * 100 + 1
        >>> reduced = masked_reduction(angular, mask, axis=(0, 1))
        >>> print(reduced)
        [[1 2 3 4]]

        Now we "recover" the original full size array:
        >>> data = masked_expansion(reduced, mask, axis=(0, 1))
        >>> print(data)
        [[[1 1 1 1 4 4 4 4]
          [1 1 1 1 4 4 4 4]
          [1 1 1 1 4 4 4 4]
          [1 1 1 1 4 4 4 4]
          [2 2 2 2 3 3 3 3]
          [2 2 2 2 3 3 3 3]
          [2 2 2 2 3 3 3 3]
          [2 2 2 2 3 3 3 3]]]
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

    shape = (data.shape[0],) + masks[..., xmin : xmax + 1, ymin : ymax + 1].shape
    expanded = np.zeros(shape, dtype=data.dtype)
    exmasks = np.tile(
        masks[..., xmin : xmax + 1, ymin : ymax + 1],
        (expanded.shape[0],) + (1,) * len(masks.shape),
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


def coordinates(
    data: StrainMapData,
    datasets: Tuple[str, ...],
    nrad: int = 3,
    nang: int = 24,
    background: str = "Estimated",
    resample=True,
) -> np.ndarray:

    rkey = f"radial x{nrad} - {background}"
    akey = f"angular x{nang} - {background}"

    m_iter = (data.masks[d][rkey] + 100 * data.masks[d][akey] for d in datasets)
    z_loc = np.array([data.data_files.slice_loc(d) for d in datasets])
    theta0_iter = (find_theta0(data.zero_angle[d]) for d in datasets)
    origin_iter = (data.zero_angle[d][..., 1] for d in datasets)
    px_size = data.data_files.pixel_size(datasets[0])
    t_iter = tuple((data.data_files.time_interval(d) for d in datasets))

    # z_loc should be increasing with the dataset
    z_loc = -z_loc if all(np.diff(z_loc) < 0) else z_loc

    def to_cylindrical(mask, theta0, origin):
        t, x, y = mask.nonzero()
        means = np.zeros((2,) + mask.shape)

        xx = x - origin[t, 1]
        yy = y - origin[t, 0]
        means[(0, t, x, y)] = np.sqrt(xx ** 2 + yy ** 2) * px_size
        means[(1, t, x, y)] = np.mod(np.arctan2(yy, xx) + theta0[t], 2 * np.pi)

        return masked_reduction(means, mask, axis=(2, 3))

    in_plane = np.array(
        [r_theta for r_theta in map(to_cylindrical, m_iter, theta0_iter, origin_iter)]
    )
    out_plane = (
        np.ones_like(in_plane[:, :1])
        * z_loc[(...,) + (None,) * (len(in_plane[:, :1].shape) - 1)]
    )
    result = np.concatenate((out_plane, in_plane), axis=1).transpose((1, 2, 0, 3, 4))
    return resample_interval(result, t_iter) if resample else result


def displacement(
    data: StrainMapData,
    datasets: Tuple[str, ...],
    nrad: int = 3,
    nang: int = 24,
    lreg: int = 6,
    background: str = "Estimated",
    effective_displacement=True,
    resample=True,
) -> np.ndarray:

    vkey = f"cylindrical - {background}"
    rkey = f"radial x{nrad} - {background}"
    akey = f"angular x{nang} - {background}"
    img_axis = tuple(range(len(data.masks[datasets[0]][vkey].shape)))[-2:]

    cyl_iter = (data.masks[d][vkey] for d in datasets)
    m_iter = (data.masks[d][rkey] + 100 * data.masks[d][akey] for d in datasets)
    t_iter = tuple((data.data_files.time_interval(d) for d in datasets))
    reduced_vel_map = map(partial(masked_reduction, axis=img_axis), cyl_iter, m_iter)

    # Create a mask to define the regions over which to calculate the background
    # for the longitudinal case
    treg = nrad * nang
    lmask = np.ceil(np.arange(1, treg + 1) / treg * lreg).reshape((nang, nrad)).T
    lmask = np.tile(lmask, (data.data_files.frames, 1, 1))

    disp = []
    for r, t in zip(reduced_vel_map, t_iter):
        # Radial and circumferential subtract the average of all slices and frames
        disp.append((r[1:] - r[1:].mean(axis=(1, 2, 3), keepdims=True)) * t)

        # Longitudinal subtract by lreg (=6) angular sectors
        vlong = (
            np.sum(
                np.where(lmask == i, r[0] - r[0][lmask == i].mean(keepdims=True), 0)
                for i in range(1, lreg + 1)
            )
            * t
        )

        # The signs of the in-plane displacement are reversed to be consistent with ECHO
        disp[-1] = np.concatenate((vlong[None, ...], -disp[-1]), axis=0)

    disp = np.asarray(disp)

    result = np.cumsum(disp, axis=2).transpose((1, 2, 0, 3, 4))
    if effective_displacement:
        backward = np.flip(np.flip(disp, axis=2).cumsum(axis=2), axis=2).transpose(
            (1, 2, 0, 3, 4)
        )
        weight = np.arange(0, result.shape[1])[None, :, None, None, None] / (
            result.shape[1] - 1
        )
        result = result * (1 - weight) - backward * weight

    return resample_interval(result, t_iter) if resample else result


def resample_interval(disp: np.ndarray, interval: Tuple[float, ...]) -> np.ndarray:
    """ Re-samples the displacement to the same time interval.

    Total number of frames will increase, in general.

    Args:
        disp: Array of shape [components, frames, z, radial, angular]
        interval: Tuple of len Z with the time interval of each slice.

    Returns:
        The re-sampled displacement.
    """
    frames = disp.shape[1]
    teff = np.arange(0, max(interval) * frames, min(interval))
    t = (np.arange(0, max(interval) * (frames + 1), rr) for rr in interval)
    fdisp = (
        interpolate.interp1d(tt, d[:, : len(tt)], axis=1)
        for tt, d in zip(t, np.moveaxis(np.concatenate([disp, disp], axis=1), 2, 0))
    )
    return np.moveaxis(np.array([f(teff) for f in fdisp]), 0, 2)


def unresample_interval(
    disp: np.ndarray, interval: Tuple[float, ...], frames: int
) -> np.ndarray:
    """ Un-do the re-sample done before to return to the original interval.

    Total number of frames will decrease, in general.

    Args:
        disp: Array of shape [components, frames, z, radial, angular]
        interval: Tuple of len Z with the time interval of each slice.
        frames: Number of frames of the output array.

    Returns:
        The un-resampled displacement.
    """
    teff = np.linspace(0, min(interval) * disp.shape[1], disp.shape[1], endpoint=False)
    t = np.linspace(0, np.array(interval) * frames, frames, endpoint=False).T
    fdisp = (
        interpolate.interp1d(teff, d, axis=1, fill_value="extrapolate")
        for d in np.moveaxis(disp, 2, 0)
    )
    return np.moveaxis(np.array([f(tt) for tt, f in zip(t, fdisp)]), 0, 2)


def reconstruct_strain(
    strain,
    masks,
    datasets,
    resample: bool,
    interval: tuple,
    nrad: int = 3,
    nang: int = 24,
    background: str = "Estimated",
):
    rkey = f"radial x{nrad} - {background}"
    akey = f"angular x{nang} - {background}"
    m_iter = (masks[d][rkey] + 100 * masks[d][akey] for d in datasets)
    frames = masks[datasets[0]][rkey].shape[0]

    strain = unresample_interval(strain, interval, frames) if resample else strain
    return (
        masked_expansion(s, m, axis=(2, 3))
        for s, m in zip(strain.transpose((2, 0, 1, 3, 4)), m_iter)
    )


def calculate_strain(
    data: StrainMapData,
    datasets: Tuple[str, ...],
    callback: Callable = terminal,
    effective_displacement=True,
    resample=True,
    recalculate=False,
):
    """Calculates the strain and updates the Data object with the result."""
    steps = 6.0
    # Do we need to calculate the strain?
    if all([d in data.strain.keys() for d in datasets]) and not recalculate:
        return

    data.strain = defaultdict(dict)
    data.strain_markers = defaultdict(dict)

    # Do we need to regenerate the velocities?
    to_regen = [d for d in datasets if list(data.velocities[d].values())[0] is None]
    if len(to_regen) > 0:
        regenerate(data, to_regen, callback=callback)

    sorted_datasets = tuple(datasets)

    if len(sorted_datasets) < 2:
        callback("Insufficient datasets to calculate strain. At least 2 are needed.")
        return 1

    callback("Preparing dependent variables", 1 / steps)
    disp = displacement(
        data,
        sorted_datasets,
        effective_displacement=effective_displacement,
        resample=resample,
    )

    callback("Preparing independent variables", 2 / steps)
    space = coordinates(data, sorted_datasets, resample=resample)

    callback("Calculating derivatives", 3 / steps)
    reduced_strain = differentiate(disp, space)

    callback("Calculating the regional strains", 4 / steps)
    strain = reconstruct_strain(
        reduced_strain,
        data.masks,
        sorted_datasets,
        resample=resample,
        interval=tuple((data.data_files.time_interval(d) for d in datasets)),
    )
    data.strain = calculate_regional_strain(strain, data.masks, sorted_datasets)

    callback("Calculating markers", 5 / steps)
    for d in datasets:
        labels = [
            s
            for s in data.strain[d].keys()
            if "cylindrical" not in s and "radial" not in s
        ]
        if data.strain_markers.get(d, {}).get(labels[0], None) is None:
            initialise_markers(data, d, labels)
            data.save(*[["strain_markers", d, vel] for vel in labels])

    data.gls = global_longitudinal_strain(
        disp=disp,
        markers=tuple(data.markers[d] for d in datasets),
        times=tuple(data.data_files.time_interval(d) for d in datasets),
        locations=tuple(data.data_files.slice_loc(d) for d in datasets),
        resample=resample,
    )

    callback("Done!", 1)
    return 0


def differentiate(disp, space) -> np.ndarray:
    """ Calculate the strain out of the velocity data.

    Args:
        disp: Array of cumulative displacement with shape (3, frames, z, nrad, nang)
        space: Array of spatial locations with shape (3, frames, z, nrad, nang)

    Returns:
        Array with shape (3, frames, z, nrad, nang) containing the strain.
    """
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
    b = x0[2:] - x0[1:-1]
    c = x0[1:-1] - x0[:-2]
    result[1:-1] = (
        c ** 2 * f0[2:] + (b ** 2 - c ** 2) * f0[1:-1] - b ** 2 * f0[:-2]
    ) / (b * c * (b + c))

    # Boundaries
    if period:
        a = x0[0] - x0[-1] + period
        b = x0[1] - x0[0]
        c = x0[-2] - x0[-1]
        result[0] = (a ** 2 * f0[1] + (b ** 2 - a ** 2) * f0[0] - b ** 2 * f0[-1]) / (
            a * b * (a + b)
        )
        result[-1] = (c ** 2 * f0[0] + (a ** 2 - c ** 2) * f0[-1] - a ** 2 * f0[-2]) / (
            a * c * (a + c)
        )
    else:
        b = x0[1] - x0[0]
        c = x0[-1] - x0[-2]
        result[0] = (f0[1] - f0[0]) / b
        result[-1] = (f0[-1] - f0[-2]) / c

    return np.moveaxis(result, 0, axis)


def calculate_regional_strain(strain, masks, datasets) -> Dict:
    """Calculate the regional strains (1D curves)."""
    from strainmap.models.contour_mask import masked_means

    vkey = "cylindrical - Estimated"
    gkey = "global - Estimated"
    akey = "angular x6 - Estimated"
    rkey = "radial x3 - Estimated"

    result: Dict[Text, Dict[str, np.ndarray]] = defaultdict(dict)
    for d, s in zip(datasets, strain):
        result[d][gkey] = masked_means(s, masks[d][gkey], axes=(2, 3)) * 100
        result[d][akey] = masked_means(s, masks[d][akey], axes=(2, 3)) * 100
        result[d][rkey] = masked_means(s, masks[d][rkey], axes=(2, 3)) * 100
        result[d][vkey] = s

    return result


def update_marker(
    data: StrainMapData,
    dataset: str,
    label: str,
    region: int,
    component: int,
    marker_idx: int,
    position: int,
):
    """Updates the position of an existing marker in the data object.

    If the marker modified is "ES" (marker_idx = 3), then all markers are updated.
    Otherwise just the chosen one is updated.
    """
    value = data.strain[dataset][label][region][component, position]

    data.strain_markers[dataset][label][region][component, marker_idx, :] = [
        position,
        value,
        position * data.data_files.time_interval(dataset),
    ]
    data.save(["strain_markers", dataset, label])


def initialise_markers(data: StrainMapData, dataset: str, str_labels: list):
    """Initialises the markers for all the available strains.

    TODO This function needs to be updated when the proper strain markers location
    TODO is implemented.
    """
    from copy import deepcopy

    data.strain_markers[dataset] = deepcopy(data.markers[dataset])
    for r in list(data.strain_markers[dataset].keys()):
        if r not in data.strain[dataset]:
            del data.strain_markers[dataset][r]
            continue
        else:
            regions = data.strain_markers[dataset][r]
        for j in range(len(regions)):
            for k in range(len(regions[j])):
                for i in range(len(regions[j][k])):
                    position = int(data.strain_markers[dataset][r][j, k, i, 0])
                    data.strain_markers[dataset][r][j, k, i, 1:] = [
                        data.strain[dataset][r][j, k, position],
                        position * data.data_files.time_interval(dataset),
                    ]


def global_longitudinal_strain(
    disp: np.ndarray,
    markers: Tuple[Dict, ...],
    times: Tuple[float, ...],
    locations: Tuple[float, ...],
    resample=True,
):
    """ Calculates the global longitudinal strain by a line fitting of the displacement.

    It takes into account if the data has been resampled, to pick the correct resampled
    frame from the displacement."""

    ldisp = disp[0].mean(axis=(-2, -1))
    tmin = min(times)
    gls = np.zeros(len(locations))
    for i, m in enumerate(markers):
        pos = int(m["global - Estimated"][0, 1, 3, 0])
        corrected = int(round(times[i] / tmin * pos)) if resample else pos
        gls[i] = ldisp[corrected, i]

    return abs(
        np.polynomial.polynomial.Polynomial.fit(locations, gls, 1).deriv().coef[0]
    )
