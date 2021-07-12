from collections import defaultdict
from itertools import product
from typing import Callable, Dict, Optional, Text, Tuple

import numpy as np
import xarray as xr
from scipy import interpolate

from strainmap.models.strainmap_data_model import StrainMapData
from strainmap.models.writers import terminal
from strainmap.coordinates import Region


def calculate_strain(
    data: StrainMapData,
    cines: Tuple[str, ...],
    callback: Callable = terminal,
    effective_displacement=True,
    resample=True,
    recalculate=False,
    timeshift: Optional[float] = None,
):
    """Calculates the strain and updates the Data object with the result."""
    steps = 7.0

    if len(cines) < 2:
        callback("Insufficient cines to calculate strain. At least 2 are needed.")
        return 1

    if timeshift is not None:
        data.timeshift = timeshift
        data.save(["timeshift"])

    callback("Calculating displacement", 1 / steps)
    disp = displacement(
        data, cines, effective_displacement=effective_displacement, resample=resample
    )

    callback("Calculating coordinates", 2 / steps)
    space = coordinates(data, cines, resample=resample)

    callback("Calculating twist", 3 / steps)
    # data.twist = twist(data, cines)

    callback("Calculating strain", 4 / steps)
    reduced_strain = differentiate(disp, space)

    callback("Calculating the regional strains", 5 / steps)
    data.strain = calculate_regional_strain(
        reduced_strain,
        data.masks,
        cines,
        resample=resample,
        interval=tuple((data.data_files.time_interval(d) for d in cines)),
        timeshift=data.timeshift,
    )

    callback("Calculating markers", 6 / steps)
    for d in cines:
        labels = [
            s
            for s in data.strain[d].keys()
            if "cylindrical" not in s and "radial" not in s
        ]
        if data.strain_markers.get(d, {}).get(labels[0], None) is None or recalculate:
            initialise_markers(data, d, labels)
            data.save(*[["strain_markers", d, vel] for vel in labels])

    data.gls = global_longitudinal_strain(
        disp=disp,
        markers=tuple(data.strain_markers[d] for d in cines),
        times=tuple(data.data_files.time_interval(d) for d in cines),
        locations=tuple(data.data_files.slice_loc(d) for d in cines),
        resample=resample,
    )

    callback("Done!", 1)
    return 0


def _masked_reduction(
    data: xr.DataArray, radial: xr.DataArray, angular: xr.DataArray
) -> xr.DataArray:
    """Reduces the size of an array by averaging the regions defined by the masks.

    The radial and angular masks define a collection of regions of interest in 2D space
    all together forming a torus, with Na angular segments and Nr radial segments.
    This means that a large 2D array with data (it can have more dimensions) can be
    reduced to a much smaller and easy to handle (Nr, Na) array, where the value of
    each entry is the mean value of the pixels in the regions defined by both masks.

    The reduced array has the dimensions of the input data with 'row' and 'col' removed
    and 'radius' and 'angle' added to the end. So if data shape is (N0, N1, N2, N3, N4)
    then the reduced array will have shape (N0, N3, N4, Nr, Na).

    Simplified example: If we had 2 radial and 4 angular regions, (8 in total) regions
    in an otherwise 8x8 array, the reduced array will be a 2x4 array, with each element
    the mean value of all the elements in that region:

    Regions:
    [[0, 0, 0, 0, 0, 0, 0, 0],
     [0, 5, 5, 5, 6, 6, 6, 0],
     [0, 5, 1, 1, 2, 2, 6, 0],
     [0, 5, 1, 0, 0, 2, 6, 0],
     [0, 8, 4, 0, 0, 3, 7, 0],
     [0, 8, 4, 4, 3, 3, 7, 0],
     [0, 8, 8, 8, 7, 7, 7, 0],
     [0, 0, 0, 0, 0, 0, 0, 0]]

     Reduced array, with the numbers indicating the region they related to:
     [[1, 2, 3, 4],
      [5, 6, 7, 8]]

    Args:
        data: Large array to reduce. Must contain all the dimensions of the masks
            except 'region'.
        radial: Radial mask. Must contain 'region', 'row' and 'col' dimensions, at
            least.
        angular: Angular mask. Must have same dimensions (and shape) that the radial
            mask.

    Returns:
        A DataArray with reduced size.
    """

    dims = [d for d in data.dims if d not in ("row", "col")]
    nrad = radial.sizes["region"]
    nang = angular.sizes["region"]
    reduced = xr.DataArray(
        np.zeros([data.sizes[d] for d in dims] + [nrad, nang], dtype=data.dtype),
        dims=dims + ["radius", "angle"],
        coords={d: data[d] for d in dims},
    )

    for r, a in product(range(nrad), range(nang)):
        mask = xr.ufuncs.logical_and(radial.isel(region=r), angular.isel(region=a))
        reduced.loc[{"radius": r, "angle": a}] = (
            data.sel(row=mask.row, col=mask.col).where(mask).mean(dim=("row", "col"))
        )

    return reduced


def _masked_expansion(
    data: xr.DataArray,
    radial: xr.DataArray,
    angular: xr.DataArray,
    nrow: int = 512,
    ncol: int = 512,
) -> xr.DataArray:
    """Transforms a reduced array into a full array with the same shape as the masks.

    This function, partially opposite to `masked_reduction`, will recover a full size
    array with the same shape as the original one and with the masked elements equal to
    the corresponding entries of the reduced array. All other elements are nan.

    Args:
        data: Reduced array to expand. Should include dimensions 'radius' and 'angle'
            as well as any dimension of the masks other than 'region', 'row' and
            'column'.
        radial: Radial mask. Must contain 'region', 'row' and 'col' dimensions, at
            least.
        angular: Angular mask. Must have same dimensions (and shape) that the radial
            mask.
        nrow: Size of 'row' dimension in the expanded array iof larger than the one of
            the radial array.
        ncol: Size of 'col' dimension in the expanded array iof larger than the one of
            the radial array.

    Returns:
        The expanded array.
    """
    dims = [d for d in data.dims if d not in ("radius", "angle")]
    nrad = data.sizes["radius"]
    nang = data.sizes["angle"]
    expanded = xr.DataArray(
        np.full(
            [data.sizes[d] for d in dims] + [radial.sizes["row"], radial.sizes["col"]],
            np.nan,
            dtype=data.dtype,
        ),
        dims=dims + ["row", "col"],
        coords={**{d: data[d] for d in dims}, "row": radial.row, "col": radial.col},
    )

    for r, a in product(range(nrad), range(nang)):
        mask = xr.ufuncs.logical_and(radial.sel(region=r), angular.sel(region=a))
        expanded = xr.where(mask, data.sel(radius=r, angle=a), expanded)

    # Now we ensure that the dimensions are in the correct order
    expanded = expanded.transpose(*tuple(dims + ["row", "col"]))

    # Finally, we populate an array with the correct number of rows and cols, if needed
    if nrow != radial.sizes["row"] or ncol != radial.sizes["col"]:
        output = xr.DataArray(
            np.full(
                [data.sizes[d] for d in dims] + [nrow, ncol], np.nan, dtype=data.dtype
            ),
            dims=dims + ["row", "col"],
            coords={
                **{d: data[d] for d in dims},
                "row": np.arange(0, nrow),
                "col": np.arange(0, ncol),
            },
        )
        output.loc[{"row": expanded.row, "col": expanded.col}] = expanded

        return output
    else:
        return expanded


def coordinates(
    data: StrainMapData,
    datasets: Tuple[str, ...],
    nrad: int = 3,
    nang: int = 24,
    resample=True,
    use_frame_zero=True,
) -> np.ndarray:

    rkey = f"radial x{nrad}"
    akey = f"angular x{nang}"

    m_iter = (data.masks[d][rkey] + 100 * data.masks[d][akey] for d in datasets)
    z_loc = np.array([data.data_files.cine_loc(d) for d in datasets])
    # theta0_iter = (find_theta0(data.septum[d]) for d in datasets)
    theta0_iter = ()
    origin_iter = (data.septum[d][..., 1] for d in datasets)
    px_size = data.data_files.pixel_size(datasets[0])
    t_iter = tuple((data.data_files.time_interval(d) for d in datasets))
    ts = data.timeshift

    # z_loc should be increasing with the dataset
    z_loc = -z_loc if all(np.diff(z_loc) < 0) else z_loc

    def to_cylindrical(mask, theta0, origin):
        t, x, y = mask.nonzero()
        means = np.zeros((2,) + mask.shape)

        xx = x - origin[t, 1]
        yy = y - origin[t, 0]
        means[(0, t, x, y)] = np.sqrt(xx ** 2 + yy ** 2) * px_size
        means[(1, t, x, y)] = np.mod(np.arctan2(yy, xx) + theta0[t], 2 * np.pi)

        return _masked_reduction(means, mask, axis=(2, 3))

    in_plane = np.array(
        [r_theta for r_theta in map(to_cylindrical, m_iter, theta0_iter, origin_iter)]
    )
    out_plane = (
        np.ones_like(in_plane[:, :1])
        * z_loc[(...,) + (None,) * (len(in_plane[:, :1].shape) - 1)]
    )
    result = np.concatenate((out_plane, in_plane), axis=1).transpose((1, 2, 0, 3, 4))

    # We shift the coordinates
    for i, t in enumerate(t_iter):
        result[:, :, i, ...] = shift_data(
            result[:, :, i, ...], time_interval=t, timeshift=ts, axis=1
        )
    # We pick just the first element, representing the pixel locations at time zero.
    # Yeah, this function is an massive overkill and will be changed asap.
    if use_frame_zero:
        result[...] = result[:, :1, ...]

    return resample_interval(result, t_iter) if resample else result


def displacement(
    data: StrainMapData,
    cines: Tuple[str, ...],
    nrad: int = 3,
    nang: int = 24,
    lreg: int = 6,
    effective_displacement=True,
    resample=True,
) -> np.ndarray:

    times = xr.DataArray(
        tuple((data.data_files.time_interval(d) for d in cines)),
        dims=["cine"],
        coords={"cine": data.cylindrical.cine},
    )

    reduced = _masked_reduction(
        data.cylindrical,
        data.masks.sel(region=Region.RADIAL_x3),
        data.masks.sel(region=Region.ANGULAR_x24),
    )

    instant = (reduced - reduced.mean(["frame", "radius", "angle"])) * times
    return instant.cumsum("frame")


# vkey = "cylindrical"
# rkey = f"radial x{nrad}"
# akey = f"angular x{nang}"
# img_axis = tuple(range(len(data.masks[cines[0]][vkey].shape)))[-2:]
#
# cyl_iter = (data.masks[d][vkey] for d in cines)
# m_iter = (data.masks[d][rkey] + 100 * data.masks[d][akey] for d in cines)
# t_iter = tuple((data.data_files.time_interval(d) for d in cines))
# ts = data.timeshift
# reduced_vel_map = map(partial(_masked_reduction, axis=img_axis), cyl_iter, m_iter)
#
# # Create a mask to define the regions over which to calculate the background
# # for the longitudinal case
# treg = nrad * nang
# lmask = np.ceil(np.arange(1, treg + 1) / treg * lreg).reshape((nang, nrad)).T
# lmask = np.tile(lmask, (data.data_files.frames, 1, 1))
#
# disp = []
# for r, t in zip(reduced_vel_map, t_iter):
#     # Radial and circumferential subtract the average of all slices and frames
#     disp.append((r[1:] - r[1:].mean(axis=(1, 2, 3), keepdims=True)) * t)
#
#     # Longitudinal subtract by lreg (=6) angular sectors
#     vlong = (
#         np.sum(
#             np.where(lmask == i, r[0] - r[0][lmask == i].mean(keepdims=True), 0)
#             for i in range(1, lreg + 1)
#         )
#         * t
#     )
#
#     # The signs of the in-plane displacement are reversed to be consistent with ECHO
#     disp[-1] = np.concatenate((vlong[None, ...], -disp[-1]), axis=0)
#
#     # We shift the data to the correct time
#     disp[-1] = shift_data(disp[-1], time_interval=t, timeshift=ts, axis=1)
#
# disp = np.asarray(disp)
#
# result = np.cumsum(disp, axis=2).transpose((1, 2, 0, 3, 4))
# if effective_displacement:
#     backward = np.flip(np.flip(disp, axis=2).cumsum(axis=2), axis=2).transpose(
#         (1, 2, 0, 3, 4)
#     )
#     weight = np.arange(0, result.shape[1])[None, :, None, None, None] / (
#         result.shape[1] - 1
#     )
#     result = result * (1 - weight) - backward * weight
#
# return resample_interval(result, t_iter) if resample else result


def resample_interval(disp: np.ndarray, interval: Tuple[float, ...]) -> np.ndarray:
    """Re-samples the displacement to the same time interval.

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
    """Un-do the re-sample done before to return to the original interval.

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


def differentiate(disp, space) -> np.ndarray:
    """Calculate the strain out of the velocity data.

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
    """Calculate the finite differences of a multidimensional array along an axis.

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


def calculate_regional_strain(
    reduced_strain: np.ndarray,
    masks: dict,
    datasets: tuple,
    resample: bool,
    interval: tuple,
    timeshift: float,
    nrad: int = 3,
    nang: int = 24,
    lreg: int = 6,
):
    """Calculate the regional strains (1D curves)."""
    vkey = "cylindrical"
    gkey = "global"
    akey = "angular x6"
    a24key = "angular x24"
    rkey = "radial x3"

    data_shape = masks[datasets[0]][vkey].shape
    m_iter = (masks[d][rkey] + 100 * masks[d][a24key] for d in datasets)

    strain = (
        unresample_interval(reduced_strain, interval, data_shape[1])
        if resample
        else reduced_strain
    )

    # Mask to define the 6x angular and 3x radial regional masks for the reduced strain
    treg = nrad * nang
    lmask = (
        np.ceil(np.arange(1, treg + 1) / treg * lreg)
        .reshape((nang, nrad))
        .T[None, None, ...]
    )
    rmask = (
        np.arange(1, nrad + 1)[None, None, :, None] * np.ones(nang)[None, None, None, :]
    )

    result: Dict[Text, Dict[str, np.ndarray]] = defaultdict(dict)
    vars = zip(datasets, strain.transpose((2, 0, 1, 3, 4)), m_iter, interval)
    for d, s, m, t in vars:

        # When calculating the regional strains from the reduced strain, we need the
        # superpixel area. This has to be shifted to match the times of the strain.
        rm = shift_data(
            superpixel_area(m, data_shape, axis=(2, 3)), t, timeshift, axis=1
        )

        # Global and regional strains are calculated by modifying the relevant weights
        result[d][gkey] = np.average(s, weights=rm, axis=(2, 3))[None, ...] * 100
        result[d][akey] = np.stack(
            np.average(s, weights=rm * (lmask == i), axis=(2, 3)) * 100
            for i in range(1, lreg + 1)
        )
        result[d][rkey] = np.stack(
            np.average(s, weights=rm * (rmask == i), axis=(2, 3)) * 100
            for i in range(1, nrad + 1)
        )

        # To match the strain with the masks, we shift the strain in the opposite
        # direction
        result[d][vkey] = _masked_expansion(
            shift_data(s, t, -timeshift, axis=1), m, axis=(2, 3)
        )

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


def update_strain_es_marker(data: StrainMapData, dataset: str, **kwargs):
    """Updates the strain markers after the ES marker is updated in the velocities.

    The ES marker set in the global radial velocity affects the location of the strain
    markers, so when it is updated, so must be the strain markers."""

    if data.strain.get(dataset, None) is None:
        return

    for d in data.strain.keys():
        labels = [
            s for s in data.strain[d].keys() if s not in ("cylindrical", "radial")
        ]
        initialise_markers(data, d, labels)
        data.save(*[["strain_markers", d, vel] for vel in labels])


def initialise_markers(data: StrainMapData, dataset: str, str_labels: list):
    """Initialises the markers for all the available strains.

    The positions of the markers are given by:
        - Peak systole strain: max/min systole
        - End systole strain: defined with the global radial velocity.
        - Peak strain: max/min of strain during the whole cardiac cycle

    In a healthy patient, the three markers should be roughly at the same position.
    """
    # The location of the ES marker is shifted by an approximate number of frames
    pos_es = int(
        data.markers[dataset]["global"][0, 1, 3, 0]
        - round(data.timeshift / data.data_files.time_interval(dataset))
    )

    # Loop over the region types (global, angular, etc)
    for r in data.strain[dataset].keys():
        if r not in str_labels:
            continue
        else:
            regions = data.strain[dataset][r]

        data.strain_markers[dataset][r] = np.zeros((len(regions), 3, 3, 3), dtype=float)

        # Loop over the individual regions (1 global region, 6 angular regions, etc.)
        for j in range(len(regions)):

            # Loop over the components: longitudinal, radial and circumferential
            extreme = (np.argmin, np.argmax, np.argmin)
            for k in range(len(regions[j])):
                s = data.strain[dataset][r][j, k]
                maxid = len(s) - 1
                # Frame location for peak systole strain, end systole and peak strain
                loc = (
                    min(extreme[k](s[: pos_es + 1]), maxid),
                    min(pos_es, maxid),
                    min(extreme[k](s), maxid),
                )

                # Loop over the markers
                for i, pos in enumerate(loc):
                    data.strain_markers[dataset][r][j, k, i, :] = [
                        pos,
                        s[pos],
                        pos * data.data_files.time_interval(dataset),
                    ]


def global_longitudinal_strain(
    disp: np.ndarray,
    markers: Tuple[Dict, ...],
    times: Tuple[float, ...],
    locations: Tuple[float, ...],
    resample=True,
):
    """Calculates the global longitudinal strain by a line fitting of the displacement.

    It takes into account if the data has been resampled, to pick the correct resampled
    frame from the displacement."""

    ldisp = disp[0].mean(axis=(-2, -1))
    tmin = min(times)
    gls = np.zeros((len(locations), 3))
    for i, m in enumerate(markers):

        # Loop over the markers: PS, ES, P
        # The markers of interest are those corresponding to the longitudinal component
        for j in range(gls.shape[1]):
            pos = int(m["global"][0, 0, j, 0])
            corrected = int(round(times[i] / tmin * pos)) if resample else pos
            gls[i, j] = ldisp[corrected, i]

    return abs(np.polynomial.polynomial.polyfit(locations, gls, 1)[1])


# def twist(
#     data: StrainMapData, datasets: Tuple[str, ...], nrad: int = 3, nang: int = 24
# ) -> LabelledArray:

#     vkey = "cylindrical"
#     rkey = f"radial x{nrad}"
#     akey = f"angular x{nang}"
#     img_axis = tuple(range(len(data.masks[datasets[0]][vkey].shape)))[-2:]

#     cyl_iter = (data.masks[d][vkey] for d in datasets)
#     m_iter = (data.masks[d][rkey] + 100 * data.masks[d][akey] for d in datasets)
#     reduced_vel_map = map(partial(masked_reduction, axis=img_axis), cyl_iter, m_iter)
#     radius = coordinates(data, datasets,resample=False, use_frame_zero=False)[1].mean(
#         axis=(2, 3)
#     )

#     vels = (
#         np.array([v[2].mean(axis=(1, 2)) - v[2].mean() for v in reduced_vel_map])
#         / radius.T
#     )

#     return LabelledArray(
#         dims=["dataset", "frame", "item"],
#         coords={"dataset": datasets, "item": ["angular_velocity", "radius"]},
#         values=np.stack((vels, radius.T), axis=-1),
#     )


def shift_data(
    data: np.ndarray, time_interval: float, timeshift: float, axis: int = 0
) -> np.ndarray:
    """Interpolates the data to account for a timeshift correction."""
    time = np.arange(-1, data.shape[axis] + 1)
    d = np.moveaxis(data, axis, 0)
    d = np.concatenate([d[-1:], d, d[:1]], axis=0)

    shift_frames = int(round(timeshift / time_interval))
    remainder = timeshift - time_interval * shift_frames
    new_time = np.arange(data.shape[axis]) + remainder
    new_data = np.roll(
        interpolate.interp1d(time, d, axis=0)(new_time), -shift_frames, axis=0
    )
    return np.moveaxis(new_data, 0, axis)


def superpixel_area(masks: np.ndarray, data_shape: tuple, axis: tuple) -> np.ndarray:
    from functools import reduce

    assert data_shape[-len(masks.shape) :] == masks.shape

    mask_max = masks.max()
    nrad, nang = mask_max % 100, mask_max // 100
    nz = np.nonzero(masks)
    xmin, xmax, ymin, ymax = (
        nz[-2].min(),
        nz[-2].max() + 1,
        nz[-1].min(),
        nz[-1].max() + 1,
    )
    smasks = masks[..., xmin : xmax + 1, ymin : ymax + 1]

    shape = [s for i, s in enumerate(data_shape) if i not in axis] + [nrad, nang]
    reduced = np.zeros(shape, dtype=int)

    tile_shape = (
        (data_shape[0],) + (1,) * len(masks.shape)
        if data_shape != masks.shape
        else (1,) * len(masks.shape)
    )

    def reduction(red, idx):
        elements = tuple([...] + [k - 1 for k in idx])
        i = idx[0] + 100 * idx[1]
        red[elements] = np.tile(smasks == i, tile_shape).sum(axis=axis).data
        return red

    return reduce(reduction, product(range(1, nrad + 1), range(1, nang + 1)), reduced)
