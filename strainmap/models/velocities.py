from itertools import chain
from typing import Callable, Dict, NamedTuple, Optional, Sequence, Text, Tuple

import numpy as np
import xarray as xr
from skimage.draw import polygon2mask

from ..coordinates import Comp, Region, VelMark
from .strainmap_data_model import StrainMapData
from .writers import terminal


def theta_origin(centroid: xr.DataArray, septum: xr.DataArray):
    """Finds theta0 out of the centroid and septum mid-point"""
    shifted = septum - centroid
    theta0 = np.arctan2(shifted.sel(coord="col"), shifted.sel(coord="row"))
    return theta0


def process_phases(
    raw_phase: xr.DataArray,
    sign_reversal: xr.DataArray,
    swap: bool = False,
    scale: float = 1 / 4096,
) -> xr.DataArray:
    """Prepare the phases for further analysis.

    Several manipulations are done
    - scaling and shifting them to the [-0.5, 0.5] range
    - swap X and Y components, if needed
    - Reverse the sign of one or more components

    Args:
        raw_phase (xr.DataArray): Un-processed phase images. It should have 3 components
            (x, y, z) and any number of frames, rows and columns.
        swap (bool): If the x and y components should be swapped.
        sign_reversal (Sequence[bool]): If the sign of any of the components should be
            reversed.
        scale (float): The scale to apply to the data, which depends on the color depth.

    Returns:
        An DataArray with the phases adjusted by the input parameters.
    """
    phase = raw_phase * scale - 0.5

    if swap:
        phase.loc[{"comp": Comp.X}], phase.loc[{"comp": Comp.Y}] = (
            phase.sel(comp=Comp.Y).copy(),
            phase.sel(comp=Comp.X).copy(),
        )

    phase[...] *= sign_reversal

    return phase


def ring_mask(
    outer: xr.DataArray, inner: xr.DataArray, shape: Tuple[int, int]
) -> np.ndarray:
    """Finds a ring-shaped mask between the outer and inner contours vs frame."""
    out_T = outer.transpose("frame", "point", "coord")
    in_T = inner.transpose("frame", "point", "coord")

    return np.array(
        [
            polygon2mask(shape, out_T.sel(frame=i).data)
            ^ polygon2mask(shape, in_T.sel(frame=i).data)
            for i in out_T.frame
        ]
    )


def global_mask(
    segments: xr.DataArray, shape: Tuple[int, int], mask: xr.DataArray
) -> None:
    """Finds the global mask"""
    mask[...] = ring_mask(
        segments.sel(side="epicardium"), segments.sel(side="endocardium"), shape
    )


def angular_mask(
    centroid: xr.DataArray,
    theta0: xr.DataArray,
    global_mask: xr.DataArray,
    angular_mask: xr.DataArray,
    clockwise: bool = True,
):
    """Calculates the angular mask for the requested angular region."""
    iframe, irow, icol = np.nonzero(global_mask.data)

    crow = centroid.sel(frame=iframe, coord="row").data
    ccol = centroid.sel(frame=iframe, coord="col").data
    th0 = theta0.sel(frame=iframe).data

    theta = np.mod(np.arctan2(icol - ccol, irow - crow) - th0, 2 * np.pi)
    if clockwise:
        theta = 2 * np.pi - theta

    nsegments = angular_mask.region[0].item().value
    steps = np.linspace(0, 2 * np.pi, nsegments + 1)

    for i in range(nsegments):
        idx = (theta > steps[i]) & (theta <= steps[i + 1])
        angular_mask.data[i, iframe[idx], irow[idx], icol[idx]] = True

    return


def radial_mask(
    segments: xr.DataArray, shape: Tuple[int, int], radial_mask: xr.DataArray
):
    """Calculates the angular mask for the requested radial region."""
    nsegments = radial_mask.region[0].item().value
    diff = segments.sel(side="epicardium") - segments.sel(side="endocardium")
    for i in range(nsegments):
        outer = segments.sel(side="endocardium") + diff * (i + 1) / nsegments
        inner = segments.sel(side="endocardium") + diff * i / nsegments
        radial_mask.data[i] = ring_mask(outer, inner, shape)

    return


def find_masks(
    segments: xr.DataArray,
    centroid: xr.DataArray,
    theta0: xr.DataArray,
    shape: Tuple[int, int],
) -> xr.DataArray:
    """Calculates all the masks defined in the Region class.

    The output mask array is of type boolean and has as many elements in the region
    dimension as the sum of individual regions, ie. 1x global + 6x angular + 24x angular
    + 3x radial. The coordinates of each region is the corresponding Region element,
    so selecting each of them, selects the whole group.
    """
    reg = list(chain.from_iterable(([r] * r.value for r in Region)))

    mask = xr.DataArray(
        np.full((len(reg), len(segments.frame), shape[0], shape[1]), False, dtype=bool),
        dims=["region", "frame", "row", "col"],
        coords={"region": reg, "frame": segments.frame},
    )

    gmask = mask.sel(region=Region.GLOBAL)
    global_mask(segments, shape, gmask)

    for region in (Region.ANGULAR_x6, Region.ANGULAR_x24):
        amask = mask.sel(region=region)
        angular_mask(centroid, theta0, gmask, amask)
        mask.loc[{"region": region}] = amask

    rmask = mask.sel(region=Region.RADIAL_x3)
    radial_mask(segments, shape, rmask)
    mask.loc[{"region": Region.RADIAL_x3}] = rmask

    return mask


def cartesian_to_cylindrical(
    centroid: xr.DataArray,
    theta0: xr.DataArray,
    global_mask: xr.DataArray,
    phase: xr.DataArray,
    clockwise: bool = True,
):
    """Transform the phases from cartesian to cylindrical coordinates.

    The calculation has several steps:
        1- We check the non-zero elements in the global mask. Any further calculation is
            done on those elements.
        2- Calculate the bulk movement of the heart. Only the bulk in the plane should
            be accounted for.
        3- Calculate the phases after the subtraction of the bulk.
        4- Calculate theta for of each of the nonzero elements.
        5- Build the rotation matrix
        6- Calculate the sum product of the rotation and the cartesian coordinates to
            get the cylindrical ones.

    Args:
        centroid (xr.DataArray): Centroid of the masks.
        theta0 (xr.DataArray): Origin of angles.
        global_mask (xr.DataArray): Global masks.
        phase (xr.DataArray): The three components of the phase.
        clockwise (optional, bool): The direction of the theta variable.

    Returns:
        DataArray with the cylindrical coordinates.
    """
    iframe, irow, icol = np.nonzero(global_mask.data)
    bulk = xr.where(global_mask, phase, np.nan).mean(dim=("row", "col"))
    bulk.loc[{"comp": Comp.Z}] = 0
    cartesian = (
        (phase - bulk)
        .transpose("frame", "row", "col", "comp")
        .data[iframe, irow, icol, :]
    )
    cylindrical = np.zeros_like(phase.data)

    crow = centroid.sel(frame=iframe, coord="row").data
    ccol = centroid.sel(frame=iframe, coord="col").data
    th0 = theta0.sel(frame=iframe).data

    theta = np.mod(np.arctan2(icol - ccol, irow - crow) - th0, 2 * np.pi)
    if clockwise:
        theta = 2 * np.pi - theta

    cylindrical[:, iframe, irow, icol] = (
        cylindrical_rotation_matrix(theta) @ cartesian[..., None]
    ).T

    return xr.DataArray(
        cylindrical, dims=phase.dims, coords=phase.coords
    ).assign_coords(comp=[Comp.RAD, Comp.CIRC, Comp.LONG])


def cylindrical_rotation_matrix(theta: np.ndarray) -> np.ndarray:
    """Calculate the rotation matrix based on the given theta.

    If the shape of theta is (M, ...), then the shape of the rotation matrix will
    be (M, ..., 3, 3).
    """
    cos = np.cos(theta)
    sin = np.sin(theta)
    one = np.ones_like(theta)
    zer = np.zeros_like(theta)

    return np.stack(
        (
            np.stack((-cos, sin, zer), axis=-1),
            np.stack((-sin, -cos, zer), axis=-1),
            np.stack((zer, zer, one), axis=-1),
        ),
        axis=-1,
    )


def calculate_velocities(
    data: StrainMapData,
    cine: Text,
    sign_reversal: Tuple[bool, ...] = (1, 1, 1),
    init_markers: bool = True,
):
    """Calculates the velocity of the chosen cine and regions."""
    data.sign_reversal[...] = np.array(sign_reversal)

    # We start by processing the phase images
    swap, signs = data.data_files.phase_encoding(cine)
    raw_phase = data.data_files.images(cine).sel(comp=[Comp.X, Comp.Y, Comp.Z])
    phase = process_phases(raw_phase, data.sign_reversal, swap)

    # Now we calculate all the masks
    shape = raw_phase.sizes["row"], raw_phase.sizes["col"]
    segments = data.segments.sel(cine=cine).drop_vars("cine")
    centroid = data.centroid.sel(cine=cine).drop_vars("cine")
    septum = data.septum.sel(cine=cine).drop_vars("cine")
    theta0 = theta_origin(centroid, septum)
    masks = find_masks(segments, centroid, theta0, shape)

    # Next we calculate the velocities in cylindrical coordinates
    cylindrical = (
        cartesian_to_cylindrical(
            centroid, theta0, masks.sel(region=Region.GLOBAL).drop_vars("region"), phase
        )
        * data.data_files.sensitivity
        * signs
    )

    # We are now ready to calculate the velocities
    velocity = xr.where(masks, cylindrical, np.nan).mean(dim=["row", "col"])

    return velocity


def initialise_markers(velocity: xr.DataArray) -> xr.DataArray:
    result = xr.DataArray(
        np.full((velocity.sizes["comp"], velocity.sizes["region"], 7, 3), np.nan),
        dims=["comp", "region", "marker", "quantity"],
        coords={
            "comp": velocity.comp,
            "region": velocity.region,
            "marker": list(VelMark),
            "quantity": ["frame", "velocity", "time"],
        },
    )

    for m, opt in markers_options.items():
        if opt.fun is None:
            continue

        low = min(opt.low, velocity.sizes["frame"] - 1)
        high = min(opt.high, velocity.sizes["frame"] - 1)
        vel = velocity.sel(frame=slice(low, high), comp=opt.comp)

        idx = opt.fun(vel, dim="frame")
        result.loc[{"marker": m, "quantity": "frame", "comp": opt.comp}] = idx.T + low
        result.loc[{"marker": m, "quantity": "velocity", "comp": opt.comp}] = vel.isel(
            frame=idx
        ).T

    # Search for ES
    opt = markers_options[VelMark.ES]
    pdidx = int(
        result.sel(
            marker=VelMark.PD, quantity="frame", comp=Comp.RAD, region=Region.GLOBAL
        ).item()
    )
    low = min(opt.low, pdidx)
    high = min(opt.high, pdidx)
    if high - low == 0:
        low = 0
    vel = velocity.sel(frame=slice(low, high), comp=opt.comp, region=Region.GLOBAL)
    dvel = vel.differentiate("frame")
    idx = dvel.argmin(dim="frame")
    if idx == pdidx or xr.ufuncs.fabs(vel).isel(frame=idx).item() < -2.5:
        idx = vel.argmin(dim="frame")

    result.loc[{"marker": VelMark.ES, "quantity": "frame", "comp": opt.comp}] = (
        idx.T + low
    )
    result.loc[
        {"marker": VelMark.ES, "quantity": "velocity", "comp": opt.comp}
    ] = vel.isel(frame=idx).T

    return result


class _MSearch(NamedTuple):
    low: int = 0
    high: int = 50
    fun: Optional[Callable] = None
    comp: Sequence[Comp] = ()


markers_options: Dict[VelMark, _MSearch] = {
    VelMark.PS: _MSearch(1, 15, xr.DataArray.argmax, [Comp.RAD, Comp.LONG]),
    VelMark.PD: _MSearch(15, 35, xr.DataArray.argmin, [Comp.RAD, Comp.LONG]),
    VelMark.PAS: _MSearch(35, 47, xr.DataArray.argmin, [Comp.RAD, Comp.LONG]),
    VelMark.ES: _MSearch(14, 21, None, [Comp.RAD]),
    VelMark.PC1: _MSearch(1, 5, xr.DataArray.argmin, [Comp.CIRC]),
    VelMark.PC2: _MSearch(6, 12, xr.DataArray.argmax, [Comp.CIRC]),
    VelMark.PC3: _MSearch(0, 0, None, [Comp.CIRC]),
}


def px_velocity_curves(
    data: StrainMapData, cine: str, nrad: int = 3, nang: int = 24
) -> np.ndarray:
    """ TODO: Remove in the final version. """
    from .strain import masked_reduction

    vkey = f"cylindrical"
    rkey = f"radial x{nrad}"
    akey = f"angular x{nang}"
    img_axis = tuple(range(len(data.masks[cine][vkey].shape)))[-2:]

    cyl = data.masks[cine][vkey]
    m = data.masks[cine][rkey] + 100 * data.masks[cine][akey]
    r = masked_reduction(cyl, m, axis=img_axis)

    return r - r.mean(axis=(1, 2, 3), keepdims=True)


def regenerate(data, cines, callback: Callable = terminal):
    """Regenerate velocities and masks information after loading from h5 file."""
    for i, d in enumerate(cines):
        callback(
            f"Regenerating existing velocities {i+1}/{len(cines)}.", i / len(cines)
        )
        vels = data.velocities[d]
        regions: Dict = {}
        for k, v in vels.items():
            if k == "global":
                regions["global_velocity"] = True
            else:
                rtype, num = k.split(" x")
                if f"{rtype}_regions" in regions:
                    regions[f"{rtype}_regions"].append(int(num))
                else:
                    regions[f"{rtype}_regions"] = [int(num)]

        calculate_velocities(
            data=data,
            cine=d,
            sign_reversal=data.sign_reversal,
            init_markers=False,
            **regions,
        )
    callback(f"Regeneration complete!", 1)


def marker(comp, low=1, high=49, maximum=True):
    """Finds the index and value of the marker position within the given range."""
    low = min(low, len(comp) - 1)
    high = min(high, len(comp) - 1)

    idx = (
        np.argmax(comp[low : high + 1]) if maximum else np.argmin(comp[low : high + 1])
    ) + low
    return idx, comp[idx], 0


def marker_es(comp, pd, low=14, high=21):
    """Finds the default position of the ES marker."""
    low = min(low, int(pd[0]) - 1)
    high = min(high, int(pd[0]) - 1)

    if high - low == 0:
        low = 0

    grad = np.diff(comp[low : high + 1])
    idx = np.argmin(abs(grad)) + low

    if idx == pd[0] or comp[idx] < -2.5:
        idx = np.argmin(abs(comp[low : high + 1])) + low

    return idx, comp[idx], 0


def marker_pc3(comp, es):
    """Finds the default position of the PC3 marker."""
    low = int(es[0])
    high = int(min(low + 10, len(comp) - 1))

    pos = np.max(comp[low : high + 1])
    neg = np.min(comp[low : high + 1])

    idx = (
        np.argmax(comp[low : high + 1])
        if pos > abs(neg)
        else np.argmin(comp[low : high + 1])
    ) + low

    value = comp[idx]
    if abs(value) < 0.5:
        idx = 0
        value = np.nan

    return idx, value, 0


def normalised_times(markers: np.ndarray, frames: int) -> np.ndarray:
    """Calculates the normalised values for the marker times.

    The timings of the peaks (PS, PD and PAS) within the cardiac cycle are heart rate
    dependent. To reduce variability due to changes in heart rate, we normalise the
    cardiac cycle to a fixed length of 1000ms (ie a heart rate of 60bpm). As the heart
    rate changes, the length of systole remains relatively fixed while the length of
    diastole changes substantially – so we do this in two stages.

    Our normalised curves  have a systolic length of 350 ms and a diastolic length of
    650ms.
    """
    es = markers[1, 3, 0]
    markers[:, :, 2] = np.where(
        markers[:, :, 0] <= es,
        smaller_than_es(markers[:, :, 0], es),
        larger_than_es(markers[:, :, 0], es, frames),
    )

    return markers


def smaller_than_es(x, es, syst=350):
    """Normalization for times smaller than ES."""
    return x / es * syst


def larger_than_es(x, es, frames, syst=350, dias=650):
    """Normalization for times larger than ES."""
    return syst + (x - es) / (frames - es) * dias


def _markers_positions(
    velocity: np.ndarray, es: Optional[np.ndarray] = None
) -> np.ndarray:
    """Find the position of the markers for the chosen cine and velocity.

    The default positions for the markers are:

    Longitudinal and radial:
        PS = maximum in frames 1 - 15
        PD = minimum in frames 15 - 30
        PAS = minimum in frames 31 - 45

    Circumferential:
        PC1 = minimum in frames 1 – 5
        PC2 = maximum in frames 6 – 12
        PC3 = largest peak (negative OR positive) between ES and ES+10. However if the
            magnitude of the detected peak is <0.5cm/s, then don’t display a PC3 marker
            or PC3 values.

    The default position for the ES marker in the global radial plot is the position of
    the first small negative peak about 1/3 of the way through the cardiac cycle. If
    for any reason that peak doesn’t exist, then we would pick up PD by mistake.
    We can avoid this by saying that if the peak detected is <-2cm/s, use the x axis
    cross-over point as the default marker position instead.

    For the regional plots, the default ES position is the position of the global ES.

    Returns:
        Array with dimensions [component, 4, 3], where axes 1 represents the marker name
        (see above) and axes 2 represents the marker information (index, value,
        normalised time). Only component 1 has a 4th marker.
    """
    markers_lbl = (("PS", "PD", "PAS"), ("PS", "PD", "PAS"), ("PC1", "PC2"))
    markers = np.zeros((3, 4, 3), dtype=float)

    for i in range(3):
        for j, key in enumerate(markers_lbl[i]):
            markers[i, j] = marker(velocity[i], **markers_options[key])

    markers[1, 3] = (
        marker_es(velocity[1], markers[1, 1], **markers_options["ES"])
        if es is None
        else (int(es[0]), velocity[1, int(es[0])], 0)
    )
    markers[2, 2] = marker_pc3(velocity[2], markers[1, 3])

    return normalised_times(markers, len(velocity[0]))


def markers_positions(velocity: np.ndarray, es: Optional[np.ndarray] = None):
    """Find the position of the markers for the chosen cine and velocity."""
    markers = []
    for i in range(velocity.shape[0]):
        markers.append(_markers_positions(velocity[i], es))

    return np.array(markers)


def _update_marker(
    velocities: np.ndarray,
    markers: np.ndarray,
    component: int,
    marker_idx: int,
    position: int,
):
    """Updates the position of an existing marker in the data object.

    If the marker modified is "ES" (marker_idx = 3), then all markers are updated.
    Otherwise just the chosen one is updated.
    """
    value = velocities[component, position]
    frames = len(velocities[component])

    if marker_idx == 3:
        markers[1, 3, 0] = position
        markers[1, 3, 1] = value
        markers = normalised_times(markers, frames)

    else:
        es = markers[1, 3, 0]
        markers[component, marker_idx, :] = [
            position,
            value,
            smaller_than_es(position, es)
            if position <= es
            else larger_than_es(position, es, frames),
        ]

    return markers


def update_marker(
    data: StrainMapData,
    cine: str,
    vel_label: str,
    region: int,
    component: int,
    marker_idx: int,
    position: int,
):
    """Updates the position of an existing marker in the data object.

    If the marker modified is "ES" (marker_idx = 3), then all markers are updated.
    Otherwise just the chosen one is updated.
    """
    if marker_idx == 3:
        for l in data.markers[cine]:
            for r in range(len(data.markers[cine][l])):
                data.markers[cine][l][r] = _update_marker(
                    data.velocities[cine][l][r],
                    data.markers[cine][l][r],
                    component,
                    marker_idx,
                    position,
                )
    else:
        data.markers[cine][vel_label][region] = _update_marker(
            data.velocities[cine][vel_label][region],
            data.markers[cine][vel_label][region],
            component,
            marker_idx,
            position,
        )
    data.save(["markers", cine, vel_label])


# def initialise_markers(data: StrainMapData, cine: str, vel_labels: list):
#     """Initialises the markers for all the available velocities."""
#     data.markers[cine]["global"] = markers_positions(data.velocities[cine]["global"])
#
#     rvel = (key for key in vel_labels if key != "global")
#     for vel_label in rvel:
#         data.markers[cine][vel_label] = markers_positions(
#             data.velocities[cine][vel_label], data.markers[cine]["global"][0][1, 3],
#         )
#     return data
