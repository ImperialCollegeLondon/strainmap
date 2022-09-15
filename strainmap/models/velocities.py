import logging
import time
from itertools import chain
from typing import Callable, Dict, NamedTuple, Optional, Sequence, Text, Tuple

import numpy as np
import xarray as xr
from skimage.draw import polygon2mask

from ..coordinates import Comp, Mark, Region
from .strainmap_data_model import StrainMapData
from .transformations import masked_reduction, theta_origin


class _MSearch(NamedTuple):
    low: int = 0
    high: int = 50
    fun: Optional[Callable] = None
    comp: Sequence[str] = ()


MARKERS_OPTIONS: Dict[Mark, _MSearch] = {
    Mark.PS: _MSearch(1, 15, xr.DataArray.argmax, [Comp.RAD.name, Comp.LONG.name]),
    Mark.PD: _MSearch(15, 35, xr.DataArray.argmin, [Comp.RAD.name, Comp.LONG.name]),
    Mark.PAS: _MSearch(35, 47, xr.DataArray.argmin, [Comp.RAD.name, Comp.LONG.name]),
    Mark.ES: _MSearch(14, 21, None, [Comp.RAD.name]),
    Mark.PC1: _MSearch(1, 5, xr.DataArray.argmin, [Comp.CIRC.name]),
    Mark.PC2: _MSearch(6, 12, xr.DataArray.argmax, [Comp.CIRC.name]),
    Mark.PC3: _MSearch(0, 0, None, [Comp.CIRC.name]),
}


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
        phase.loc[{"comp": Comp.X.name}], phase.loc[{"comp": Comp.Y.name}] = (
            phase.sel(comp=Comp.Y.name).copy(),
            phase.sel(comp=Comp.X.name).copy(),
        )

    return phase * sign_reversal


def ring_mask(outer: np.ndarray, inner: np.ndarray) -> Sequence[np.ndarray]:
    """Finds a ring-shaped mask between the outer and inner contours vs frame."""
    out_T = outer[..., ::-1]
    in_T = inner[..., ::-1]
    maxi = int(round(outer.max())) + 1
    shape = (maxi, maxi)

    return np.nonzero(
        [
            polygon2mask(shape, out_T[i]) ^ polygon2mask(shape, in_T[i])
            for i in range(out_T.shape[0])
        ]
    )


def global_mask(segments: xr.DataArray) -> np.ndarray:
    """Finds the indices of the global mask.

    The output array has shape (4, n), with n the number of non-zero elements of the
    mask, and the 4 rows representing, respectively: iregion, frame, row and col.
    """
    seg = segments.transpose("side", "frame", "point", "coord")
    values = ring_mask(
        seg.sel(side="epicardium").data, seg.sel(side="endocardium").data
    )
    return np.array([np.zeros_like(values[0]), *values])


def angular_mask(
    centroid: xr.DataArray,
    theta0: xr.DataArray,
    non_zero_data: np.ndarray,
    regions: int,
    start_index: int = 0,
    clockwise: bool = True,
) -> np.ndarray:
    """Calculates the angular mask for the requested angular region."""
    _, iframe, irow, icol = non_zero_data

    crow = centroid.sel(frame=iframe, coord="row").data
    ccol = centroid.sel(frame=iframe, coord="col").data
    th0 = theta0.sel(frame=iframe).data

    theta = np.mod(np.arctan2(icol - ccol, irow - crow) - th0, 2 * np.pi)
    if clockwise:
        theta = 2 * np.pi - theta

    steps = np.linspace(0, 2 * np.pi, regions + 1)

    amask = []
    for i in range(regions):
        idx = (theta > steps[i]) & (theta <= steps[i + 1])
        ifr, ir, ic = iframe[idx], irow[idx], icol[idx]
        amask.append(np.array([np.full_like(ifr, start_index + i), ifr, ir, ic]))

    return np.concatenate(amask, axis=1)


def radial_mask(
    segments: xr.DataArray, regions: int, start_index: int = 0
) -> np.ndarray:
    """Calculates the angular mask for the requested radial region."""
    seg = segments.transpose("side", "frame", "point", "coord")
    diff = seg.sel(side="epicardium") - seg.sel(side="endocardium")
    rmask = []
    for i in range(regions):
        outer = seg.sel(side="endocardium") + diff * (i + 1) / regions
        inner = seg.sel(side="endocardium") + diff * i / regions
        values = ring_mask(outer.data, inner.data)
        rmask.append(np.array([np.full_like(values[0], start_index + i), *values]))

    return np.concatenate(rmask, axis=1)


def find_masks(
    segments: xr.DataArray, centroid: xr.DataArray, theta0: xr.DataArray
) -> xr.DataArray:
    """Calculates all the masks defined in the Region class.

    The output mask array is of type boolean and has as many elements in the region
    dimension as the sum of individual regions, ie. 1x global + 6x angular + 24x angular
    + 3x radial. The coordinates of each region is the corresponding Region element,
    so selecting each of them, selects the whole group.
    """
    reg = list(chain.from_iterable(([r.name] * r.value for r in Region)))
    masks: Dict[Region, np.ndarray] = dict()

    masks[Region.GLOBAL] = global_mask(segments)

    start_index = masks[Region.GLOBAL][0].max() + 1
    for region in (Region.ANGULAR_x6, Region.ANGULAR_x24):
        masks[region] = angular_mask(
            centroid=centroid,
            theta0=theta0,
            non_zero_data=masks[Region.GLOBAL],
            regions=region.value,
            start_index=start_index,
        )
        start_index = masks[region][0].max() + 1

    masks[Region.RADIAL_x3] = radial_mask(
        segments=segments, regions=Region.RADIAL_x3.value, start_index=start_index
    )

    coords = np.concatenate(list(masks.values()), axis=1)
    rows = np.arange(min(coords[2]), max(coords[2]) + 0.5).astype(int)
    cols = np.arange(min(coords[3]), max(coords[3]) + 0.5).astype(int)
    shape = len(reg), segments.sizes["frame"], len(rows), len(cols)

    coords[2] = coords[2] - min(rows)
    coords[3] = coords[3] - min(cols)
    data = np.full(shape, fill_value=False)
    data[tuple(coords)] = True
    return xr.DataArray(
        data,
        dims=["region", "frame", "row", "col"],
        coords={"region": reg, "frame": segments.frame, "row": rows, "col": cols},
    )


def cartesian_to_cylindrical(
    centroid: xr.DataArray,
    global_mask: xr.DataArray,
    phase: xr.DataArray,
    clockwise: bool = False,
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
        global_mask (xr.DataArray): Global masks.
        phase (xr.DataArray): The three components of the phase.
        clockwise (optional, bool): The direction of the theta variable.

    Returns:
        DataArray with the cylindrical coordinates.
    """
    iframe, irow, icol = np.nonzero(global_mask.data)
    p = phase.sel(row=global_mask.row, col=global_mask.col)
    bulk = xr.where(global_mask, p, np.nan).mean(dim=("row", "col"))
    bulk.loc[{"comp": Comp.Z.name}] = 0
    cartesian = (
        (p - bulk).transpose("frame", "row", "col", "comp").data[iframe, irow, icol, :]
    )

    crow = centroid.sel(frame=iframe, coord="row").data
    ccol = centroid.sel(frame=iframe, coord="col").data

    row = global_mask.row[irow].data - crow
    col = global_mask.col[icol].data - ccol

    theta = np.arctan2(col, row)
    if clockwise:
        theta = 2 * np.pi - theta

    coords = np.concatenate(
        [
            np.stack([np.full_like(iframe, i), iframe, irow, icol], axis=0)
            for i in range(phase.sizes["comp"])
        ],
        axis=1,
    )
    data = np.full(p.shape, fill_value=0.0)
    data[tuple(coords)] = (
        cylindrical_rotation_matrix(theta) @ cartesian[..., None]
    ).T.flatten()

    return xr.DataArray(
        data,
        dims=p.dims,
        coords={
            "comp": [Comp.CIRC.name, Comp.RAD.name, Comp.LONG.name],
            "frame": global_mask.frame,
            "row": global_mask.row,
            "col": global_mask.col,
        },
    )


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
    sign_reversal: Tuple[int, ...] = (1, 1, 1),
    update_velocities=False,
) -> None:
    """Calculates the velocity of the chosen cine and regions.

    Args:
        data: StrainMap data object with all the information
        cine: The cine to calculate the velocities for
        sign_reversal: Tuple containing the sign change of the phase components.
        update_velocities: If this is an update of the existing velocities, in which
            case the masks don't need to be recalculated.

    Returns:
        None
    """
    data.sign_reversal[...] = np.array(sign_reversal)

    # We start by processing the phase images
    start = time.time()
    swap, signs = data.data_files.phase_encoding(cine)
    raw_phase = data.data_files.images(cine).sel(
        comp=[Comp.X.name, Comp.Y.name, Comp.Z.name]
    )
    phase = process_phases(raw_phase, data.sign_reversal, swap)
    lap1 = time.time()
    logging.info(f"Time for phase: {round(lap1 - start, 2)}s")

    # Now we calculate all the masks, but only if it is not an update
    centroid = data.centroid.sel(cine=cine).drop_vars("cine")
    septum = data.septum.sel(cine=cine).drop_vars("cine")
    theta0 = theta_origin(centroid, septum)
    if not update_velocities:
        segments = data.segments.sel(cine=cine).drop_vars("cine")
        masks = find_masks(segments, centroid, theta0)
        data.add_data(cine, masks=masks)
    else:
        masks = data.masks.sel(cine=cine).drop_vars("cine")
    lap2 = time.time()
    logging.info(f"Time for masks: {round(lap2 - lap1, 2)}s")

    # Next we calculate the velocities in cylindrical coordinates
    cylindrical = (
        cartesian_to_cylindrical(
            centroid, masks.sel(region=Region.GLOBAL.name).drop_vars("region"), phase
        )
        * data.data_files.sensitivity
        * signs
    )
    lap3 = time.time()
    logging.info(f"Time for cyl: {round(lap3 - lap2, 2)}s")

    # We are now ready to calculate the velocities
    velocity = _calculate_velocities(masks, cylindrical)
    lap4 = time.time()
    logging.info(f"Time for vel: {round(lap4 - lap3, 2)}s")

    # And the markers
    markers = initialise_markers(velocity)
    lap5 = time.time()
    logging.info(f"Time for marker: {round(lap5 - lap4, 2)}s")

    # Finally, we add all the information to the StrainMap data object
    data.add_data(cine, cylindrical=cylindrical, velocities=velocity, markers=markers)


def _calculate_velocities(
    masks: xr.DataArray, cylindrical: xr.DataArray
) -> xr.DataArray:
    output = xr.where(masks, cylindrical, np.nan).mean(dim=["row", "col"])
    return output - output.mean(dim="frame")


def initialise_markers(velocity: xr.DataArray) -> xr.DataArray:
    """Find the initial position and value of the markers for the chosen velocity.

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

    Args:
        velocity (xr.DataArray): Array of velocities versus frame for all regions
            and components

    Returns:
        xr.DataArray with all markers information.
    """
    result = xr.DataArray(
        np.full((velocity.sizes["region"], velocity.sizes["comp"], 7, 3), np.nan),
        dims=["region", "comp", "marker", "quantity"],
        coords={
            "region": velocity.region,
            "comp": velocity.comp,
            "marker": [m.name for m in Mark if m not in (Mark.PSS, Mark.ESS)],
            "quantity": ["frame", "velocity", "time"],
        },
    )
    # Shortcuts
    pd = Mark.PD.name
    es = Mark.ES.name
    pc3 = Mark.PC3.name
    glr = Region.GLOBAL.name
    rad = Comp.RAD.name

    # Search for normal markers
    for m, opt in MARKERS_OPTIONS.items():
        if m.name in (es, pc3):
            continue
        idx, value = marker_x(velocity, opt)
        result.loc[{"marker": m.name, "quantity": "frame", "comp": opt.comp}] = idx
        result.loc[{"marker": m.name, "quantity": "velocity", "comp": opt.comp}] = value

    # Search for ES
    pd_loc = int(result.sel(marker=pd, quantity="frame", comp=rad, region=glr).item())
    idx, value = marker_es(velocity, MARKERS_OPTIONS[Mark.ES], pd_loc)
    result.loc[{"marker": es, "quantity": "frame"}] = idx
    result.loc[{"marker": es, "quantity": "velocity"}] = value

    # Search for PC3
    es_loc = int(result.sel(marker=es, quantity="frame", comp=rad, region=glr).item())
    opt = MARKERS_OPTIONS[Mark.PC3]
    idx, value = marker_pc3(velocity, opt, es_loc)
    result.loc[{"marker": pc3, "quantity": "frame", "comp": opt.comp}] = idx
    result.loc[{"marker": pc3, "quantity": "velocity", "comp": opt.comp}] = value

    # Calculate the normalised times
    normalise_times(result, velocity.sizes["frame"])

    return result


def update_markers(
    markers: xr.DataArray,
    marker_label: Mark,
    component: Comp,
    region: Region,
    iregion: int,
    location: int,
    velocity_value: float,
    frames: int,
) -> None:
    """

    Args:
        markers (xr.DataArray): DataArray of all markers
        marker_label (Mark): Marker to update
        component (Comp): Component to update
        region (Region): Region to update
        iregion (int): Specific subregion markers to update
        location (int): New location of the marker
        velocity_value (float): New velocity value of the marker
        frames (int): Total number of frames. Needed for time normalization

    Returns:
        None
    """
    loc_dict = {
        "marker": marker_label.name,
        "quantity": "frame",
        "comp": component.name,
        "region": region.name,
    }
    vel_dict = {
        "marker": marker_label.name,
        "quantity": "velocity",
        "comp": component.name,
        "region": region.name,
    }
    if marker_label == Mark.ES:
        markers.loc[{"marker": Mark.ES.name, "quantity": "frame"}] = location
        markers.loc[{"marker": Mark.ES.name, "quantity": "velocity"}] = velocity_value
    elif region == Region.GLOBAL:
        markers.loc[loc_dict] = location
        markers.loc[vel_dict] = velocity_value
    else:
        new_frame = markers.loc[loc_dict]
        new_frame[iregion] = location
        markers.loc[loc_dict] = new_frame
        new_vel = markers.loc[vel_dict]
        new_vel[iregion] = velocity_value
        markers.loc[vel_dict] = new_vel

    normalise_times(markers, frames)


def marker_x(
    velocity: xr.DataArray, options: _MSearch
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Calculates the location and value of normal markers (not ES nor PC3).

    Args:
        velocity (xr.DataArray): Array of velocities versus frame for all regions
            and components
        options (_MSearch): Options to calculate this specific marker

    Returns:
        Tuple of an array with the locations and a second array with the velocity values
        at those locations.
    """
    low = min(options.low, velocity.sizes["frame"])
    high = min(options.high, velocity.sizes["frame"])
    if high - low == 0:
        low = 0

    vel = velocity.sel(frame=slice(low, high), comp=options.comp)
    idx = options.fun(vel, dim="frame")
    return idx + low, vel.isel(frame=idx)


def marker_es(
    velocity: xr.DataArray, options: _MSearch, pd_loc: int
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Calculates the location and value of the ES marker.

    Args:
        velocity (xr.DataArray): Array of velocities versus frame for all regions
            and components
        options (_MSearch): Options to calculate this specific marker
        pd_loc (int): Location of the PD marker.

    Returns:
        Tuple of an array with the locations and a second array with the velocity values
        at those locations.
    """
    low = min(options.low, pd_loc)
    high = min(options.high, pd_loc)
    if high - low == 0:
        low = 0
        high = velocity.sizes["frame"]

    vel = velocity.sel(
        frame=slice(low, high), comp=Comp.RAD.name, region=Region.GLOBAL.name
    )
    idx = abs(vel.differentiate("frame")).argmin(dim="frame")
    if idx + low == pd_loc or vel.isel(frame=idx).item() < -2.5:
        idx = abs(vel).argmin(dim="frame")

    idx = idx.drop_vars(["region", "comp"])

    return idx.T + low, vel.isel(frame=idx).T.drop_vars(["comp"])


def marker_pc3(
    velocity: xr.DataArray, options: _MSearch, es_loc: int
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Calculates the location and value of the PC3 marker.

    Args:
        velocity (xr.DataArray): Array of velocities versus frame for all regions
            and components
        options (_MSearch): Options to calculate this specific marker
        es_loc (int): Location of the ES marker.


    Returns:
        Tuple of an array with the locations and a second array with the velocity values
        at those locations.
    """

    high = min(es_loc + 10, velocity.frame.max().item())

    vel = velocity.sel(frame=slice(es_loc, high + 1), comp=options.comp)
    pos = vel.max(dim="frame")
    neg = vel.min(dim="frame")

    idx = xr.where(pos > abs(neg), vel.argmax(dim="frame"), vel.argmin(dim="frame"))

    return (
        idx + es_loc,
        xr.where(
            abs(vel.isel(frame=idx)) < 0.5,
            np.nan * vel.isel(frame=idx),
            vel.isel(frame=idx),
        ),
    )


def normalise_times(markers: xr.DataArray, frames: int) -> None:
    """Calculates the normalised values for the marker times.

    The timings of the peaks (PS, PD and PAS) within the cardiac cycle are heart rate
    dependent. To reduce variability due to changes in heart rate, we normalise the
    cardiac cycle to a fixed length of 1000ms (ie a heart rate of 60bpm). As the heart
    rate changes, the length of systole remains relatively fixed while the length of
    diastole changes substantially – so we do this in two stages.

    Our normalised curves  have a systolic length of 350 ms and a diastolic length of
    650ms.

    Args:
        markers (xr.DataArray): Arrays with all markers values, where the result will be
            stored
        frames (int): Total number of frames

    Returns:
        None
    """
    es = int(
        markers.sel(
            marker=Mark.ES.name,
            quantity="frame",
            comp=Comp.RAD.name,
            region=Region.GLOBAL.name,
        ).item()
    )
    markers.loc[{"quantity": "time"}] = xr.where(
        markers.sel(quantity="frame") <= es,
        smaller_than_es(markers.sel(quantity="frame"), es),
        larger_than_es(markers.sel(quantity="frame"), es, frames),
    )


def smaller_than_es(x, es, syst=350):
    """Normalization for times smaller than ES."""
    return x / es * syst


def larger_than_es(x, es, frames, syst=350, dias=650):
    """Normalization for times larger than ES."""
    return syst + (x - es) / (frames - es) * dias


def superpixel_velocity_curves(
    cylindrical: xr.DataArray, radial: xr.DataArray, angular: xr.DataArray
) -> xr.DataArray:
    """Reduces the cylindrical velocities to regions and calculates the displacement.

    Some manipulation is also performed to match the format of echo data and adjusts
    the origin of the times.

    Args:
        cylindrical: Velocities in cylindrical coordinates.
        radial: Masks defining the radial regions.
        angular: Mask defining the angular regions.

    Returns:
        Reduced array with the displacement
    """
    reduced = masked_reduction(cylindrical, radial, angular)
    return reduced - reduced.mean(["frame", "radius", "angle"])
