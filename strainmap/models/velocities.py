from itertools import chain
from typing import Callable, Dict, NamedTuple, Optional, Sequence, Text, Tuple

import numpy as np
import xarray as xr
from skimage.draw import polygon2mask

from ..coordinates import Comp, Region, Mark
from .strainmap_data_model import StrainMapData
from .writers import terminal


class _MSearch(NamedTuple):
    low: int = 0
    high: int = 50
    fun: Optional[Callable] = None
    comp: Sequence[Comp] = ()


MARKERS_OPTIONS: Dict[Mark, _MSearch] = {
    Mark.PS: _MSearch(1, 15, xr.DataArray.argmax, [Comp.RAD, Comp.LONG]),
    Mark.PD: _MSearch(15, 35, xr.DataArray.argmin, [Comp.RAD, Comp.LONG]),
    Mark.PAS: _MSearch(35, 47, xr.DataArray.argmin, [Comp.RAD, Comp.LONG]),
    Mark.ES: _MSearch(14, 21, None, [Comp.RAD]),
    Mark.PC1: _MSearch(1, 5, xr.DataArray.argmin, [Comp.CIRC]),
    Mark.PC2: _MSearch(6, 12, xr.DataArray.argmax, [Comp.CIRC]),
    Mark.PC3: _MSearch(0, 0, None, [Comp.CIRC]),
}


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
    out_T = outer.transpose("frame", "point", "coord")[..., ::-1]
    in_T = inner.transpose("frame", "point", "coord")[..., ::-1]

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
        coords={
            "region": reg,
            "frame": segments.frame,
            "row": np.arange(0, shape[0]),
            "col": np.arange(0, shape[1]),
        },
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
    ).drop_vars("cine")

    # We are now ready to calculate the velocities
    velocity = xr.where(masks, cylindrical, np.nan).mean(dim=["row", "col"])
    markers = initialise_markers(velocity)

    # Finally, we add all the information to the StrainMap data object
    data.add_data(
        cine, masks=masks, cylindrical=cylindrical, velocities=velocity, markers=markers
    )
    return


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
            "marker": [m for m in Mark if m not in (Mark.PSS, Mark.ESS)],
            "quantity": ["frame", "velocity", "time"],
        },
    )
    # Shortcuts
    pd = Mark.PD
    es = Mark.ES
    pc3 = Mark.PC3
    glr = Region.GLOBAL
    rad = Comp.RAD

    # Search for normal markers
    for m, opt in MARKERS_OPTIONS.items():
        if m in (es, pc3):
            continue
        idx, value = marker_x(velocity, opt)
        result.loc[{"marker": m, "quantity": "frame", "comp": opt.comp}] = idx
        result.loc[{"marker": m, "quantity": "velocity", "comp": opt.comp}] = value

    # Search for ES
    pd_loc = int(result.sel(marker=pd, quantity="frame", comp=rad, region=glr).item())
    idx, value = marker_es(velocity, MARKERS_OPTIONS[es], pd_loc)
    result.loc[{"marker": es, "quantity": "frame"}] = idx
    result.loc[{"marker": es, "quantity": "velocity"}] = value

    # Search for PC3
    es_loc = int(result.sel(marker=es, quantity="frame", comp=rad, region=glr).item())
    opt = MARKERS_OPTIONS[pc3]
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
        location (int): New location of the marker
        velocity_value (float): New velocity value of the marker
        frames (int): Total number of frames. Needed for time normalization

    Returns:
        None
    """
    if marker_label == Mark.ES:
        markers.loc[{"marker": Mark.ES, "quantity": "frame"}] = location
        markers.loc[{"marker": Mark.ES, "quantity": "velocity"}] = velocity_value
        normalise_times(markers, frames)
    else:
        markers.loc[
            {
                "marker": marker_label,
                "quantity": "frame",
                "comp": component,
                "region": region,
            }
        ] = location
        markers.loc[
            {
                "marker": marker_label,
                "quantity": "velocity",
                "comp": component,
                "region": region,
            }
        ] = velocity_value


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

    vel = velocity.sel(frame=slice(low, high), comp=Comp.RAD, region=Region.GLOBAL)
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
            marker=Mark.ES, quantity="frame", comp=Comp.RAD, region=Region.GLOBAL
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
