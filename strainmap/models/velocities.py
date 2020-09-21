from typing import (
    Dict,
    List,
    Optional,
    Sequence,
    Text,
    Union,
    Callable,
    Tuple,
)
from enum import Enum

import numpy as np
from scipy import ndimage
import xarray as xr
from skimage.draw import polygon2mask
from sparse import COO


from strainmap.models.contour_mask import cylindrical_projection, masked_means

from .contour_mask import Contour, angular_segments, contour_diff, radial_segments
from .strainmap_data_model import StrainMapData
from .writers import terminal


class Region(Enum):
    GLOBAL = "global"
    ANGULAR_x6 = "angular x6"
    ANGULAR_x24 = "angular x24"
    RADIAL_x3 = "radial x3"


def theta_origin(centroid: xr.DataArray, septum: xr.DataArray):
    """Finds theta0 out of the centroid and septum mid-point"""
    shifted = septum - centroid
    theta0 = np.arctan2(shifted.sel(coord="col"), shifted.sel(coord="row"))
    return theta0


def process_phases(
    raw_phase: xr.DataArray,
    swap: bool = False,
    sign_reversal: Sequence[bool] = (False, False, False),
    scale: float = 1 / 4096,
) -> xr.DataArray:
    """Prepare the phases, scaling them and shifting them to the [-0.5, 0.5] range.

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
        phase = phase.assign_coords(comp=["y", "x", "z"])

    for comp, r in zip(phase.comp, sign_reversal):
        phase.loc[{"comp": comp}] *= -1 if r else 1

    return phase


def global_masks_and_origin(outer, inner):
    """Finds the global masks and origin versus time frame."""
    ymin, ymax = int(outer[:, 0].min()), int(outer[:, 0].max()) + 1
    xmin, xmax = int(outer[:, 1].min()), int(outer[:, 1].max()) + 1
    shift = np.array([ymin, xmin])
    masks = np.array(
        [
            contour_diff(c1=o.T, c2=i.T, shape=(ymax - ymin + 1, xmax - xmin + 1)).T
            for o, i in zip(outer - shift[None, :, None], inner - shift[None, :, None])
        ]
    )
    origin = np.array([*map(ndimage.measurements.center_of_mass, masks)])

    return masks, origin, (xmin, xmax, ymin, ymax)


def global_mask(segments: xr.DataArray, shape=Tuple[int, int]) -> xr.DataArray:
    """Finds the global masks and origin versus time frame."""
    data = np.array(
        [
            polygon2mask(
                shape,
                segments.sel(frame=i, side="epicardium")
                .transpose("point", "coord")
                .data,
            )
            ^ polygon2mask(
                shape,
                segments.sel(frame=i, side="endocardium")
                .transpose("point", "coord")
                .data,
            )
            for i in segments.frame
        ]
    )[None, None, ...].astype(np.uint8)

    return xr.DataArray(
        COO.from_numpy(data),
        dims=["cine", "region", "frame", "row", "col"],
        coords={
            "cine": [segments.cine.item()],
            "region": [Region.GLOBAL],
            "frame": segments.frame,
        },
    )


def transform_to_cylindrical(phase: np.ndarray, masks: np.ndarray, origin: np.ndarray):
    """Transform the velocities to cylindrical coordinates.

    It also accounts for the different origin of coordinates for each time step (frame),
    and substracts for bulk movement of the heart in the plane.
    """
    num_frames = phase.shape[1]
    cylindrical = np.zeros_like(phase)
    for i in range(num_frames):
        bulk_velocity = masked_means(phase[:, i, :, :], masks[i], axes=(1, 2))[0]
        bulk_velocity[-1] = 0

        cylindrical[:, i, :, :] = cylindrical_projection(
            phase[:, i, :, :] - bulk_velocity[:, None, None],
            origin[i],
            component_axis=0,
            image_axes=(1, 2),
        )

    return cylindrical


def subtract_bg(velocities: np.ndarray, axis: int = 2):
    """Subtracts the estimated background to the velocities."""
    return velocities - velocities.mean(axis=axis)[..., None]


def velocity_global(cylindrical: np.ndarray, mask: np.ndarray):
    """Calculate the global velocity."""
    label = f"global"
    velocities = {label: masked_means(cylindrical, mask, axes=(2, 3))}
    velocities[label] = subtract_bg(velocities[label])
    masks = {label: mask}
    return velocities, masks


def velocities_angular(
    cylindrical: np.ndarray,
    zero_angle: np.ndarray,
    origin: np.ndarray,
    mask: np.ndarray,
    regions: Sequence[int] = (6,),
):
    """Calculate the angular velocities for angular regions."""
    theta0 = find_theta0(zero_angle)

    velocities: Dict[str, np.ndarray] = dict()
    masks: Dict[str, np.ndarray] = dict()
    for ang in regions:
        label = f"angular x{ang}"
        region_labels = angular_segments(
            nsegments=ang, origin=origin, theta0=theta0, shape=cylindrical.shape[2:]
        ).transpose((2, 0, 1))
        velocities[label] = masked_means(cylindrical, region_labels * mask, axes=(2, 3))
        velocities[label] = subtract_bg(velocities[label])
        masks[label] = region_labels * mask

    return velocities, masks


def velocities_radial(
    cylindrical: np.ndarray,
    segments: Dict[str, np.ndarray],
    origin: np.ndarray,
    shift: np.ndarray,
    mask: np.ndarray,
    regions: Sequence[int] = (3,),
):
    """Calculates the regional velocities for radial regions."""
    outer = segments["epicardium"] - shift[None, ::-1, None]
    inner = segments["endocardium"] - shift[None, ::-1, None]

    velocities: Dict[str, np.ndarray] = dict()
    masks: Dict[str, Union[list, np.ndarray]] = dict()
    for nr in regions:
        label = f"radial x{nr}"
        velocities[label] = np.zeros((nr, cylindrical.shape[0], cylindrical.shape[1]))
        masks[label] = []

        for i in range(cylindrical.shape[1]):
            masks[label].append(
                radial_segments(
                    outer=Contour(outer[i].T, shape=cylindrical.shape[2:]),
                    inner=Contour(inner[i].T, shape=cylindrical.shape[2:]),
                    mask=mask[i],
                    nr=nr,
                    shape=cylindrical.shape[2:],
                    center=origin[i],
                )
            )
            velocities[label][:, :, i] = masked_means(
                cylindrical[:, i, :, :], masks[label][-1], axes=(1, 2)
            )

        velocities[label] = subtract_bg(velocities[label])
        masks[label] = np.array(masks[label])

    return velocities, masks


def remap_array(data, new_shape, roi):
    result = np.zeros(data.shape[:-2] + new_shape, dtype=data.dtype)
    result[..., roi[0] : roi[1] + 1, roi[2] : roi[3] + 1] = data
    return result


def calculate_velocities(
    data: StrainMapData,
    cine: Text,
    global_velocity: bool = True,
    angular_regions: Sequence[int] = (),
    radial_regions: Sequence[int] = (),
    sign_reversal: Tuple[bool, ...] = (False, False, False),
    init_markers: bool = True,
):
    """Calculates the velocity of the chosen cine and regions."""
    swap, signs = data.data_files.orientation(cine)  # type: ignore
    phase = scale_phase(data, cine, swap, sign_reversal)  # type: ignore
    mask, orig, (xmin, xmax, ymin, ymax) = global_masks_and_origin(
        outer=data.segments[cine]["epicardium"],
        inner=data.segments[cine]["endocardium"],
    )
    origin = data.septum[cine][..., 1][:, ::-1]
    shift = np.array([xmin, ymin])
    rm_mask = remap_array(mask, phase.shape[-2:], (xmin, xmax, ymin, ymax))
    cylindrical = (
        transform_to_cylindrical(phase, rm_mask, origin)
        * (data.data_files.sensitivity * signs)[:, None, None, None]  # type: ignore
    )
    data.masks[cine][f"cylindrical"] = cylindrical
    data.sign_reversal = sign_reversal

    vel_labels: List[str] = []
    if global_velocity:
        velocities, masks = velocity_global(
            cylindrical[..., xmin : xmax + 1, ymin : ymax + 1], mask
        )
        masks = {
            k: remap_array(v, phase.shape[-2:], (xmin, xmax, ymin, ymax))
            for k, v in masks.items()
        }
        data.velocities[cine].update(velocities)
        data.masks[cine].update(masks)
        vel_labels += list(velocities.keys())

    if angular_regions:
        velocities, masks = velocities_angular(
            cylindrical[..., xmin : xmax + 1, ymin : ymax + 1],
            data.septum[cine],
            origin - shift[None, :],
            mask,
            angular_regions,
        )
        masks = {
            k: remap_array(v, phase.shape[-2:], (xmin, xmax, ymin, ymax))
            for k, v in masks.items()
        }
        data.velocities[cine].update(velocities)
        data.masks[cine].update(masks)
        vel_labels += list(velocities.keys())

    if radial_regions:
        velocities, masks = velocities_radial(
            cylindrical[..., xmin : xmax + 1, ymin : ymax + 1],
            data.segments[cine],
            origin - shift[None, :],
            shift,
            mask,
            radial_regions,
        )
        masks = {
            k: remap_array(v, phase.shape[-2:], (xmin, xmax, ymin, ymax))
            for k, v in masks.items()
        }
        data.velocities[cine].update(velocities)
        data.masks[cine].update(masks)
        vel_labels += list(velocities.keys())

    if init_markers:
        initialise_markers(data, cine, vel_labels)
        data.save(*[["markers", cine, vel] for vel in vel_labels], ["sign_reversal"])


def regenerate(data, cines, callback: Callable = terminal):
    """Regenerate velocities and masks information after loading from h5 file."""
    for i, d in enumerate(cines):
        callback(
            f"Regenerating existing velocities {i+1}/{len(cines)}.", i / len(cines),
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
                    regions[f"{rtype}_regions"] = [
                        int(num),
                    ]

        calculate_velocities(
            data=data,
            cine=d,
            sign_reversal=data.sign_reversal,
            init_markers=False,
            **regions,
        )
    callback(f"Regeneration complete!", 1)


markers_options = {
    "PS": dict(low=1, high=15, maximum=True),
    "PD": dict(low=15, high=30, maximum=False),
    "PAS": dict(low=35, high=47, maximum=False),
    "PC1": dict(low=1, high=5, maximum=False),
    "PC2": dict(low=6, high=12, maximum=True),
    "ES": dict(low=14, high=21),
}


def px_velocity_curves(
    data: StrainMapData, cine: str, nrad: int = 3, nang: int = 24,
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


def initialise_markers(data: StrainMapData, cine: str, vel_labels: list):
    """Initialises the markers for all the available velocities."""
    data.markers[cine]["global"] = markers_positions(data.velocities[cine]["global"])

    rvel = (key for key in vel_labels if key != "global")
    for vel_label in rvel:
        data.markers[cine][vel_label] = markers_positions(
            data.velocities[cine][vel_label], data.markers[cine]["global"][0][1, 3],
        )
    return data
