from typing import Dict, List, Optional, Sequence, Text, Tuple, Union, Callable

import numpy as np
from scipy import ndimage

from strainmap.models.contour_mask import cylindrical_projection, masked_means

from .contour_mask import Contour, angular_segments, contour_diff, radial_segments
from .strainmap_data_model import StrainMapData
from .writers import terminal


def find_theta0(zero_angle: np.ndarray):
    """Finds theta0 out of the COM and septum mid-point for all timeframes.

    zero_angle: array with dimensions [num_timeframes, 2, 2], where the last axis
    discriminates between the center of mass (index 0) and the septum (index 1), and
    the second axis are the cartesian coordinates x and y.

    For any timeframe,  it contains the center of mass of the mask and the position
    of the septum mid-point. Out of these two points, the angle that is the origin of
    angular coordinates is calculated.
    """
    shifted = zero_angle[:, :, 0] - zero_angle[:, :, 1]
    theta0 = np.arctan2(shifted[:, 1], shifted[:, 0])
    return theta0


def scale_phase(
    data: StrainMapData,
    dataset_name: Text,
    bg: str = "Estimated",
    swap: bool = False,
    sign_reversal=(False, False, False),
    scale=1 / 4096,
):
    """Prepare the phases, scaling them and substracting the phantom, if needed."""
    phase = data.data_files.phase(dataset_name) * scale

    if data.bg_files is not None and bg in data.bg_files.datasets:  # type: ignore
        phantom_phase = data.bg_files.phase(bg) * scale
    else:
        phantom_phase = 0.5

    phase = phase - phantom_phase

    if swap:
        phase[0], phase[1] = phase[1], phase[0].copy()

    for i, r in enumerate(sign_reversal):
        phase[i] *= -1 if r else 1

    return phase


def global_masks_and_origin(outer, inner, img_shape):
    """Finds the global masks and origin versus time frame."""
    ymin, ymax = int(outer[:, 0].min()), int(outer[:, 0].max()) + 1
    xmin, xmax = int(outer[:, 1].min()), int(outer[:, 1].max()) + 1
    shift = np.array([ymin, xmin])
    masks = np.array(
        [
            contour_diff(c1=o.T, c2=i.T, shape=(ymax - ymin, xmax - xmin)).T
            for o, i in zip(outer - shift[None, :, None], inner - shift[None, :, None])
        ]
    )
    origin = np.array([*map(ndimage.measurements.center_of_mass, masks)])

    return masks, origin, (xmin, xmax, ymin, ymax)


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


def substract_estimated_bg(
    velocities: np.ndarray, bg: str = "Estimated", axis: int = 2
):
    """Subtracts the estimated background to the velocities, if required."""
    if bg != "Estimated":
        return velocities

    return velocities - velocities.mean(axis=axis)[:, :, None]


def velocity_global(cylindrical: np.ndarray, mask: np.ndarray, bg: str):
    """Calculate the global velocity."""
    label = f"global - {bg}"
    velocities = {label: masked_means(cylindrical, mask, axes=(2, 3))}
    velocities[label] = substract_estimated_bg(velocities[label], bg=bg)
    masks = {label: mask}
    return velocities, masks


def velocities_angular(
    cylindrical: np.ndarray,
    zero_angle: np.ndarray,
    origin: np.ndarray,
    mask: np.ndarray,
    bg: str,
    regions: Sequence[int] = (6,),
):
    """Calculate the angular velocities for angular regions."""
    theta0 = find_theta0(zero_angle)

    velocities: Dict[str, np.ndarray] = dict()
    masks: Dict[str, np.ndarray] = dict()
    for ang in regions:
        label = f"angular x{ang} - {bg}"
        region_labels = angular_segments(
            nsegments=ang, origin=origin, theta0=theta0, shape=cylindrical.shape[2:]
        ).transpose((2, 0, 1))
        velocities[label] = masked_means(cylindrical, region_labels * mask, axes=(2, 3))
        velocities[label] = substract_estimated_bg(velocities[label], bg=bg)
        masks[label] = region_labels * mask

    return velocities, masks


def velocities_radial(
    cylindrical: np.ndarray,
    segments: Dict[str, np.ndarray],
    origin: np.ndarray,
    shift: np.ndarray,
    mask: np.ndarray,
    bg: str,
    regions: Sequence[int] = (3,),
):
    """Calculates the regional velocities for radial regions."""
    outer = segments["epicardium"] - shift[None, :, None]
    inner = segments["endocardium"] - shift[None, :, None]

    velocities: Dict[str, np.ndarray] = dict()
    masks: Dict[str, Union[list, np.ndarray]] = dict()
    for nr in regions:
        label = f"radial x{nr} - {bg}"
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

        velocities[label] = substract_estimated_bg(velocities[label], bg=bg)
        masks[label] = np.array(masks[label])

    return velocities, masks


def remap_array(data, new_shape, roi):
    result = np.zeros(data.shape[:-2] + new_shape, dtype=data.dtype)
    result[..., roi[0] : roi[1], roi[2] : roi[3]] = data
    return result


def calculate_velocities(
    data: StrainMapData,
    dataset_name: Text,
    global_velocity: bool = True,
    angular_regions: Sequence[int] = (),
    radial_regions: Sequence[int] = (),
    bg: str = "Estimated",
    sign_reversal: Tuple[bool, ...] = (False, False, False),
    init_markers: bool = True,
):
    """Calculates the velocity of the chosen dataset and regions."""
    swap, signs = data.data_files.orientation  # type: ignore
    phase = scale_phase(data, dataset_name, bg, swap, sign_reversal)  # type: ignore
    mask, origin, (xmin, xmax, ymin, ymax) = global_masks_and_origin(
        outer=data.segments[dataset_name]["epicardium"],
        inner=data.segments[dataset_name]["endocardium"],
        img_shape=phase.shape[2:],
    )
    shift = np.array([ymin, xmin])
    cylindrical = (
        transform_to_cylindrical(phase[..., xmin:xmax, ymin:ymax], mask, origin)
        * (data.data_files.sensitivity * signs)[:, None, None, None]  # type: ignore
    )
    data.masks[dataset_name][f"cylindrical - {bg}"] = remap_array(
        cylindrical, phase.shape[-2:], (xmin, xmax, ymin, ymax)
    )
    data.sign_reversal = sign_reversal

    vel_labels: List[str] = []
    if global_velocity:
        velocities, masks = velocity_global(cylindrical, mask, bg)
        masks = {
            k: remap_array(v, phase.shape[-2:], (xmin, xmax, ymin, ymax))
            for k, v in masks.items()
        }
        data.velocities[dataset_name].update(velocities)
        data.masks[dataset_name].update(masks)
        vel_labels += list(velocities.keys())

    if angular_regions:
        velocities, masks = velocities_angular(
            cylindrical,
            data.zero_angle[dataset_name],
            origin,
            mask,
            bg,
            angular_regions,
        )
        masks = {
            k: remap_array(v, phase.shape[-2:], (xmin, xmax, ymin, ymax))
            for k, v in masks.items()
        }
        data.velocities[dataset_name].update(velocities)
        data.masks[dataset_name].update(masks)
        vel_labels += list(velocities.keys())

    if radial_regions:
        velocities, masks = velocities_radial(
            cylindrical,
            data.segments[dataset_name],
            origin,
            shift,
            mask,
            bg,
            radial_regions,
        )
        masks = {
            k: remap_array(v, phase.shape[-2:], (xmin, xmax, ymin, ymax))
            for k, v in masks.items()
        }
        data.velocities[dataset_name].update(velocities)
        data.masks[dataset_name].update(masks)
        vel_labels += list(velocities.keys())

    if init_markers:
        initialise_markers(data, dataset_name, vel_labels)
        data.save(
            *[["markers", dataset_name, vel] for vel in vel_labels], ["sign_reversal"]
        )


def regenerate(data, datasets, callback: Callable = terminal):
    """Regenerate velocities and masks information after loading from h5 file."""
    for i, d in enumerate(datasets):
        callback(
            f"Regenerating existing velocities {i+1}/{len(datasets)}.",
            i / len(datasets),
        )
        vels = data.velocities[d]
        regions = dict()
        for k, v in vels.items():
            info = k.split(" - ")
            if info[-1] not in regions.keys():
                regions[info[-1]] = dict(angular_regions=[], radial_regions=[])
            if "global" in info[0]:
                regions[info[-1]]["global_velocity"] = True
            else:
                rtype, num = info[0].split(" x")
                regions[info[-1]][f"{rtype}_regions"].append(int(num))

        for bg, region in regions.items():
            calculate_velocities(
                data=data,
                dataset_name=d,
                bg=bg,
                sign_reversal=data.sign_reversal,
                init_markers=False,
                **region,
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
    """Find the position of the markers for the chosen dataset and velocity.

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
    """Find the position of the markers for the chosen dataset and velocity."""
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
    dataset: str,
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
    data.markers[dataset][vel_label][region] = _update_marker(
        data.velocities[dataset][vel_label][region],
        data.markers[dataset][vel_label][region],
        component,
        marker_idx,
        position,
    )
    data.save(["markers", dataset, vel_label])


def initialise_markers(data: StrainMapData, dataset: str, vel_labels: list):
    """Initialises the markers for all the available velocities."""
    gvel = [key for key in vel_labels if "global" in key]
    rvel = [key for key in vel_labels if "global" not in key]

    for vel_label in gvel:
        data.markers[dataset][vel_label] = markers_positions(
            data.velocities[dataset][vel_label]
        )

    for vel_label in rvel:
        bg = vel_label.split("-")[-1]
        global_vel = f"global -{bg}"
        data.markers[dataset][vel_label] = markers_positions(
            data.velocities[dataset][vel_label],
            data.markers[dataset][global_vel][0][1, 3],
        )
    return data
