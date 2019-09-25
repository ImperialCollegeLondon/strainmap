from typing import Text, Tuple, Optional, Sequence
import numpy as np
from scipy import ndimage

from strainmap.models.contour_mask import masked_means, cylindrical_projection
from strainmap.models.readers import ImageTimeSeries

from .strainmap_data_model import StrainMapData
from .readers import read_all_images, images_to_numpy, velocity_sensitivity
from .contour_mask import contour_diff, angular_segments, radial_segments, Contour


def find_theta0(zero_angle: np.ndarray):
    """Finds theta0 out of the COM and septum mid-point for all timeframes."""
    shifted = zero_angle[:, :, 0] - zero_angle[:, :, 1]
    theta0 = np.mod(np.arctan2(shifted[:, 1], shifted[:, 0]), 2 * np.pi)
    return theta0


def scale_phase(data: StrainMapData, dataset_name: Text, phantom: bool = False):
    """Prepare the phases, scaling them and substracting the phantom, if needed."""
    images = list(
        images_to_numpy(
            read_all_images({dataset_name: data.data_files[dataset_name]})
        ).values()
    )[0]
    phase = images.phase / 4096

    if len(data.bg_files) > 0 and phantom:
        phantom = (
            list(
                images_to_numpy(
                    read_all_images({dataset_name: data.bg_files[dataset_name]})
                ).values()
            )[0].phase
            / 4096
        )
    else:
        phantom = np.full_like(phase, 0.5)

    return phase - phantom


def global_masks_and_origin(outer, inner, img_shape):
    """Finds the global masks and origin versus time frame."""
    masks = [contour_diff(outer[i], inner[i], img_shape) for i in range(len(outer))]
    origin = list(map(ndimage.measurements.center_of_mass, masks))

    return np.array(masks), np.array(origin)


def transform_to_cylindrical(phase: np.ndarray, masks: np.ndarray, origin: np.ndarray):
    """Transform the velocities to cylindrical coordinates.

    It also accounts for the different origin of coordinates for each time step (frame),
    and substracts for bulk movement of the heart in the plane."""
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


def velocities_radial_segments(
    cylindrical: np.ndarray,
    outer: np.ndarray,
    inner: np.ndarray,
    origin: np.ndarray,
    segments: int = 3,
):
    """Calculates the regional velocities of the chosen dataset for radial regions."""
    velocities = np.zeros((segments, cylindrical.shape[0], cylindrical.shape[1]))

    for i in range(cylindrical.shape[1]):
        labels = radial_segments(
            outer=Contour(outer[i], shape=cylindrical.shape[2:]),
            inner=Contour(inner[i], shape=cylindrical.shape[2:]),
            nr=segments,
            shape=cylindrical.shape[2:],
            center=origin[i],
        )
        velocities[:, :, i] = masked_means(cylindrical[:, i, :, :], labels, axes=(1, 2))

    return velocities


def calculate_velocities(
    data: StrainMapData,
    dataset_name: Text,
    global_velocity: bool = True,
    angular_regions: Sequence[int] = (),
    radial_regions: Sequence[int] = (),
    phantom: bool = False,
):
    """Calculates the velocity of the chosen dataset and regions."""
    phase = scale_phase(data, dataset_name, phantom)
    masks, origin = global_masks_and_origin(
        outer=data.segments[dataset_name]["epicardium"],
        inner=data.segments[dataset_name]["endocardium"],
        img_shape=phase.shape[2:],
    )
    cylindrical = transform_to_cylindrical(phase, masks, origin)
    sensitivity = velocity_sensitivity(data.data_files[dataset_name]["PhaseZ"][0]) * 2

    if global_velocity:
        s = np.tile(sensitivity, (cylindrical.shape[1], 1)).T
        data.velocities[dataset_name]["global"] = (
            masked_means(cylindrical, masks, axes=(2, 3)) * s
        )[0]

    for ang in angular_regions:
        s = np.tile(sensitivity, (cylindrical.shape[1], ang, 1)).transpose((1, 2, 0))
        theta0 = find_theta0(data.zero_angle[dataset_name])
        labels = angular_segments(
            nsegments=ang, origin=origin, theta0=theta0, shape=cylindrical.shape[2:]
        ).transpose((2, 0, 1))
        data.velocities[dataset_name][f"angular x{ang}"] = (
            masked_means(cylindrical, labels * masks, axes=(2, 3)) * s
        )

    for rad in radial_regions:
        s = np.tile(sensitivity, (cylindrical.shape[1], rad, 1)).transpose((1, 2, 0))
        epi = data.segments[dataset_name]["epicardium"]
        endo = data.segments[dataset_name]["endocardium"]
        data.velocities[dataset_name][f"radial x{rad}"] = (
            velocities_radial_segments(cylindrical, epi, endo, origin, rad) * s
        )

    return data


def mean_velocities(
    velocities: np.ndarray,
    labels: np.ndarray,
    component_axis: int = ImageTimeSeries.component_axis,
    image_axes: Tuple[int, int] = ImageTimeSeries.image_axes,
    time_axis: Optional[int] = ImageTimeSeries.time_axis,
    origin: Optional[np.ndarray] = None,
    **kwargs,
) -> np.ndarray:
    """Global and masked mean velocities in cylindrical basis.

    Args:
        velocities: phases in a cartesian basis
        labels: regions for which to compute the mean
        component_axis: axis of the (x, y, z) components
        image_axes: axes corresponding to the image
        time_axis: axis corresponding to time
        figure: the figure on which to plot
        origin: origin of the cylindrical basis

    Returns:
        a numpy array where the first axis indicates the label, and the second axis
        indicates the component (from `component_axis`). Subsequent axes are the extra
        axes from `velocities` (other than component and image axes). Label 0 is the
        global mean.
    """
    assert velocities.ndim >= len(image_axes) + (1 if time_axis is None else 2)
    if origin is None:
        origin = ndimage.measurements.center_of_mass(labels > 0)

    bulk_velocity = masked_means(velocities, labels > 0, axes=image_axes).reshape(
        tuple(1 if i in image_axes else v for i, v in enumerate(velocities.shape))
    )
    cylindrical = cylindrical_projection(
        velocities - bulk_velocity,
        origin,
        component_axis=component_axis,
        image_axes=image_axes,
    )
    local_means = masked_means(cylindrical, labels, axes=image_axes)
    global_means = masked_means(cylindrical, labels > 0, axes=image_axes)

    return np.rollaxis(
        np.concatenate([global_means, local_means], axis=0),
        component_axis - sum(i < component_axis for i in image_axes),
    )


markers_options = {
    "PS": dict(low=1, high=15, maximum=True),
    "PD": dict(low=15, high=30, maximum=False),
    "PAS": dict(low=31, high=41, maximum=False),
    "PC1": dict(low=1, high=5, maximum=False),
    "PC2": dict(low=6, high=12, maximum=True),
}


def marker(comp, low=1, high=49, maximum=True):
    """Finds the index and value of the marker position within the given range."""
    low = min(low, len(comp) - 1)
    high = min(high, len(comp) - 1)

    idx = (
        np.argmax(comp[low : high + 1]) if maximum else np.argmin(comp[low : high + 1])
    ) + low
    return [idx, comp[idx]]


def marker_es(comp, pd):
    """Finds the default position of the ES marker."""
    idx = round(len(comp) / 3)
    low = min(max(1, idx - 5), len(comp) - 1)
    high = min(idx + 5, len(comp) - 1)

    idx = np.argmin(comp[low : high + 1]) + low

    if idx == pd[0] or comp[idx] < -2:
        idx = np.argmin(abs(comp[low : high + 1])) + low

    return [idx, comp[idx]]


def marker_pc3(comp, es):
    """Finds the default position of the PC3 marker."""
    low = es[0]
    high = min(low + 10, len(comp) - 1)

    pos = np.max(comp[low : high + 1])
    neg = np.min(comp[low : high + 1])

    idx = (
        np.argmax(comp[low : high + 1])
        if pos > abs(neg)
        else np.argmin(comp[low : high + 1])
    ) + low

    value = comp[idx]
    if abs(value) < 0.5:
        idx = np.nan
        value = np.nan

    return [idx, value]


def normalised_times(markers: Tuple, frames: int) -> Tuple:
    """Calculates the normalised values for the marker times.

    The timings of the peaks (PS, PD and PAS) within the cardiac cycle are heart rate
    dependent. To reduce variability due to changes in heart rate, we normalise the
    cardiac cycle to a fixed length of 1000ms (ie a heart rate of 60bpm). As the heart
    rate changes, the length of systole remains relatively fixed while the length of
    diastole changes substantially – so we do this in two stages.

    Our normalised curves  have a systolic length of 350 ms and a diastolic length of
    650ms.
    """
    es = markers[1]["ES"][0]

    for i in range(3):
        for k, x in markers[i].items():
            if len(x) == 3:
                del markers[i][k][-1]
            if x[0] <= es:
                markers[i][k].append(smaller_than_es(x[0], es))
            else:
                markers[i][k].append(larger_than_es(x[0], es, frames))

    return markers


def smaller_than_es(x, es, syst=350):
    """Normalization for times smaller than ES."""
    return x / es * syst


def larger_than_es(x, es, frames, syst=350, dias=650):
    """Normalization for times larger than ES."""
    return syst + (x - es) / (frames - es) * dias


def _markers_positions(velocity: np.ndarray, es: Optional[Tuple] = None) -> Tuple:
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
    """
    markers: Tuple[dict, dict, dict] = (
        {"PS": [], "PD": [], "PAS": []},
        {"PS": [], "PD": [], "PAS": []},
        {"PC1": [], "PC2": []},
    )

    for i in range(3):
        for k in markers[i].keys():
            markers[i][k] = marker(velocity[i], **markers_options[k])

    markers[1]["ES"] = marker_es(velocity[1], markers[1]["PD"]) if es is None else es
    markers[2]["PC3"] = marker_pc3(velocity[2], markers[1]["ES"])

    return normalised_times(markers, len(velocity[0]))


def markers_positions(
    data: StrainMapData, dataset: str, vel_label: str, global_vel: str = None
):
    """Find the position of the markers for the chosen dataset and velocity."""
    velocity = data.velocities[dataset][vel_label]
    es = data.markers[dataset][global_vel][1]["ES"] if global_vel is not None else None
    data.markers[dataset][vel_label] = _markers_positions(velocity, es)
    return data


def update_marker(
    data: StrainMapData,
    dataset: str,
    vel_label: str,
    component_idx: int,
    marker_labl: str,
    marker: int,
):
    """Updates the position of an existing marker in the data object.

    If the marker modified is "ES", then all markers are updated. Otherwise just the
    chosen one is updated."""
    value = data.velocities[dataset][vel_label][component_idx][marker]
    frames = len(data.velocities[dataset][vel_label][component_idx])

    if marker_labl == "ES":
        data.markers[dataset][vel_label][1]["ES"][0] = marker
        data.markers[dataset][vel_label][1]["ES"][1] = value
        data.markers[dataset][vel_label] = normalised_times(
            data.markers[dataset][vel_label], frames
        )

    else:
        es = data.markers[dataset][vel_label][1]["ES"][0]

        if marker <= es:
            normalised = smaller_than_es(marker, es)
        else:
            normalised = larger_than_es(marker, es, frames)

        data.markers[dataset][vel_label][component_idx][marker_labl] = [
            marker,
            value,
            normalised,
        ]

    return data
