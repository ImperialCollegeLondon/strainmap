from typing import Text, Tuple, Optional, Sequence
import numpy as np
from scipy import ndimage

from strainmap.models.contour_mask import masked_means, cylindrical_projection
from strainmap.models.readers import ImageTimeSeries

from .strainmap_data_model import StrainMapData
from .readers import read_all_images, images_to_numpy, velocity_sensitivity
from .contour_mask import contour_diff, angular_segments, radial_segments, Contour


def find_theta0(zero_angle: np.ndarray):
    """Finds theta0 out of the COM and septum mid-point for all timeframes.

zero_angle: array with dimensions [num_timeframes, 2, 2], where the second axis discriminates between the center of mass (index 0) and the septum (index 1), and the last axis are the cartesian coordinates x and y.

    For any timeframe,  it contains the center of mass of the mask and the position
    of the septum mid-point. Out of these two points, the angle that is the origin of
    angular coordinates is calculated.
    """
    shifted = zero_angle[:, :, 0] - zero_angle[:, :, 1]
    theta0 = np.mod(np.arctan2(shifted[:, 1], shifted[:, 0]), 2 * np.pi)
    return theta0


def scale_phase(
    data: StrainMapData, dataset_name: Text, phantom: bool = False, scale=1 / 4096
):
    """Prepare the phases, scaling them and substracting the phantom, if needed."""
    images = images_to_numpy(
        read_all_images({dataset_name: data.data_files[dataset_name]})
    )[dataset_name]
    phase = images.phase * scale

    if len(data.bg_files) > 0 and phantom:
        phantom_phase = (
            images_to_numpy(
                read_all_images({dataset_name: data.bg_files[dataset_name]})
            )[dataset_name].phase
            * scale
        )

    else:
        phantom_phase = 0.5

    return phase - phantom_phase


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
