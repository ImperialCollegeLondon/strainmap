from typing import Optional, Tuple

import numpy as np
import numpy.ma
from matplotlib import pyplot as plt


def plot_velocities(
    velocities: np.ndarray,
    labels: np.ndarray,
    component_axis: int = 0,
    image_axes: Tuple[int, int] = (1, 2),
    figure: Optional[plt.Figure] = None,
    origin: Optional[np.ndarray] = None,
    **kwargs,
) -> np.ndarray:
    """Plot global and masked mean velocities in cylindrical basis.

    Args:
        velocities: phases in a cartesian basis
        labels: regions for which to compute the mean
        component_axis: axis of the (x, y, z) components
        image_axes: axes corresponding to the image
        figure: the figure on which to plot
        origin: origin of the cylindrical basis

    Returns:
        An array of subplot axes.
    """
    from ..models.contour_mask import masked_means, cylindrical_projection

    assert velocities.ndim >= len(image_axes) + 1
    if figure is None:
        figure = plt.gcf()

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
    all_means = np.concatenate([global_means, local_means], axis=0)
    subaxes = figure.subplots(1, velocities.shape[component_axis])
    axis = [
        j if j != component_axis else None for j in range(len(velocities.shape))
    ].index(None) + 1
    titles = ("rS", "θ", "z")
    for j, (axes, title) in enumerate(zip(subaxes, titles)):
        axes.plot(np.moveaxis(np.take(all_means, j, axis=axis), 0, -1))
        axes.title.set_text(title)
        axes.set_xlabel("Frame")
        axes.set_ylabel("Velocity")
    return subaxes
