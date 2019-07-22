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
) -> None:
    from ..models.contour_mask import masked_means, cylindrical_projection

    assert velocities.ndim >= len(image_axes) + 1
    if figure is None:
        figure = plt.gcf()

    bulk_velocity = masked_means(velocities, labels > 0, axes=image_axes)[0, ...]
    cylindrical = cylindrical_projection(
        velocities - bulk_velocity,
        origin,
        component_axis=component_axis,
        image_axes=image_axes,
    )
    local_means = masked_means(cylindrical, labels, axes=image_axes)
    global_means = masked_means(cylindrical, labels > 0, axes=image_axes)
    subaxes = figure.subplots(2, velocities.shape[component_axis])
    for j in range(velocities.shape[component_axis]):
        subaxes[0, j].plot(np.take(global_means, j, axis=component_axis))
        subaxes[1, j].plot(np.take(local_means, j, axis=component_axis))
