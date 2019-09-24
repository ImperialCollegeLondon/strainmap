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
):
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
    from strainmap.models.velocities import mean_velocities

    assert velocities.ndim == len(image_axes) + 1 + 1

    if figure is None:
        figure = plt.figure()

    meanvel = mean_velocities(
        velocities,
        labels,
        component_axis=component_axis,
        image_axes=image_axes,
        origin=origin,
    )

    for (c, ax) in enumerate(figure.subplots(1, meanvel.shape[1])):
        ax.plot(meanvel[:, c].T, **kwargs)
        ax.title.set_text(["r", "theta", "z"][c])
        ax.legend([f"region {i}" for i in range(meanvel.shape[0])])
