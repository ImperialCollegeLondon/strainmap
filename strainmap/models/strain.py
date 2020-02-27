from typing import Callable, Dict, Optional, Sequence, Text, Tuple

import numpy as np

from .strainmap_data_model import StrainMapData


def calculate_inplane_strain(
    data: StrainMapData,
    bg: str = "Estimated",
    datasets: Optional[Sequence[Text]] = None,
) -> Dict[Text, np.ndarray]:
    """Calculates the strain and updates the Data object with the result."""
    from strainmap.models.velocities import global_masks_and_origin

    swap, signs = data.data_files.orientation
    if datasets is None:
        datasets = list(data.data_files.files)

    result: Dict[Text, np.ndarray] = {}
    for dataset in datasets:
        phases = data.masks.get(dataset, {}).get(f"cylindrical - {bg}", None)
        if phases is None:
            msg = f"Phases from {dataset} with background {bg} are not available."
            raise RuntimeError(msg)
        mask, origin = global_masks_and_origin(
            outer=data.segments[dataset]["epicardium"],
            inner=data.segments[dataset]["endocardium"],
            img_shape=phases.shape[-2:],
        )

        result[dataset] = np.zeros_like(phases[1:])
        for t in range(phases.shape[1]):
            result[dataset][:, t] = (
                inplane_strain_rate(
                    np.ma.array(
                        phases[:2, t], mask=np.repeat(~mask[t : t + 1], 2, axis=0)
                    ),
                    origin=origin[t],
                ).data
                * mask[t : t + 1]
            )

        factor = 1.0
        try:
            factor *= data.data_files.time_interval(dataset)
        except AttributeError:
            pass
        try:
            factor /= data.data_files.pixel_size(dataset)
        except AttributeError:
            pass
        result[dataset] *= factor

    return result


def calculate_outofplane_strain(
    data: StrainMapData,
    bg: str = "Estimated",
    nangular: int = 6,
    datasets: Optional[Sequence[Text]] = None,
    component_axis: int = 1,
    image_axes: Tuple[int, int] = (2, 3),
    regions: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Wrangles zz strain component from existing data."""
    from operator import itemgetter
    from strainmap.models.contour_mask import masked_means

    if datasets is None:
        datasets = list(data.data_files.files)
    vzz = []
    zs = []
    for dataset in datasets:
        phases = data.masks.get(dataset, {}).get(f"cylindrical - {bg}", None)
        if phases is None:
            msg = f"Phases from {dataset} with background {bg} are not available."
            raise RuntimeError(msg)
        masks = data.masks.get(dataset, {}).get(f"angular x{nangular} - {bg}", None)
        if masks is None:
            msg = (
                f"{nangular}-fold angular masks from {dataset} "
                f"with background {bg} are not available."
            )
            raise RuntimeError(msg)

        factor = 1.0
        try:
            factor *= data.data_files.time_interval(dataset)
        except AttributeError:
            pass

        iaxes = image_axes[0] % phases.ndim, image_axes[1] % phases.ndim
        iaxes = (
            image_axes[0] - int(iaxes[0] > (component_axis % phases.ndim)),
            image_axes[1] - int(iaxes[1] > (component_axis % phases.ndim)),
        )

        vzz.append(
            masked_means(
                np.take(phases, -1, component_axis), masks, axes=iaxes, regions=regions,
            )
            * factor
        )
        zs.append(data.data_files.slice_loc(dataset))

    levels = sorted(zip(vzz, zs), key=itemgetter(1))
    return np.gradient([l[0] for l in levels], [l[1] for l in levels], axis=0)


def inplane_strain_rate(
    velocities: np.ndarray,
    component_axis: int = 0,
    spline: Optional[Callable] = None,
    origin: Sequence[float] = (0, 0),
    **kwargs,
) -> np.ndarray:
    """Stain rate for a single image.

    Computes the strain rate for a single MRI image. The image can be masked, in which
    case velocities outside the mask are not taken into account.

    The strain is obtained by first creating an interpolating spline and then taking
    it's derivatives.

    Parameters:
        velocities: velocities with radial and azimutal components on the
            `component_axis`. Can also be a masked array.
        component_axis: axis of the radial and azimutal components of the velocities.
        spline: Function from which to create an interpolation object. Defaults to
            SmoothBivariateSpline for masked arrays and RectBivariateSpline otherwise.
        origin: Origin of the cylindrical coordinate system. Because image vs array
            issues, `origin[1]` correspond to `x` and  `origin[0]` to `y`.
        **kwargs: passed on the spline function.

    .. note::
        In the context of the MRI velocities, the instantaneous strain and the strain
        rate differ only by a factor equal to the time-step.

    Example:

        We can create a velocity field with a linear radial component and no azimutal
        component. It should yield equal radial and azimutal strain, at least outside of
        a small region at the origin:

        >>> from pytest import approx
        >>> import numpy as np
        >>> from strainmap.models.strain import inplane_strain_rate
        >>> origin = np.array((50.5, 60.5))
        >>> cart_index = np.mgrid[:120,:100] - origin[::-1, None, None]
        >>> r = np.linalg.norm(cart_index, axis=0)
        >>> strain = inplane_strain_rate((r, np.zeros_like(r)), origin=origin)
        >>> np.max(np.abs(strain[0])).round(2)
        1.05
        >>> (np.argmax(np.abs(strain[0])) // strain.shape[2]) in range(50, 71)
        True
        >>> (np.argmax(np.abs(strain[0])) % strain.shape[2]) in range(40, 61)
        True
        >>> strain[0, 50:71, 40:61] = 1
        >>> assert strain[0] == approx(1, rel=1e-4)
        >>> assert strain[1] == approx(1)


        To test the azimutal strain, we can create a simple vortex with no radial
        velocity and 1/r angular velocity. The mask removes consideration outside of a
        ring centered on the vortex. It also removes the vortex itself from
        consideration. We expect the diagonal strain components to be almost zero (only
        the shear strain - which is not computed here
        - is nonzero).

        >>> import numpy as np
        >>> from strainmap.models.strain import inplane_strain_rate
        >>> v_theta = 1 / (r + 1)
        >>> velocities = np.ma.array(
        ...     (np.zeros_like(v_theta), v_theta),
        ...     mask = np.repeat(np.logical_or(r < 15, r > 50)[None], 2, 0),
        ... )
        >>> strain = inplane_strain_rate(velocities, origin=origin)
        >>> assert np.abs(strain.max()) < 1e-3

        Increasing the size of the grid yields better results (e.g. the bound on the
        strain is lower).
    """
    from scipy.interpolate import SmoothBivariateSpline, RectBivariateSpline

    radial_velocity = np.take(velocities, 0, component_axis)
    azimutal_velocity = np.take(velocities, 1, component_axis)
    is_masked = hasattr(velocities, "mask")
    if not is_masked:
        points = np.ogrid[
            range(radial_velocity.shape[0]), range(radial_velocity.shape[1])
        ]
        spline = spline or RectBivariateSpline
    else:
        assert (radial_velocity.mask == azimutal_velocity.mask).all()
        points = (~radial_velocity.mask).nonzero()
        radial_velocity = radial_velocity[points[0], points[1]].data
        azimutal_velocity = azimutal_velocity[points[0], points[1]].data
        spline = spline or SmoothBivariateSpline

    radial_spline = spline(*points, radial_velocity, **kwargs)
    azimutal_spline = spline(*points, azimutal_velocity, **kwargs)

    x = points[0] - origin[1]
    y = points[1] - origin[0]
    r = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)

    # fmt: off
    radial_strain = (
        radial_spline(*points, dx=1, grid=False) * np.cos(theta)
        + radial_spline(*points, dy=1, grid=False) * np.sin(theta)
    )
    azimutal_strain = (
        radial_spline(*points, grid=False) / r
        + azimutal_spline(*points, dy=1, grid=False) * np.cos(theta)
        - azimutal_spline(*points, dx=1, grid=False) * np.sin(theta)
    )
    # fmt: on

    def rhs(values):
        if not is_masked:
            return values

        x = np.zeros_like(np.take(velocities, 0, component_axis))
        x[points[0], points[1]] = values
        return x

    # Transform to cartesian coordinates
    result = np.zeros_like(velocities)
    indices = [slice(i) for i in result.shape]
    indices[component_axis] = 0  # type: ignore
    result[tuple(indices)] = rhs(radial_strain)
    indices[component_axis] = 1  # type: ignore
    result[tuple(indices)] = rhs(azimutal_strain)
    return result
