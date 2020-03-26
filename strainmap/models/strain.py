from typing import Callable, Dict, Optional, Sequence, Text, Tuple, Union

import numpy as np

from .strainmap_data_model import StrainMapData
from .velocities import regenerate


def calculate_strain(
    data: StrainMapData, dataset_name: Union[Sequence[Text], Text]
) -> None:
    """Calculate all components of the strain for the given datasets.

    If there is only one dataset, the longitudinal strain is not calculated. In the end,
    the StrainMapData object is updated with the new information."""

    datasets = dataset_name if isinstance(dataset_name, list) else [dataset_name]
    datasets = sorted(datasets, key=data.data_files.slice_loc)

    # Do we need to calculate the strain?
    if all([d in data.strain.keys() for d in datasets]):
        return

    # Do we need to regenerate the velocities?
    to_regen = [d for d in datasets if list(data.velocities[d].values())[0] is None]
    if len(to_regen) > 0:
        regenerate(data, to_regen)

    # Do we have enough slices to calculate out-of-plane strain?
    results_long: Optional[Dict[str, np.ndarray]] = None
    if len(datasets) > 1:
        results_long = calculate_outofplane_strain(
            data, datasets=datasets, component_axis=0
        )

    result = calculate_inplane_strain(data, datasets=datasets)

    for d, cyl in result.items():
        regional = calculate_regional_strain(cyl, data.masks[d])

        if results_long is None:
            # If there are no longitudinal results, the needed arrays are filled with
            # zeros
            for i, r in regional.items():
                regional[i] = np.concatenate((np.zeros_like(r[:, :1, :]), r), axis=1)
            cyl = np.concatenate((np.zeros_like(cyl[:1, ...]), cyl), axis=0)

        else:
            # Otherwise, the longitudinal data - only angular x6, for now - is
            # incorporated.
            j = datasets.index(d)
            for i, r in regional.items():
                lng = results_long.get(i, None)
                if lng is None:
                    regional[i] = np.concatenate(
                        (np.zeros_like(r[:, :1, :]), r), axis=1
                    )
                else:
                    regional[i] = np.concatenate((lng[j][:, None, :], r), axis=1)

            cyl = np.concatenate((np.zeros_like(cyl[:1, ...]), cyl), axis=0)

        data.strain[d].update(regional)
        data.strain[d]["cylindrical - Estimated"] = cyl


def calculate_regional_strain(
    strain: np.ndarray, masks: Dict[Text, np.ndarray]
) -> Dict[Text, np.ndarray]:
    """Calculate the regional strains."""
    from strainmap.models.contour_mask import masked_means

    result: Dict[Text, np.ndarray] = {}
    for k, mask in masks.items():
        if "cylindrical" in k or "24" in k:
            continue
        result[k] = masked_means(strain, mask, axes=(2, 3))

    return result


def calculate_inplane_strain(
    data: StrainMapData,
    bg: str = "Estimated",
    datasets: Optional[Sequence[Text]] = None,
) -> Dict[Text, np.ndarray]:
    """Calculates the strain and updates the Data object with the result."""
    from strainmap.models.velocities import global_masks_and_origin

    if datasets is None:
        datasets = list(data.data_files.files)

    result: Dict[Text, np.ndarray] = {}
    for dataset in datasets:
        cylindrical = data.masks.get(dataset, {}).get(f"cylindrical - {bg}", None)
        if cylindrical is None:
            msg = f"Velocities from {dataset} with background {bg} are not available."
            raise RuntimeError(msg)

        mask, origin = global_masks_and_origin(
            outer=data.segments[dataset]["epicardium"],
            inner=data.segments[dataset]["endocardium"],
            img_shape=cylindrical.shape[-2:],
        )

        result[dataset] = np.zeros_like(cylindrical[1:])
        for t in range(cylindrical.shape[1]):
            result[dataset][:, t] = (
                inplane_strain_rate(
                    np.ma.array(
                        cylindrical[:2, t], mask=np.repeat(~mask[t : t + 1], 2, axis=0)
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
    component_axis: int = 0,
    image_axes: Tuple[int, int] = (-2, -1),
    regions: Optional[Sequence[int]] = None,
) -> Dict[str, np.ndarray]:
    """Wrangles zz strain component from existing data."""
    from operator import itemgetter
    from strainmap.models.contour_mask import masked_means

    if datasets is None:
        datasets = list(data.data_files.files)
    vzz = []
    vzz_global = []

    for dataset in datasets:
        phases = data.masks.get(dataset, {}).get(f"cylindrical - {bg}", None)
        if phases is None:
            msg = f"Phases from {dataset} with background {bg} are not available."
            raise RuntimeError(msg)
        masks = data.masks.get(dataset, {}).get(f"angular x{nangular} - {bg}", None)
        masks_global = data.masks.get(dataset, {}).get(f"global - {bg}", None)
        if masks is None:
            msg = (
                f"{nangular}-fold angular masks from {dataset} "
                f"with background {bg} are not available."
            )
            raise RuntimeError(msg)

        factor = 1.0
        try:
            factor *= data.data_files.time_interval(dataset) / 50
        except AttributeError:
            pass

        zcomp = np.take(phases, 0, component_axis)
        vzz.append(
            np.cumsum(
                masked_means(zcomp, masks, axes=image_axes, regions=regions) * factor,
                axis=-1,
            )
        )
        vzz_global.append(
            np.cumsum(
                np.mean(
                    np.ma.masked_array(zcomp, mask=~masks_global) * factor,
                    axis=image_axes,
                ),
                axis=-1,
            )[None, :]
        )

    zs = [data.data_files.slice_loc(d) for d in datasets]
    levels = sorted(zip(vzz, zs), key=itemgetter(1))
    levels_global = sorted(zip(vzz_global, zs), key=itemgetter(1))

    result = {
        f"angular x{nangular} - {bg}": np.gradient(
            [l[0] for l in levels], [l[1] for l in levels], axis=0
        )
    }
    result[f"global - {bg}"] = np.gradient(
        [l[0] for l in levels_global], [l[1] for l in levels_global], axis=0
    )
    return result


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
            RectBivariateSpline otherwise.
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
    from scipy.interpolate import RectBivariateSpline
    from scipy.ndimage import gaussian_filter

    radial_velocity = gaussian_filter(np.take(velocities, 0, component_axis), (10, 10))
    azimutal_velocity = gaussian_filter(
        np.take(velocities, 1, component_axis), (10, 10)
    )
    is_masked = hasattr(velocities, "mask")
    grid = np.ogrid[range(radial_velocity.shape[0]), range(radial_velocity.shape[1])]

    if not is_masked:
        points = grid
    else:
        radial_velocity = np.ma.masked_array(
            radial_velocity, mask=np.take(velocities, 0, component_axis).mask
        )
        azimutal_velocity = np.ma.masked_array(
            azimutal_velocity, mask=np.take(velocities, 1, component_axis).mask
        )
        assert (radial_velocity.mask == azimutal_velocity.mask).all()
        points = (~radial_velocity.mask).nonzero()

    spline = spline or RectBivariateSpline

    radial_spline = spline(*grid, radial_velocity, **kwargs)
    azimutal_spline = spline(*grid, azimutal_velocity, **kwargs)

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

    result = np.zeros_like(velocities)
    result[0, points[0], points[1]] = radial_strain
    result[1, points[0], points[1]] = azimutal_strain
    infs = np.isinf(result)
    if infs.any():
        result[infs] = np.mean(result[~infs])
    return result
