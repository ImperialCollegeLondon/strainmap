from itertools import product
from typing import Callable, Dict, Optional, Sequence, Text, Tuple, Union

import numpy as np

from .readers import DICOMReaderBase
from .strainmap_data_model import StrainMapData
from .velocities import find_theta0


def cartcoords(shape: tuple, *sizes: Union[float, np.ndarray]) -> Tuple:
    """Create cartesian coordinates of the given shape based on the pixel sizes.

    sizes can be an scalar indicating the step value of the coordinate in the given
    dimension, or an array already indicating the positions. In the later case, these
    are shifted so the coordinate starts at 0.
    """
    assert len(shape) == len(sizes)

    def build_coordinate(length, value):
        if isinstance(value, np.ndarray):
            return value - value[0]
        else:
            return np.linspace(0, value * length, length, endpoint=False)

    return tuple(map(build_coordinate, shape, sizes))


def cylcoords(
    z: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    origin: np.ndarray,
    theta0: Union[float, np.ndarray],
    lent: int,
) -> Tuple:
    """Calculate the cylindrical coordinates out of X, Y, Z, an origin and theta0.

    Notice that this is different from the velocity calculations where origin and theta0
    might depend on time (different frames/cines), not on Z (different heart slices).

    In other words, these are the spacial coordinates for a specific time frame and all
    relevant short axis slices (typically there will be 8 of them).
    """

    org = validate_origin(origin, len(z), lent)
    th0 = validate_theta0(theta0, len(z), lent)

    coords = np.meshgrid(z, x, y, indexing="ij")
    zz, xx, yy = (np.tile(c, (lent, 1, 1, 1)) for c in coords)
    xx -= org[:, :, -2:-1, None]
    yy -= org[:, :, -1:, None]
    r = np.sqrt(xx ** 2 + yy ** 2)
    theta = np.arctan2(yy, xx) - th0[:, :, None, None]
    return zz, r, theta


def validate_origin(origin: np.ndarray, lenz: int, lent: int):
    """Validates the shape of the origin array."""

    msg = (
        f"Origin must be a 1D array of length 2, a 2D array of shape ({lenz}, 2) or"
        f"({lent}, 2) or a 3D array of shape ({lent}, {lenz}, 2)"
    )
    if origin.shape == (2,):
        return np.tile(origin, (lent, lenz, 1))
    elif origin.shape == (lenz, 2):
        return np.tile(origin, (lent, 1, 1))
    elif origin.shape == (lent, 2):
        return np.tile(origin, (lenz, 1, 1)).transpose((1, 0, 2))
    elif origin.shape == (lent, lenz, 2):
        return origin
    else:
        raise ValueError(msg)


def validate_theta0(theta0: Union[float, np.ndarray], lenz: int, lent: int):
    """Validates the shape of the theta0 array."""
    if isinstance(theta0, np.ndarray):
        msg = (
            f"If an array, theta0 must have shape ({lenz},), ({lent},) or "
            f"({lent}, {lenz})"
        )
        if theta0.shape == (lenz,):
            return np.tile(theta0, (lent, 1))
        elif theta0.shape == (lent,):
            return np.tile(theta0, (lenz, 1)).T
        elif theta0.shape == (lent, lenz):
            return theta0
        else:
            raise ValueError(msg)
    else:
        return np.full((lent, lenz), theta0)


def masked_reduction(data: np.ndarray, *masks: np.ndarray, axis: tuple) -> np.ndarray:
    """Reduces array in cylindrical coordinates to the non-zero elements in the masks.

    The masks must have the same shape than the input array.

    In the case of interest of having two masks, the radial and angular masks,
    these define a region of interest in 2D space in the shape of
    a torus, with Na angular segments and Nr radial segments. The rest of the
    space is not relevant. This means that a large 2D array with data can be reduced to
    a much smaller and easy to handle (Nr, Na) array, where the value of each entry
    is the mean values of the pixels in the regions defined by both masks.

    Examples:
        This example reduces the (8, 8) angular array (serving also as input data) to an
        array of shape (Nr=2, Na=4) where each element is the average of the input
        pixels in the 8 regions (2x4) defined by the angular and radial masks. There is
        no radial dependency in the reduced array (all rows are the same) because there
        is no radial dependency in the input array either.
        >>> radial = np.array([
        ...     [0, 0, 0, 0, 0, 0, 0, 0],
        ...     [0, 2, 2, 2, 2, 2, 2, 0],
        ...     [0, 2, 1, 1, 1, 1, 2, 0],
        ...     [0, 2, 1, 0, 0, 1, 2, 0],
        ...     [0, 2, 1, 0, 0, 1, 2, 0],
        ...     [0, 2, 1, 1, 1, 1, 2, 0],
        ...     [0, 2, 2, 2, 2, 2, 2, 0],
        ...     [0, 0, 0, 0, 0, 0, 0, 0],
        ... ])
        >>> angular = np.array([
        ...     [1, 1, 1, 1, 4, 4, 4, 4],
        ...     [1, 1, 1, 1, 4, 4, 4, 4],
        ...     [1, 1, 1, 1, 4, 4, 4, 4],
        ...     [1, 1, 1, 1, 4, 4, 4, 4],
        ...     [2, 2, 2, 2, 3, 3, 3, 3],
        ...     [2, 2, 2, 2, 3, 3, 3, 3],
        ...     [2, 2, 2, 2, 3, 3, 3, 3],
        ...     [2, 2, 2, 2, 3, 3, 3, 3],
        ... ])
        >>> reduced = masked_reduction(angular, radial, angular, axis=(0, 1))
        >>> print(reduced)
        [[1 2 3 4]
         [1 2 3 4]]

        We can repeat this using the radial mask as input. In this case, there is no
        angular dependency, as expected.
        >>> reduced = masked_reduction(radial, radial, angular, axis=(0, 1))
        >>> print(reduced)
        [[1 1 1 1]
         [2 2 2 2]]

        In general, if there are no symmetries in the input array, all elements of the
        reduced array will be different.
        >>> np.random.seed(12345)
        >>> reduced = masked_reduction(np.random.rand(*radial.shape), radial, angular,
        ...     axis=(0, 1))
        >>> print(reduced)
        [[0.89411584 0.46596842 0.17654222 0.51028107]
         [0.79289128 0.28042882 0.73393468 0.18159693]]

    The reduced array has the dimensions defined in axis removed and the extra
    dimensions (one per mask) added to the end. So if data shape is (N0, N1, N2, N3, N4)
    and axis is (1, 2), then the reduced array in the case of having the above radial
    and angular masks will have shape (N0, N3, N4, Nr, Na)
    """
    from numpy.ma import MaskedArray

    assert len(masks) > 0
    assert all([data.shape == m.shape for m in masks])
    assert len(data.shape) > max(axis)

    indices = [set(m.flatten()) - {0} for m in masks]
    shape = [s for i, s in enumerate(data.shape) if i not in axis] + [
        len(idx) for idx in indices
    ]
    reduced = np.zeros(shape, dtype=data.dtype)
    for idx in product(*indices):
        elements = tuple([...] + [k - 1 for k in idx])
        condition = ~np.all([masks[i] == k for i, k in enumerate(idx)], axis=0)
        reduced[elements] = MaskedArray(data, condition).mean(axis=axis).data
    return reduced


def masked_expansion(
    reduced: np.ndarray, *masks: np.ndarray, axis: tuple
) -> np.ndarray:
    """Transforms a reduced array into a full array with the same shape as the masks.

    This function, partially opposite to `masked_reduction`, will recover a full size
    array with the same shape as the masks and with the masked elements equal to the
    corresponding entries of the reduced array. All other elements are masked.

        >>> radial = np.array([
        ...     [0, 0, 0, 0, 0, 0, 0, 0],
        ...     [0, 2, 2, 2, 2, 2, 2, 0],
        ...     [0, 2, 1, 1, 1, 1, 2, 0],
        ...     [0, 2, 1, 0, 0, 1, 2, 0],
        ...     [0, 2, 1, 0, 0, 1, 2, 0],
        ...     [0, 2, 1, 1, 1, 1, 2, 0],
        ...     [0, 2, 2, 2, 2, 2, 2, 0],
        ...     [0, 0, 0, 0, 0, 0, 0, 0],
        ... ])
        >>> angular = np.array([
        ...     [1, 1, 1, 1, 4, 4, 4, 4],
        ...     [1, 1, 1, 1, 4, 4, 4, 4],
        ...     [1, 1, 1, 1, 4, 4, 4, 4],
        ...     [1, 1, 1, 1, 4, 4, 4, 4],
        ...     [2, 2, 2, 2, 3, 3, 3, 3],
        ...     [2, 2, 2, 2, 3, 3, 3, 3],
        ...     [2, 2, 2, 2, 3, 3, 3, 3],
        ...     [2, 2, 2, 2, 3, 3, 3, 3],
        ... ])
        >>> reduced = masked_reduction(angular, radial, angular, axis=(0, 1))
        >>> print(reduced)
        [[1 2 3 4]
         [1 2 3 4]]

        Now we "recover" the original full size array:
        >>> data = masked_expansion(reduced, radial, angular, axis=(0, 1))
        >>> print(data)
        [[-- -- -- -- -- -- -- --]
         [-- 1 1 1 4 4 4 --]
         [-- 1 1 1 4 4 4 --]
         [-- 1 1 -- -- 4 4 --]
         [-- 2 2 -- -- 3 3 --]
         [-- 2 2 2 3 3 3 --]
         [-- 2 2 2 3 3 3 --]
         [-- -- -- -- -- -- -- --]]
    """
    from numpy.ma import MaskedArray

    assert len(masks) > 0
    assert all([masks[0].shape == m.shape for m in masks[1:]])
    assert len(masks[0].shape) > max(axis)

    indices = reduced.shape[-len(masks) :]
    data = np.zeros_like(masks[0], dtype=reduced.dtype)

    for idx in product(*[range(i) for i in indices]):
        condition = np.all([masks[i] == k + 1 for i, k in enumerate(idx)], axis=0)
        values = reduced[(...,) + idx]
        for ax in axis:
            values = np.expand_dims(values, ax)
        data[condition] = (values * condition)[condition]

    return MaskedArray(data, ~np.all(masks, axis=0))


def prepare_coordinates(
    data: DICOMReaderBase, zero_angle: Dict[str, np.ndarray], datasets: Tuple[str, ...]
):
    """Prepares the coordinate arrays to calculate the strain.

    Returns:
        - time - Array with shape (time, z)
        - space - Array with shape (component, z, r, theta)
    """
    lenz = len(datasets)
    lent, lenx, leny = data.mag(datasets[0]).shape

    z_location = np.zeros(lenz)
    t_interval = np.zeros(lenz)
    theta0 = np.zeros((lent, lenz))
    origin = np.zeros((lent, lenz, 2))

    for i, d in enumerate(datasets):
        z_location[i] = data.slice_loc(d)
        t_interval[i] = data.time_interval(d)

        theta0[:, i] = find_theta0(zero_angle[d])
        origin[:, i, :] = zero_angle[d][:, :, 1]

    px_size = data.pixel_size(datasets[0])
    z = z_location - z_location[0]
    x = np.linspace(0, px_size * lenx, lenx, endpoint=False)
    y = np.linspace(0, px_size * leny, leny, endpoint=False)
    zz, r, theta = cylcoords(z, x, y, origin, theta0, lent)
    time = np.linspace(0, t_interval * lent, lent, endpoint=False)

    return time, np.array([zz, r, theta])


def prepare_masks_and_velocities(
    masks: Dict[str, Dict[str, np.ndarray]],
    datasets: Tuple[str, ...],
    nrad: int = 3,
    nang: int = 24,
    background: str = "Estimated",
):
    """Prepare the masks and cylindrical velocities to be used in strain calculation.

    Returns:
        - velocities - Array with shape (component, time, z, r, theta)
        - radial masks - Array with shape (time, z, r, theta)
        - angular masks - Array with shape (time, z, r, theta)
    """
    vkey = f"cylindrical - {background}"
    rkey = f"radial x{nrad} - {background}"
    akey = f"angular x{nang} - {background}"

    vel = []
    radial = []
    angular = []

    for i, d in enumerate(datasets):
        vel.append(masks[d][vkey])
        radial.append(masks[d][rkey])
        angular.append(masks[d][akey])

    vel = np.array(vel).transpose((1, 2, 0, 3, 4))
    radial = np.array(radial).transpose((1, 0, 2, 3))
    angular = np.array(angular).transpose((1, 0, 2, 3))

    return vel, radial, angular


def calculate_inplane_strain(
    data: StrainMapData,
    angular_regions: Sequence[int] = (),
    radial_regions: Sequence[int] = (),
    bg: str = "Estimated",
    sign_reversal: Tuple[bool, ...] = (False, False, False),
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


def inplane_strain_rate(
    velocities: np.ndarray,
    component_axis: int = 0,
    spline: Optional[Callable] = None,
    origin: Sequence[float] = (0, 0),
    **kwargs,
):
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


def differentiate(reduced_vel, reduced_space, time) -> np.ndarray:
    """Calculate the strain out of the velocity data."""


def calculate_regional_strain(strain, masks, datasets) -> Dict:
    """Calculate the regional strains (1D curves)."""
