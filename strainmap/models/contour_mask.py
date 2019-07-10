from __future__ import annotations

import copy
from typing import Optional, Sequence, Tuple, Union

import numpy as np
from scipy import interpolate, ndimage


class Contour(object):
    """Creates a contour object.

    Contours are closed paths in a plane. They can be defined as a N-2 array in
    cartesian coordinates but also as a centroid and an N-2 array of polar coordinates.
    This later representation is useful for expanding and contracting the contour
    around the centroid.
    """

    def __init__(self, xy: np.ndarray, shape: Tuple[int, int] = (512, 512)):

        if 2 not in xy.shape:
            self._xy = image_to_coordinates(xy)
            self.shape = xy.shape
        else:
            self._xy = xy
            self.shape = shape

    @property
    def points(self):
        """Number of points of the contour."""
        return len(self._xy)

    @property
    def centroid(self):
        """Centroid of the contour, calculated as the mean value of all
        points."""
        return np.mean(self._xy, axis=0)

    @property
    def xy(self):
        """The contour in cartesian coordinates."""
        return self._xy

    @xy.setter
    def xy(self, xy):
        """Sets the contour in cartesian coordinates."""
        self._xy = xy

    @property
    def polar(self):
        """The polar part of the contour, in polar coordinates."""
        return cart2pol(self.xy - self.centroid)

    @polar.setter
    def polar(self, polar):
        """Setting the polar part of the contour, keeping the same centroid."""
        self.xy = self.centroid + pol2cart(polar)

    @property
    def image(self):
        """Returns the XY data as 1-valued pixels in a 0-valued MxN array.

        To ensure a closed contour, the curve is first finely interpolated.
        """
        idx = np.linspace(0, self.points + 1, self.shape[0] * self.shape[1] * 10)
        xy = np.vstack((self.xy, self.xy[0, :]))

        xy = (
            interpolate.CubicSpline(range(self.points + 1), xy, bc_type="periodic")(idx)
            .round()
            .astype(int)
        )

        xy[:, 0] = np.clip(xy[:, 0], a_min=0, a_max=self.shape[1] - 1)
        xy[:, 1] = np.clip(xy[:, 1], a_min=0, a_max=self.shape[0] - 1)

        result = np.zeros(self.shape, dtype=int)
        result[xy[:, 1], xy[:, 0]] = 1

        return result

    @property
    def mask(self):
        """Binary image, with 1 inside and 0 outside the contour."""
        return ndimage.morphology.binary_fill_holes(self.image)

    def dilate(self, p: float = 1) -> Contour:
        """Creates an expanded (or contracted y p<1) copy of a contour."""
        return dilate(self, p)

    @staticmethod
    def circle(
        center: Optional[Tuple[int, int]] = None,
        radius: int = 50,
        points: int = 360,
        shape: Tuple[int, int] = (512, 512),
    ):
        """Circular contour."""
        if center is None:
            center = (np.array(shape) - 1) / 2
        center = np.array(center)
        polar = np.ones((points, 2))
        polar[:, 0] *= radius
        polar[:, 1] = np.linspace(0, 2 * np.pi, points)
        xy = pol2cart(polar) + center

        return Contour(xy, shape=shape)

    @staticmethod
    def spline(
        nodes: Sequence[np.ndarray], points: int = 360, order: int = 3, **kwargs
    ):
        """Contour defined by a closed spline."""

        x = np.r_[nodes[0], nodes[0][0]]
        y = np.r_[nodes[1], nodes[1][0]]
        tck, u = interpolate.splprep([x, y], s=0, per=True, k=order)[:2]
        xy = np.array(interpolate.splev(np.linspace(0, 1, points), tck)).T

        return Contour(xy, **kwargs)


def contour_diff(
    c1: Union[Contour, np.ndarray],
    c2: Union[Contour, np.ndarray],
    shape: Tuple[int, int] = (512, 512),
) -> np.ndarray:
    """Creates a mask array from the difference between two contours.

    Returns a mask that contains c1 but not c2.
    """
    if isinstance(c1, Contour):
        mask1 = c1.mask
    elif isinstance(c1, np.ndarray) and 2 in c1.shape:
        mask1 = Contour(c1, shape).mask
    elif isinstance(c1, np.ndarray):
        mask1 = c1
    else:
        raise TypeError("Inputs must be Contours or numpy arrays.")

    if isinstance(c2, Contour):
        mask2 = c2.mask
    elif isinstance(c2, np.ndarray) and 2 in c2.shape:
        mask2 = Contour(c2, shape).mask
    elif isinstance(c2, np.ndarray):
        mask2 = c2
    else:
        raise TypeError("Inputs must be Contours or numpy arrays.")

    msg = "Error: Only contours with the same shape can make a mask."
    assert mask1.shape == mask2.shape, msg
    return mask1 & (mask1 ^ mask2)


def angular_segments(
    nsegments: int = 6,
    origin: Optional[Sequence[int]] = None,
    theta0: float = 0,
    clockwise: bool = True,
    shape: Tuple[int, int] = (512, 512),
) -> np.ndarray:
    """Array defining angular segments.

    The segments are defined by a numpy array with integer values in `range(nsegments)`.

    Args:
        nsegments: Number of angular segments.
        origin: Origin the cartesian coordinates. By default, the origin is set in the
            center of the image.
        theta0: By default theta=0 is for a vector pointing right. This argument makes
            it possible the start of the segments. This transformation is applied
            **before** correcting for handedness!
        clockwise: Clockwise by default. Set to False for counter-clockwise.
        shape: size of the resulting image

    Examples:

        We can simply quarter an image as follows:

        >>> from strainmap.models.contour_mask import angular_segments
        >>> segments = angular_segments(nsegments=4, shape=(10, 10))
        >>> print(segments)
        [[2 2 2 2 2 2 3 3 3 3]
         [2 2 2 2 2 2 3 3 3 3]
         [2 2 2 2 2 2 3 3 3 3]
         [2 2 2 2 2 2 3 3 3 3]
         [2 2 2 2 2 2 3 3 3 3]
         [1 1 1 1 1 0 0 0 0 0]
         [1 1 1 1 1 0 0 0 0 0]
         [1 1 1 1 1 0 0 0 0 0]
         [1 1 1 1 1 0 0 0 0 0]
         [1 1 1 1 1 0 0 0 0 0]]
    """
    if origin is None:
        origin = np.array(shape) / 2
    x = np.arange(0, shape[0], dtype=int)[None, :] - origin[0]
    y = np.arange(0, shape[1], dtype=int)[:, None] - origin[1]

    theta = np.mod(np.arctan2(y, x) + theta0, 2 * np.pi)
    if not clockwise:
        theta = -theta

    result = np.zeros(theta.shape, dtype=int)

    steps = np.linspace(0, 2 * np.pi, nsegments + 1)
    for n in range(1, nsegments):
        result[theta > steps[n]] = n

    return result


def radial_segments(
    outer: Contour,
    inner: Contour,
    nr: int = 3,
    shape: Optional[Tuple[int, int]] = None,
    center: Optional[Tuple[float, float]] = None,
):
    """Splits difference between two contours into several radial segments.

    The two contours should create a ribbon between outer and inner. This ribbon is
    split into `nr` seperate segments. For each angle theta in the cylindrical
    coordinate system with origin `center`, the ribbon is split into `nr` segments of
    equal width.

    Args:
        outer: Contour delineating the outer boundary of the segmented ribbon.
        inner: Contour delineating the inner boundary of the segmented ribbon.
        nr: Number of segments.
        shape: Size of the returned image. Defaults to `outer.shape`.
        center: Origin of the cylindrical coordinate system used to split the ribbons.
            Defaults to `outer.center`.

    Returns:
        An image where 0 is outside the region, and 1 - nr (included) indicate
        radially separated regions.

    Examples:
        Lets try to split a ribbon defined by two circles in two:

        >>> from strainmap.models.contour_mask import Contour, radial_segments
        >>> outer = Contour.circle(shape=(15, 15), radius=6)
        >>> inner = Contour.circle(shape=(10, 10), center=(5.5, 6), radius=3)
        >>> regions = radial_segments(outer, inner, nr=3)
        >>> print(regions)
        [[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
         [0 0 0 0 0 0 0 3 0 0 0 0 0 0 0]
         [0 0 0 0 3 2 2 2 3 3 3 0 0 0 0]
         [0 0 0 3 1 1 1 1 2 2 3 3 0 0 0]
         [0 0 3 1 0 0 0 0 1 2 2 3 3 0 0]
         [0 0 2 0 0 0 0 0 0 1 2 2 3 0 0]
         [0 0 2 0 0 0 0 0 0 1 2 2 3 0 0]
         [0 0 2 0 0 0 0 0 0 1 2 2 3 3 0]
         [0 0 2 1 0 0 0 0 1 1 2 2 3 0 0]
         [0 0 3 2 1 1 1 1 1 2 2 3 3 0 0]
         [0 0 3 2 2 2 1 1 2 2 2 3 3 0 0]
         [0 0 0 3 3 2 2 2 2 3 3 3 0 0 0]
         [0 0 0 0 3 3 3 3 3 3 3 0 0 0 0]
         [0 0 0 0 0 0 0 3 0 0 0 0 0 0 0]
         [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]

        Note that it is an error if the outer and inner boundaries are swapped:

        >>> radial_segments(inner, outer, nr=3)
        Traceback (most recent call last):
            ...
        ValueError: Inner and outer boundaries cannot cross.
    """
    if center is None:
        origin = outer.centroid
    else:
        origin = np.array(center)

    if shape is None:
        shape = outer.shape

    x = np.arange(0, shape[0], dtype=int)[None, :] - origin[0]
    y = np.arange(0, shape[1], dtype=int)[:, None] - origin[1]

    thetas = np.arctan2(y, x)
    r = np.sqrt(x * x + y * y)
    polar = cart2pol(outer.xy - origin)
    outer_pol = np.interp(thetas, polar.theta, polar.r, period=2 * np.pi)
    polar = cart2pol(inner.xy - origin)
    inner_pol = np.interp(thetas, polar.theta, polar.r, period=2 * np.pi)

    # ensure numerical noise doesn't push the inner contour to the outside
    outer_pol = np.where(
        np.isclose(outer_pol, inner_pol), np.minimum(inner_pol, outer_pol), outer_pol
    )
    if (outer_pol < inner_pol).any():
        raise ValueError("Inner and outer boundaries cannot cross.")

    result = np.zeros(shape, dtype=int)
    for i in range(nr, -1, -1):
        result[(outer_pol - inner_pol) / nr * i + inner_pol >= r] = i

    return result


def cart2pol(cart: np.ndarray) -> np.recarray:
    """Transform cartesian to polar coordinates."""
    x, y, = cart[:, 0], cart[:, 1]
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)
    return np.rec.array(
        np.array(list(zip(r, theta)), dtype=[("r", float), ("theta", float)])
    )


def pol2cart(polar: Union[np.ndarray, np.recarray]) -> np.ndarray:
    """Transform polar to cartesian coordinates."""
    if hasattr(polar, "r") and hasattr(polar, "theta"):
        r, theta = polar.r, polar.theta
    elif (getattr(polar.dtype, "fields", None) is not None) and (
        {"r", "theta"} == set(polar.dtype.fields)
    ):
        r, theta = polar["r"], polar["theta"]
    elif polar.ndim == 2 and polar.shape[1] == 2:
        r, theta = polar.T
    else:
        raise ValueError("Could not make sense of polar coordinates")

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.array([x, y]).T


def dilate(contour: Contour, p: float = 1) -> Contour:
    """Creates an expanded (or contracted if p<1) copy of a contour."""
    result = copy.copy(contour)
    polar = result.polar
    polar.r *= max(p, 0)
    result.polar = polar
    return result


def image_to_coordinates(image: np.ndarray) -> np.ndarray:
    """Transforms image of contour to array of cartesian coordinates."""
    centroid = np.array(ndimage.measurements.center_of_mass(image))
    grid = np.indices(image.shape)
    x = grid[0] - centroid[0]
    y = grid[1] - centroid[1]

    theta = np.mod(np.arctan2(y, x), 2 * np.pi)[image == 1]
    r = np.sqrt(x ** 2 + y ** 2)[image == 1]
    idx = np.argsort(theta)

    return pol2cart(np.array([r[idx], theta[idx]]).T) + centroid


def bulk_component(
    data: np.ndarray, mask: Optional[np.ndarray] = None, **kwargs
) -> np.ndarray:
    """Mean across masked array.

    Args:
        data: Data from which to extract the mean component
        mask: Mask over the data.

    Examples:
        If no mask is given, then this function is equivalent to taking the mean

        >>> from numpy import sum
        >>> from numpy.random import randint
        >>> from strainmap.models.contour_mask import bulk_component
        >>> data = randint(0, 10, (10, 10))
        >>> (bulk_component(data, axis=1) == sum(data, axis=1) / data.shape[0]).all()
        True
        >>> (bulk_component(data, axis=1) == data.mean(axis=1)).all()
        True

        If a mask is given, then those values are not taken into account in the mean:

        >>> from numpy import ma
        >>> mask = randint(0, 10, data.shape) > 3
        >>> (
        ...     sum(data * mask, axis=1) / sum(mask, axis=1)
        ...     == bulk_component(data, mask, axis=1)
        ... ).all()
        True

        More specifically, it is equivalent to the mean from a masked numpy array.

        >>> bulk_component(data, mask) == ma.array(data, mask=~mask).mean()
        True
    """
    if mask is not None:
        data = np.ma.array(data, mask=~mask)
    return data.mean(**kwargs)


def cylindrical_projection(
    field: np.ndarray, origin: np.array, axis: int = 0
) -> np.ndarray:
    """Project vector field on the local basis of a cylindrical coordinate system.

    Args:
        field: 2d or 3d vector field where the (x, y, [z]) components are on dimension
            `axis`.
        origin: Origin of the cylindrical coordinate system
        axis: Axis of the field components

    Return:
        A 2d or 3d field where dimension `axis` contains (r, theta, [z])


    Examples:
        We can create a normalized centripedal field, i.e. a vector field where each
        vector points to the origin, and each vector is normalized in magnitude. Then
        the r components of the cylindrical projection should all be 1, and the theta
        components should all be zero.

        >>> import numpy as np
        >>> from strainmap.models.contour_mask import cylindrical_projection
        >>> origin = np.array([3.5, 5])
        >>> field = np.stack(
        ...   (
        ...       np.arange(0, 10, dtype=int)[None, :] + np.zeros((10, 10), dtype=int),
        ...       np.arange(0, 10, dtype=int)[:, None] + np.zeros((10, 10), dtype=int)
        ...   ), axis=2
        ... ) - origin[None, None, :]
        >>> unit_field = field / np.linalg.norm(field, axis=2)[:, :, None]
        >>> projection = cylindrical_projection(unit_field, origin=origin, axis=2)
        >>> np.allclose(projection[:, :, 0], 1)
        True
        >>> np.allclose(projection[:, :, 1], 0)
        True

        A 3d field that is centrepedal in x and y only should give us the same result,
        with the z component unchanged from the input:

        >>> z = np.random.randint(0, 10, (10, 10))
        >>> field3d = np.concatenate((unit_field, z[:, :, None]), axis=2)
        >>> proj3d = cylindrical_projection(field3d, origin=[3.5, 5, 0], axis=2)
        >>> proj3d.shape
        (10, 10, 3)
        >>> np.allclose(proj3d[:, :, 0], 1)
        True
        >>> np.allclose(proj3d[:, :, 1], 0)
        True
        >>> np.allclose(proj3d[:, :, 2], z)
        True
    """
    origin = np.array(origin)
    field = np.array(field)

    assert field.ndim == 3
    assert axis >= 0 and axis < field.ndim
    assert origin.ndim == 1 and origin.size == field.shape[axis]
    assert field.shape[axis] in (2, 3)

    if field.shape[axis] == 3:
        result = cylindrical_projection(
            np.take(field, range(2), axis=axis), origin[:2], axis=axis
        )
        return np.concatenate((result, np.take(field, (2,), axis=axis)), axis=axis)

    x = np.arange(0, field.shape[0], dtype=int)[None, :] - origin[0]
    y = np.arange(0, field.shape[1], dtype=int)[:, None] - origin[1]

    theta = np.arctan2(y, x)
    shape = list(theta.shape)
    shape.insert(axis, 1)
    r_vec = np.concatenate(
        (np.cos(theta).reshape(shape), np.sin(theta).reshape(shape)), axis=axis
    )
    theta_vec = np.concatenate(
        (-np.sin(theta).reshape(shape), np.cos(theta).reshape(shape)), axis=axis
    )
    result = np.concatenate(
        (
            np.sum(r_vec * field, axis=axis).reshape(shape),
            np.sum(theta_vec * field, axis=axis).reshape(shape),
        ),
        axis=axis,
    )
    return result


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    center = np.array([250, 250])
    c = Contour.circle(center, radius=60)

    xy = image_to_coordinates(c.image)

    assert np.all(
        c.image[xy[:, 1].round().astype(int), xy[:, 0].round().astype(int)] == 1
    )
    plt.imshow(c.image)
    plt.plot(xy[:, 0], xy[:, 1])
    plt.show()
