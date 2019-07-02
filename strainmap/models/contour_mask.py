import copy
from typing import Mapping, Optional, Sequence, Text, Tuple, Union

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
            self._xy = xy2d_to_xy(xy)
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
    def xy2d(self):
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
        return ndimage.morphology.binary_fill_holes(self.xy2d).astype(int)

    def dilate(self, p: float = 1) -> "Contour":
        """Creates an expanded (or contracted y p<1) copy of a contour."""
        return dilate(self, p)

    def to_contour(self) -> "Contour":
        """New contour preserving only the XY coordinates and shape."""
        return Contour(self.xy, shape=self.shape)

    @staticmethod
    def circle(
        center: Tuple[int, int] = (256, 256),
        edge: Tuple[int, int] = (256, 306),
        points: int = 360,
        radius: Optional[int] = None,
        **kwargs,
    ):
        """Circular contour."""
        center = np.array(center)
        radius = radius if radius else np.linalg.norm(center - np.array(edge))
        polar = np.ones((points, 2))
        polar[:, 0] *= radius
        polar[:, 1] = np.linspace(0, 2 * np.pi, points)
        xy = pol2cart(polar) + center

        return Contour(xy, **kwargs)

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


class Mask(object):
    """Creates a mask array from the difference between two contours.

    To create the combined mask, the mask associated to c2 is subtracted from c1, and
    the result clip to (0, 1). c2 can be omitted, in which case this mask is identical
    to the c1 one.

    Out of this mask, specific masks for angular sectors can be calculated.
    """

    def __init__(
        self,
        c1: Union[Contour, np.ndarray],
        c2: Union[Contour, np.ndarray, None] = None,
        shape: Tuple[int, int] = (512, 512),
    ):
        if isinstance(c1, Contour):
            mask1 = c1.mask
        elif isinstance(c1, np.ndarray) and 2 in c1.shape:
            mask1 = Contour(c1, shape).mask
        elif isinstance(c1, np.ndarray):
            mask1 = c1
        else:
            raise TypeError("Inputs must be Contours or numpy arrays.")

        if not c2:
            mask2 = 0.0 * mask1
        elif isinstance(c2, Contour):
            mask2 = c2.mask
        elif isinstance(c2, np.ndarray) and 2 in c2.shape:
            mask2 = Contour(c2, shape).mask
        elif isinstance(c2, np.ndarray):
            mask2 = c2
        else:
            raise TypeError("Inputs must be Contours or numpy arrays.")

        msg = "Error: Only contours with the same shape can make a mask."
        assert mask1.shape == mask2.shape, msg

        self.shape = mask1.shape
        self.mask = (mask1 - mask2).clip(0, 1)
        self.centroid = np.array(ndimage.measurements.center_of_mass(self.mask))

    def sector_mask(
        self,
        number: int = 6,
        zero_angle: float = 0,
        labels: Optional[Sequence[Text]] = None,
    ) -> Mapping:
        """Creates a dictionary of masks for a number of angular sectors."""
        zero_angle = np.radians(zero_angle)

        labels = labels if labels else [*range(1, number + 1)]
        number = len(labels)

        grid = np.indices(self.shape)
        x = grid[0] - self.centroid[0]
        y = grid[1] - self.centroid[1]
        theta = np.mod(np.arctan2(y, x), 2 * np.pi)

        angles = np.linspace(zero_angle, zero_angle + 2 * np.pi, number + 1)

        sectors = {}
        for i, l in enumerate(labels):
            sectors[l] = (angles[i] <= theta) * (theta < angles[i + 1]) * self.mask

        return sectors


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


def xy2d_to_xy(xd2d: np.ndarray) -> np.ndarray:
    """Transforms a contour defined as ones in a mask of zeros to a XY
    contour."""
    centroid = np.array(ndimage.measurements.center_of_mass(xd2d))
    grid = np.indices(xd2d.shape)
    x = grid[0] - centroid[0]
    y = grid[1] - centroid[1]

    theta = np.mod(np.arctan2(y, x), 2 * np.pi)[xd2d == 1]
    r = np.sqrt(x ** 2 + y ** 2)[xd2d == 1]
    idx = np.argsort(theta)

    return pol2cart(np.array([r[idx], theta[idx]]).T) + centroid


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    center = np.array([250, 250])
    c = Contour.circle(center, radius=60)

    xy = xy2d_to_xy(c.xy2d)

    assert np.all(
        c.xy2d[xy[:, 1].round().astype(int), xy[:, 0].round().astype(int)] == 1
    )
    plt.imshow(c.xy2d)
    plt.plot(xy[:, 0], xy[:, 1])
    plt.show()
