import numpy as np
from scipy import ndimage, interpolate
import copy
from typing import Tuple, Optional, Sequence


def cart2pol(cart: np.ndarray) -> np.ndarray:
    x, y, = cart[:, 0], cart[:, 1]
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return np.array([rho, phi]).T


def pol2cart(polar: np.ndarray) -> np.ndarray:
    rho, phi = polar[:, 0], polar[:, 1]
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return np.array([x, y]).T


class Contour(object):
    def __init__(self, xy: np.ndarray, shape: Tuple[int, int] = (512, 512)):

        self._xy = xy
        self.shape = shape

    @property
    def points(self):
        return len(self._xy)

    @property
    def centroid(self):
        return np.mean(self._xy, axis=0)

    @property
    def xy(self):
        return self._xy

    @xy.setter
    def xy(self, xy):
        self._xy = xy

    @property
    def polar(self):
        return cart2pol(self.xy - self.centroid)

    @polar.setter
    def polar(self, polar):
        self.xy = self.centroid + pol2cart(polar)

    @property
    def xy2d(self):
        """ Returns the XY data as 1-valued pixels in a 0-valued MxN array.

        To ensure a closed contour, the curve is first finely interpolated. """
        idx = np.linspace(0, self.points + 1, self.shape[0] * self.shape[1])
        xy = np.vstack((self.xy, self.xy[0, :]))

        xy = (
            interpolate.CubicSpline(range(self.points + 1), xy, bc_type="periodic")(idx)
            .round()
            .astype(int)
        )

        result = np.zeros(self.shape, dtype=int)
        result[xy[:, 1], xy[:, 0]] = 1

        return result

    @property
    def mask(self):
        """ Returns a 0-valued MxN array with 1-valued pixels inside the xy contour. """
        return ndimage.morphology.binary_fill_holes(self.xy2d).astype(int)

    def dilate(self, p: float = 1) -> "Contour":
        """ Expands or contracts a contour. """
        return dilate(self, p)


class Circle(Contour):
    def __init__(
        self,
        center: Tuple[int, int] = (256, 256),
        edge: Tuple[int, int] = (256, 306),
        points: int = 360,
        radius: Optional[int] = None,
        **kwargs,
    ):
        center = np.array(center)
        radius = radius if radius else np.linalg.norm(center - np.array(edge))
        polar = np.ones((points, 2))
        polar[:, 0] *= radius
        polar[:, 1] = np.linspace(0, 2 * np.pi, points)
        xy = pol2cart(polar) + center

        super().__init__(xy, **kwargs)

        self._center = center
        self._radius = radius
        self._polar = polar

    @property
    def centroid(self):
        return self._center

    @property
    def xy(self):
        return self._xy

    @xy.setter
    def xy(self, xy):
        pass

    @property
    def polar(self):
        return self._polar

    @polar.setter
    def polar(self, polar):
        self._radius = np.mean(polar[:, 0])
        self._polar = np.ones_like(polar)
        self._polar[:, 0] *= self._radius
        self._polar[:, 1] = polar[:, 1]
        self._xy = self.centroid + pol2cart(self._polar)


class Spline(Contour):
    def __init__(
        self, nodes: Sequence[np.ndarray], points: int = 360, k: int = 3, **kwargs
    ):
        x = np.r_[nodes[0], nodes[0][0]]
        y = np.r_[nodes[1], nodes[1][0]]
        tck, u = interpolate.splprep([x, y], s=0, per=True, k=k)[:2]
        xy = np.array(interpolate.splev(np.linspace(0, 1, points), tck)).T

        super().__init__(xy, **kwargs)

        self.nodes = nodes
        self.k = k


def dilate(contour: Contour, p: float = 1) -> Contour:
    """ Expands (or contracts y p<1) a contour with respect its centroid. """
    pp = np.array([max(p, 0), 1.0])
    result = copy.copy(contour)
    result.polar = contour.polar * pp

    return result


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # c = Circle(radius=60, points=15)
    c2 = Circle(radius=40, points=360)

    x = np.random.randint(100, 400, 6)
    y = np.random.randint(200, 300, 6)

    c = Spline([x, y])

    plt.scatter(x, y)
    plt.imshow(c.mask - c2.mask)
    plt.plot(c.xy[:, 0], c.xy[:, 1], "r")
    plt.show()
