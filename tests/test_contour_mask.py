from pytest import approx


def test_cart2pol():
    import numpy as np
    from strainmap.models.contour_mask import cart2pol

    xy = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    r = np.array([1, 1, 1, 1])
    theta = np.array([0, np.pi / 2, np.pi, -np.pi / 2])
    actual = cart2pol(xy)

    assert actual.r == approx(r)
    assert actual["theta"] == approx(theta)


def test_pol2cart():
    import numpy as np
    from strainmap.models.contour_mask import pol2cart

    polar = np.array([[1, 0], [1, np.pi / 2], [1, np.pi], [1, -np.pi / 2]])
    expected = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    actual = pol2cart(polar)

    assert expected == approx(actual)


def test_dilate():
    import numpy as np
    from strainmap.models.contour_mask import Contour

    xy = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])

    c = Contour(xy)
    c2 = c.dilate(2)

    assert 2 * xy == approx(c2.xy)

    c3 = c.dilate(s=1)
    expected = np.array([[2, 0], [0, 2], [-2, 0], [0, -2]])
    assert expected == approx(c3.xy)


def test_image_to_coordinates():
    import numpy as np
    from strainmap.models.contour_mask import Contour, image_to_coordinates

    c = Contour.circle((250, 250), radius=60)

    xy = image_to_coordinates(c.image)

    assert np.all(c.image[xy[:, 0], xy[:, 1]] == 1)


def test_contour():
    import numpy as np
    from strainmap.models.contour_mask import Contour

    center = np.array([1, 0])
    xy = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    r = np.array([1, 1, 1, 1])
    theta = np.array([0, np.pi / 2, np.pi, -np.pi / 2])

    c = Contour(xy)

    assert c.xy == approx(xy)
    assert c.centroid == approx(np.array([0, 0]))
    assert c.polar["r"] == approx(r)
    assert c.polar["theta"] == approx(theta)

    c.xy = c.xy + center
    assert c.centroid == approx(center)
    assert c.polar.r == approx(r)
    assert c.polar.theta == approx(theta)


def test_contour_from_image():
    import numpy as np
    from strainmap.models.contour_mask import Contour

    c = Contour.circle((250, 250), radius=60)
    c2 = Contour(c.image)

    assert np.all(c.image == c2.image)


def test_contour_image():
    import numpy as np
    from strainmap.models.contour_mask import Contour

    xy = np.array([[0, 0], [0, 2], [2, 2], [2, 0]])
    shape = (3, 3)

    c = Contour(xy, shape)

    expected = np.ones(shape)
    expected[1, 1] = 0

    assert c.image == approx(expected)


def test_contour_mask():
    import numpy as np
    from strainmap.models.contour_mask import Contour

    xy = np.array([[0, 0], [0, 2], [2, 2], [2, 0]])
    shape = (3, 3)

    c = Contour(xy, shape)

    expected = np.ones(shape)

    assert c.mask == approx(expected)


def test_circle():
    import numpy as np
    from scipy.linalg import norm
    from strainmap.models.contour_mask import Contour

    radius = 10
    c = Contour.circle((3, 0), radius=radius, points=100000)

    assert c.centroid == approx((3, 0), rel=1e-2)
    assert norm(c.xy - c.centroid, axis=1) == approx(
        np.ones(c.points) * radius, rel=1e-1
    )


def test_spline():
    import numpy as np
    from strainmap.models.contour_mask import Contour

    x = np.random.randint(100, 400, 6)
    y = np.random.randint(200, 300, 6)

    c = Contour.spline([x, y], points=12)

    assert len(c.xy) == 12
    assert c.xy[0] == approx(c.xy[-1])
    assert c.xy[0] == approx(np.array([x, y]).T[0])


def test_mask():
    from strainmap.models.contour_mask import Contour, contour_diff

    c1 = Contour.circle()
    c2 = Contour.circle()

    mask = contour_diff(c1, c2)
    assert not mask.any()


def test_masked_means():
    from numpy import zeros
    from numpy.random import randint, random
    from strainmap.models.contour_mask import masked_means

    # constructs cartesian velocities with known means
    N = 3
    cartvel = zeros((3, 5, 512, 512))
    labels = randint(0, N, (512, 512))

    meanvel = random((N, 3, cartvel.shape[1]))
    for l in range(N):
        for t in range(cartvel.shape[1]):
            view = cartvel[:, t, :, :]
            view[0][labels == l] = meanvel[l, 0, t]
            view[1][labels == l] = meanvel[l, 1, t]
            view[2][labels == l] = meanvel[l, 2, t]

    actual = masked_means(cartvel, labels, axes=(2, 3))
    assert actual == approx(meanvel[1:, :, :])
