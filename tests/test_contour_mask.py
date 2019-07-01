from pytest import approx


def test_cart2pol():
    import numpy as np
    from strainmap.models.contour_mask import cart2pol

    xy = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    rho = np.array([1, 1, 1, 1])
    phi = np.array([0, np.pi / 2, np.pi, -np.pi / 2])
    actual = cart2pol(xy)

    assert actual.rho == approx(rho)
    assert actual["phi"] == approx(phi)


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


def test_xy2d_to_xy():
    import numpy as np
    from strainmap.models.contour_mask import Circle, xy2d_to_xy

    c = Circle((250, 250), radius=60)

    xy = xy2d_to_xy(c.xy2d)
    xidx = xy[:, 1].round().astype(int)
    yidx = xy[:, 0].round().astype(int)

    assert np.all(c.xy2d[xidx, yidx] == 1)


def test_contour():
    import numpy as np
    from strainmap.models.contour_mask import Contour

    center = np.array([1, 0])
    xy = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    rho = np.array([1, 1, 1, 1])
    phi = np.array([0, np.pi / 2, np.pi, -np.pi / 2])

    c = Contour(xy)

    assert c.xy == approx(xy)
    assert c.centroid == approx(np.array([0, 0]))
    assert c.polar['rho'] == approx(rho)
    assert c.polar['phi'] == approx(phi)

    c.xy = c.xy + center
    assert c.centroid == approx(center)
    assert c.polar.rho == approx(rho)
    assert c.polar.phi == approx(phi)


def test_contour_from_xy2d():
    import numpy as np
    from strainmap.models.contour_mask import Circle, Contour

    c = Circle((250, 250), radius=60)
    c2 = Contour(c.xy2d)

    assert np.all(c.xy2d == c2.xy2d)


def test_contour_xy2d():
    import numpy as np
    from strainmap.models.contour_mask import Contour

    xy = np.array([[0, 0], [0, 2], [2, 2], [2, 0]])
    shape = (3, 3)

    c = Contour(xy, shape)

    expected = np.ones(shape)
    expected[1, 1] = 0

    assert c.xy2d == approx(expected)


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
    from strainmap.models.contour_mask import Circle

    center = np.array([3, 0])
    c = Circle(center, radius=10)

    assert c.centroid == approx(center)

    xy = c.xy
    points = c.points // 2
    c.polar = xy[:points]
    assert c.points == points
    assert len(np.unique(c.polar[:, 0])) == 1


def test_spline():
    import numpy as np
    from strainmap.models.contour_mask import Spline

    x = np.random.randint(100, 400, 6)
    y = np.random.randint(200, 300, 6)

    c = Spline([x, y], points=12)

    assert len(c.xy) == 12
    assert c.xy[0] == approx(c.xy[-1])
    assert c.xy[0] == approx(np.array([x, y]).T[0])


def test_mask():
    from strainmap.models.contour_mask import Mask, Circle

    c1 = Circle()
    c2 = Circle()

    m = Mask(c1, c2)
    assert not m.mask.any()

    m = Mask(c1)
    assert m.mask == approx(c1.mask)


def test_mask_sector_mask():
    import numpy as np
    from strainmap.models.contour_mask import Mask, Circle

    c1 = Circle()
    m = Mask(c1)
    smasks = m.sector_mask()

    sum_smasks = np.array([smasks[i].sum() for i in smasks.keys()]).sum()
    assert sum_smasks == approx(m.mask.sum())
