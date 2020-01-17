from pytest import approx, raises, fixture


@fixture
def shape():
    return 50, 512, 512


@fixture
def ang_mask(shape):
    from strainmap.models.contour_mask import angular_segments
    import numpy as np

    return np.tile(angular_segments(nsegments=24, shape=shape[1:]), (shape[0], 1, 1))


@fixture
def rad_mask(shape):
    from strainmap.models.contour_mask import Contour, radial_segments
    import numpy as np

    outer = Contour.circle(shape=shape[1:], radius=shape[1] / 4)
    inner = Contour.circle(
        shape=shape[1:], center=(shape[1] / 2, shape[2] / 2), radius=shape[1] / 4.5
    )
    return np.tile(radial_segments(outer, inner, nr=3), (shape[0], 1, 1))


def test_cartcoords():
    from strainmap.models.strain import cartcoords
    import numpy as np

    shape = tuple(np.random.randint(2, 6, 3))
    size = np.random.rand(3) * 10
    expected = cartcoords(shape, *size)
    for i, v in enumerate(expected):
        assert len(v) == shape[i]
        assert max(v) == size[i]

    zsize = np.random.rand(shape[0])
    expected = cartcoords(shape, zsize, size[1], size[2])
    assert len(expected[0]) == len(zsize) == shape[0]
    assert expected[0] == approx(zsize - zsize[0])


def test_cylcoords():
    from strainmap.models.strain import cylcoords, cartcoords
    import numpy as np

    shape = tuple(np.random.randint(2, 6, 3))
    size = np.random.rand(3) * 10
    z, x, y = cartcoords(shape, *size)
    origin = np.array([x.mean(), y.mean()])
    theta0 = np.random.rand() * np.pi * 2

    zz, r, theta = cylcoords(z, x, y, origin, theta0)
    assert zz - z[:, None, None] == approx(np.zeros_like(zz))
    assert r * np.cos(theta + theta0) + origin[-2] == approx(
        np.tile(x, (shape[0], shape[2], 1)).transpose((0, 2, 1))
    )
    assert r * np.sin(theta + theta0) + origin[-1] == approx(
        np.tile(y, (shape[0], shape[1], 1))
    )


def test_validate_origin():
    from strainmap.models.strain import validate_origin
    import numpy as np

    origin = np.random.rand(2)
    lenz = np.random.randint(10)
    expected = np.tile(origin, (lenz, 1))

    assert validate_origin(origin, lenz) == approx(expected)
    assert validate_origin(expected, lenz) == approx(expected)
    with raises(AssertionError):
        validate_origin(origin[1:], lenz)
    with raises(AssertionError):
        validate_origin(expected, 100)
    with raises(ValueError):
        validate_origin(np.zeros((1, 1, 1)), lenz)


def test_validate_theta0():
    from strainmap.models.strain import validate_theta0
    import numpy as np

    theta0 = np.random.rand()
    lenz = np.random.randint(10)
    expected = np.array([theta0] * lenz)

    assert validate_theta0(theta0, lenz) == approx(expected)
    assert validate_theta0(expected, lenz) == approx(expected)
    with raises(AssertionError):
        validate_theta0(expected, 100)


def test_reduce_array(ang_mask, rad_mask):
    from strainmap.models.strain import reduce_array

    udata = reduce_array(ang_mask, ang_mask, rad_mask)
    assert set(udata.flatten()) == set(ang_mask.flatten())
