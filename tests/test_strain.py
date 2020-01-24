from pytest import approx, raises


def test_cartcoords():
    from strainmap.models.strain import cartcoords
    import numpy as np

    shape = tuple(np.random.randint(2, 6, 3))
    size = np.random.rand(3) * 10
    expected = cartcoords(shape, *size)
    for i, v in enumerate(expected):
        assert len(v) == shape[i]
        assert np.diff(v) == approx(size[i])

    zsize = np.random.rand(shape[0])
    expected = cartcoords(shape, zsize, size[1], size[2])
    assert len(expected[0]) == len(zsize) == shape[0]
    assert expected[0] == approx(zsize - zsize[0])


def test_cylcoords():
    from strainmap.models.strain import cylcoords, cartcoords
    import numpy as np

    shape = tuple(np.random.randint(2, 6, 3))
    shape = shape + (shape[-1],)
    size = np.random.rand(3) * 10
    z, x, y = cartcoords(shape[1:], *size)
    origin = np.array([x.mean(), y.mean()])
    theta0 = np.random.rand() * np.pi * 2

    zz, r, theta = cylcoords(z, x, y, origin, theta0, shape[0])
    assert zz - z[None, :, None, None] == approx(np.zeros_like(zz))
    assert r * np.cos(theta + theta0) + origin[-2] == approx(
        np.broadcast_to(x, shape).transpose((0, 1, 3, 2))
    )
    assert r * np.sin(theta + theta0) + origin[-1] == approx(np.broadcast_to(y, shape))


def test_validate_origin():
    from strainmap.models.strain import validate_origin
    import numpy as np

    origin = np.random.rand(2)
    lenz = np.random.randint(10)
    lent = np.random.randint(10)
    expected = np.tile(origin, (lent, lenz, 1))

    assert validate_origin(origin, lenz, lent) == approx(expected)
    assert validate_origin(expected, lenz, lent) == approx(expected)
    with raises(ValueError):
        validate_origin(origin[1:], lenz, lent)
    with raises(ValueError):
        validate_origin(expected, 100, lent)
    with raises(ValueError):
        validate_origin(np.zeros((1, 1, 1)), lenz, lent)


def test_validate_theta0():
    from strainmap.models.strain import validate_theta0
    import numpy as np

    theta0 = np.random.rand()
    lenz = np.random.randint(10)
    lent = np.random.randint(10)
    expected = np.full((lent, lenz), theta0)

    assert validate_theta0(theta0, lenz, lent) == approx(expected)
    assert validate_theta0(np.full(lenz, theta0), lenz, lent) == approx(expected)
    assert validate_theta0(np.full(lent, theta0), lenz, lent) == approx(expected)
    assert validate_theta0(expected, lenz, lent) == approx(expected)
    with raises(ValueError):
        validate_theta0(expected, 100, lent)
