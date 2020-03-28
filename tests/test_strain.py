import numpy as np
from pytest import approx, mark, raises
from unittest.mock import MagicMock
from typing import Dict


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


def test_prepare_coordinates():
    from strainmap.models.readers import DICOM
    from strainmap.models.strain import prepare_coordinates
    import numpy as np

    datasets = ("Venus", "Earth", "Mars", "Jupyter")
    lenz = len(datasets)
    zval = np.arange(lenz) + np.random.randint(5, 10)
    tval = np.random.random(lenz)
    px_size = np.random.random()
    lent, lenx, leny = np.random.randint(5, 10, 3)

    data = DICOM(dict())
    data.slice_loc = MagicMock(side_effect=zval)
    data.time_interval = MagicMock(side_effect=tval)
    data.pixel_size = MagicMock(return_value=px_size)
    data.mag = MagicMock(return_value=np.zeros((lent, lenx, leny)))
    zero_angle = {k: np.zeros((lent, 2, 2)) for k in datasets}

    time, space = prepare_coordinates(data, zero_angle, datasets)

    # Checks for the time array
    assert time.shape == (lenz,)
    assert time == approx(tval)

    # Checks for the space
    assert space.shape == (3, lent, lenz, lenx, leny)
    for i in range(space.shape[0]):
        assert space[i].min() == 0

    for i, val in enumerate([0, 1, 0, 0]):
        assert np.gradient(space[0], axis=i) == approx(val)

    r, theta = space[1], space[2]
    for i, val in enumerate([0, 0, px_size, 0]):
        assert np.gradient(r * np.cos(theta), axis=i) == approx(val)

    for i, val in enumerate([0, 0, 0, px_size]):
        assert np.gradient(r * np.sin(theta), axis=i) == approx(val)


def test_prepare_masks_and_velocities():
    from strainmap.models.strain import prepare_masks_and_velocities
    import numpy as np

    background = "Estimated"
    nrad = 3
    nang = 24
    masks: Dict[str, Dict] = {"Venus": {}, "Earth": {}, "Mars": {}, "Jupyter": {}}
    keys = [
        f"cylindrical - {background}",
        f"radial x{nrad} - {background}",
        f"angular x{nang} - {background}",
    ]
    shape = tuple(np.random.randint(5, 10, 3))
    shapes = [(3,) + shape, shape, shape]

    for d in masks.keys():
        masks[d] = {k: np.random.rand(*s) for k, s in zip(keys, shapes)}

    vel, radial, angular = prepare_masks_and_velocities(masks, tuple(masks.keys()))

    assert vel.shape == (3, shape[0], len(masks), shape[1], shape[2])
    assert radial.shape == angular.shape == (shape[0], len(masks), shape[1], shape[2])


@mark.parametrize("period", [None, True])
def test_finite_differences(period):
    from strainmap.models.strain import finite_differences
    import numpy as np

    dims = np.random.randint(2, 5)
    shape = tuple(np.random.randint(3, 20, dims))
    axis = np.random.randint(dims)
    expected = np.ones(shape)
    f = x = np.cumsum(expected, axis)

    if period:
        period = shape[axis]
        expected.swapaxes(0, axis)[[0, -1]] = (2 - period) / 2

    df = finite_differences(f, x, axis=axis, period=period)
    assert df == approx(expected)


def test_inplane_strain(data_with_velocities):
    from strainmap.models.strain import calculate_inplane_strain

    strain = calculate_inplane_strain(
        data_with_velocities, datasets=data_with_velocities.data_files.datasets[:1]
    )
    assert set(strain) == set(data_with_velocities.data_files.datasets[:1])
    assert strain[data_with_velocities.data_files.datasets[0]].shape == (2, 3, 512, 512)


@mark.parametrize("deltaz", [1, 1.2])
def test_outofplane_strain(deltaz):
    from strainmap.models.strain import calculate_outofplane_strain
    from types import SimpleNamespace

    def vel(x, y, z, t):
        return [0, 0, x % 7 + 2 * y - 3 * z + 4 * t]

    data = SimpleNamespace()
    data.masks = {
        f"dodo{z}": {
            "cylindrical - Estimated": np.array(
                [
                    [[[vel(x, y, z, t) for y in range(100)] for x in range(100)]]
                    for t in range(10)
                ]
            ),
            "angular x6 - Estimated": np.repeat(
                np.arange(100, dtype=int)[:, None] % 7, 100, 1
            ),
        }
        for z in range(8)
    }
    data.data_files = SimpleNamespace()
    data.data_files.files = list(data.masks.keys())
    data.data_files.slice_loc = lambda x: {
        k: deltaz * i for i, k in enumerate(data.data_files.files)
    }[x]

    strain = calculate_outofplane_strain(
        data, image_axes=(-3, -2), component_axis=-1  # type: ignore
    )
    assert strain == approx(-3 / deltaz)
