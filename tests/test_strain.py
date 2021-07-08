from pytest import approx, fixture
import xarray as xr
import numpy as np


@fixture
def mask_shape():
    return [5, 30, 30]


@fixture
def radial_da(mask_shape) -> xr.DataArray:
    """DataArray defining 3 radial regions."""
    from skimage.draw import disk

    region, frame, row, col = shape = [3] + mask_shape
    data = np.full(shape, False)

    midr = row / 2
    midc = col / 2
    for r in range(region - 1, -1, -1):
        rr, cc = disk((midr, midc), row / (5 - r), shape=(row, col))
        data[r, :, rr, cc] = True
        if r < region - 1:
            data[r + 1, :, rr, cc] = False

    rr, cc = disk((midr, midc), row / 8, shape=(row, col))
    data[0, :, rr, cc] = False

    return xr.DataArray(data, dims=["region", "frame", "row", "col"])


@fixture
def angular_da(mask_shape) -> xr.DataArray:
    """DataArray defining 4 angular regions"""
    from skimage.draw import rectangle

    region, frame, row, col = shape = [4] + mask_shape
    data = np.full(shape, False)

    mid = int(row / 2)
    for i in range(2):
        for j in range(2):
            rr, cc = rectangle((i * mid, j * mid), extent=(mid, mid), shape=(row, col))
            data[int(i + 2 * j), :, rr, cc] = True

    return xr.DataArray(data, dims=["region", "frame", "row", "col"])


def test_masked_reduction(radial_da, angular_da):
    from strainmap.models.strain import masked_reduction

    frame = angular_da.sizes["frame"]

    # An input array with radial symmetry in the row/col plane should return an array
    #  with no angular dependence
    input_rad = (
        (radial_da * (radial_da.region + 1)).sum("region").expand_dims("beyond", 0)
    )
    expected = np.tile(
        np.arange(1, radial_da.sizes["region"] + 1),
        (angular_da.sizes["region"], frame, 1),
    ).transpose([1, 2, 0])[None, ...]
    actual = masked_reduction(input_rad, radial_da, angular_da)
    np.testing.assert_equal(actual.data, expected)
    assert all([d in actual.dims for d in input_rad.dims if d not in ["row", "col"]])
    assert all([d in actual.dims for d in ["radius", "angle"]])

    # Likewise, if input has angular symmetry, the return value should have no radial
    #  dependence
    input_ang = (
        (angular_da * (angular_da.region + 1)).sum("region").expand_dims("beyond", 0)
    )
    expected = np.tile(
        np.arange(1, angular_da.sizes["region"] + 1),
        (radial_da.sizes["region"], frame, 1),
    ).transpose([1, 0, 2])[None, ...]
    actual = masked_reduction(input_ang, radial_da, angular_da)
    np.testing.assert_equal(actual.data, expected)
    assert all([d in actual.dims for d in input_ang.dims if d not in ["row", "col"]])
    assert all([d in actual.dims for d in ["radius", "angle"]])


def test_resample():
    from strainmap.models.strain import resample_interval
    import numpy as np

    frames = 50
    interval = np.random.randint(10, 15, 3) / 100
    nframes = np.ceil(interval / min(interval) * frames).astype(int)
    t = np.linspace(0, np.ones_like(interval), frames, endpoint=False).T
    disp = np.moveaxis(np.sin(np.pi * t)[..., None, None, None], (0, 1), (2, 1))

    exframes = nframes.max() - nframes
    expected = (np.sin(np.pi * np.linspace(0, 1, n, endpoint=False)) for n in nframes)
    expected = np.array(
        [np.concatenate([ex, ex[:n]]) for n, ex in zip(exframes, expected)]
    ).T
    actual = resample_interval(disp, interval)

    assert (
        actual.shape == np.concatenate([disp, disp], axis=1)[:, : nframes.max()].shape
    )
    assert np.squeeze(actual) == approx(expected, abs=1e-1)


def test_unresample():
    from strainmap.models.strain import unresample_interval
    import numpy as np

    frames = 50
    interval = np.random.randint(10, 15, 3) / 100
    nframes = np.ceil(interval / min(interval) * frames).astype(int)

    exframes = nframes.max() - nframes
    disp = (np.sin(np.pi * np.linspace(0, 1, n, endpoint=False)) for n in nframes)
    disp = np.moveaxis(
        np.array([np.concatenate([ex, ex[:n]]) for n, ex in zip(exframes, disp)])[
            ..., None, None, None
        ],
        (0, 1),
        (2, 1),
    )

    t = np.linspace(0, np.ones_like(interval), frames, endpoint=False)
    expected = np.sin(np.pi * t)
    actual = unresample_interval(disp, interval, frames)

    assert actual.shape == disp[:, :frames].shape
    assert np.squeeze(actual) == approx(expected, abs=1e-1)
