from pytest import approx, fixture
import xarray as xr
import numpy as np


@fixture
def mask_shape():
    return [5, 30, 30]


@fixture
def radial_mask(mask_shape) -> xr.DataArray:
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

    return xr.DataArray(
        data,
        dims=["region", "frame", "row", "col"],
        coords={
            "frame": np.arange(0, frame),
            "row": np.arange(0, row),
            "col": np.arange(0, col),
        },
    )


@fixture
def angular_mask(mask_shape) -> xr.DataArray:
    """DataArray defining 4 angular regions"""
    from skimage.draw import rectangle

    region, frame, row, col = shape = [4] + mask_shape
    data = np.full(shape, False)

    mid = int(row / 2)
    for i in range(2):
        for j in range(2):
            rr, cc = rectangle((i * mid, j * mid), extent=(mid, mid), shape=(row, col))
            data[int(i + 2 * j), :, rr, cc] = True

    return xr.DataArray(
        data,
        dims=["region", "frame", "row", "col"],
        coords={
            "frame": np.arange(0, frame),
            "row": np.arange(0, row),
            "col": np.arange(0, col),
        },
    )


@fixture
def expanded_radial(radial_mask):
    return (
        (radial_mask * (radial_mask.region + 1)).sum("region").expand_dims(beyond=[0])
    )


@fixture
def expanded_angular(angular_mask):
    return (
        (angular_mask * (angular_mask.region + 1)).sum("region").expand_dims(beyond=[0])
    )


@fixture
def reduced_radial(radial_mask, angular_mask):
    frame = angular_mask.sizes["frame"]

    return xr.DataArray(
        np.tile(
            np.arange(1, radial_mask.sizes["region"] + 1),
            (angular_mask.sizes["region"], frame, 1),
        ).transpose([1, 2, 0])[None, ...],
        dims=["beyond", "frame", "radius", "angle"],
        coords={"beyond": [0], "frame": np.arange(0, frame)},
    ).astype(float)


@fixture
def reduced_angular(radial_mask, angular_mask):
    frame = angular_mask.sizes["frame"]

    return xr.DataArray(
        np.tile(
            np.arange(1, angular_mask.sizes["region"] + 1),
            (radial_mask.sizes["region"], frame, 1),
        ).transpose([1, 0, 2])[None, ...],
        dims=["beyond", "frame", "radius", "angle"],
        coords={"beyond": [0], "frame": np.arange(0, frame)},
    ).astype(float)


def test_masked_reduction(
    radial_mask,
    angular_mask,
    expanded_radial,
    expanded_angular,
    reduced_radial,
    reduced_angular,
):
    from strainmap.models.strain import _masked_reduction

    # Masks are reduced, as it will be the real case, covering only certain ROI
    radial = radial_mask.sel(row=radial_mask.row[1:-1], col=radial_mask.col[1:-1])
    angular = angular_mask.sel(row=angular_mask.row[1:-1], col=angular_mask.col[1:-1])

    # An input array with radial symmetry in the row/col plane should return an array
    #  with no angular dependence
    actual = _masked_reduction(expanded_radial, radial, angular)
    xr.testing.assert_equal(actual, reduced_radial)

    # Likewise, if input has angular symmetry, the return value should have no radial
    #  dependence
    actual = _masked_reduction(expanded_angular, radial, angular)
    np.testing.assert_equal(actual.data, reduced_angular)


def test_masked_expansion(
    radial_mask,
    angular_mask,
    expanded_radial,
    expanded_angular,
    reduced_radial,
    reduced_angular,
):
    from strainmap.models.strain import _masked_expansion

    nrow = angular_mask.sizes["row"]
    ncol = angular_mask.sizes["col"]

    # Masks are reduced, as it will be the real case, covering only certain ROI
    radial = radial_mask.sel(row=radial_mask.row[1:-1], col=radial_mask.col[1:-1])
    angular = angular_mask.sel(row=angular_mask.row[1:-1], col=angular_mask.col[1:-1])

    # An input array with no angular dependence should produce an output array with
    # radial symmetry in the row/col plane.
    actual = _masked_expansion(reduced_radial, radial, angular, nrow, ncol)
    xr.testing.assert_equal(actual, expanded_radial.where(~actual.isnull()))

    # Likewise, an input array no radial dependence should produce an output array with
    # angular symmetry in the row/col plane.
    actual = _masked_expansion(reduced_angular, radial, angular, nrow, ncol)
    xr.testing.assert_equal(actual, expanded_angular.where(~actual.isnull()))


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
