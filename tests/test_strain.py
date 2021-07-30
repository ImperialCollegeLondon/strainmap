from pytest import approx, mark
import xarray as xr
import numpy as np


def test_shift_data(reduced_radial):
    from strainmap.models.strain import _shift_data
    import scipy as sp

    interval = 0.25
    timeshift = 0.6
    frames = reduced_radial.frame.size

    data = reduced_radial * reduced_radial.frame
    time_intervals = xr.DataArray(
        [interval] * data.cine.size, dims=["cine"], coords={"cine": data.cine}
    )
    actual = _shift_data(data, time_intervals, timeshift)

    shift, rem = divmod(timeshift, interval)
    times = np.arange(0, frames * interval, interval)
    expected = sp.interpolate.interp1d(
        times,
        np.roll(np.arange(0, frames - 0.1), int(-shift)),
        fill_value="extrapolate",
    )(times + rem)

    # Actually, we just check that the expectation for one spacial location is OK
    assert actual.isel(cine=0, radius=0, angle=0).data == approx(expected)


@mark.xfail
def test_displacement():
    assert False


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
