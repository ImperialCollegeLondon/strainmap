from pytest import approx


def test_resample():
    from strainmap.models.strain import resample
    import numpy as np

    interval = (0.02, 0.03, 0.04)
    expected = np.sin(np.pi * np.linspace(0, min(interval) * 50, 50, endpoint=False))

    t = (np.linspace(0, interv * 50, 50, endpoint=False) for interv in interval)
    disp = np.moveaxis(
        np.array([np.sin(np.pi * tt) for tt in t])[..., None, None, None],
        (0, 1),
        (2, 1),
    )
    actual = resample(disp, interval)

    assert disp.shape == actual.shape
    assert np.squeeze(actual) == approx(np.tile(expected, (3, 1)).T, abs=1e-2)
