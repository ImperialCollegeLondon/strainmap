from pytest import approx


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
    nframes = np.round(interval / min(interval) * frames).astype(int)

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
    actual = unresample_interval(disp, interval)

    assert actual.shape == disp[:, :frames].shape
    assert np.squeeze(actual) == approx(expected, abs=1e-1)
