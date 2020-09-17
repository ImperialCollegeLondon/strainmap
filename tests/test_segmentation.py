from pytest import approx, raises


def test_replace_single():
    from strainmap.models.contour_mask import Contour
    from strainmap.models.segmentation import _replace_single

    c = Contour.circle((0, 0), 10)
    c2 = c.dilate(1.05)

    out = _replace_single(c, c2)
    assert out.xy == approx(c2.xy)

    out = _replace_single(c, c2, replace=False)
    assert out.xy == approx(c.xy)

    c3 = c.dilate(1.5)
    out = _replace_single(c, c3, replace=False)
    assert out.xy == approx(c3.xy)


def test_replace_in_list():
    from strainmap.models.contour_mask import Contour
    from strainmap.models.segmentation import _replace_in_list

    c = Contour.circle((0, 0), 10)
    contours = [c, c.dilate(1.5), c.dilate(1.05)]
    expected = [c, c, c]

    actual = _replace_in_list(contours, frame_threshold=2)
    for ex, ac in zip(expected, actual):
        assert ac.xy == approx(ex.xy)


def test_centroids():
    import numpy as np
    from strainmap.models.segmentation import _calc_centroids, _init_segments

    frames = np.random.randint(1, 51)
    centroid = _init_segments("mid", frames, 360)
    centroid[...] = np.random.random((1, frames, 2, 360))
    expected = np.mean(centroid.sel(side="epicardium").data, axis=3)
    actual = _calc_centroids(centroid)

    assert actual.shape == (1, frames, 2)
    assert actual.data == approx(expected)


def test_effective_centroid():
    import numpy as np
    from scipy import ndimage
    from strainmap.models.segmentation import (
        _calc_effective_centroids,
        _init_septum_and_centroid,
    )

    frames = 50
    centroid = _init_septum_and_centroid("mid", frames, "centroid").sel(cine="mid")
    centroid[...] = np.random.random((frames, 2))
    n = np.random.randint(1, frames - 1)
    weights = np.ones((2 * n + 1))
    expected = ndimage.convolve1d(centroid.data, weights, axis=0, mode="wrap") / (
        2 * n + 1
    )
    actual = _calc_effective_centroids(centroid, n)

    assert actual.shape == centroid.shape
    assert actual.data == approx(expected)


def test_update_segmentation(segments_arrays):
    import numpy as np
    from numpy.random import default_rng
    import xarray as xr
    from strainmap.models.segmentation import _update_segmentation

    segments, septum, centroid = segments_arrays
    segments[...] = np.random.random(segments.shape)
    septum[...] = np.random.random(septum.shape)
    velocities = {c: None for c in segments.cine.data}
    frames = segments.sizes["frame"]

    # Checks updating the septum of 1 frame
    frame, other = default_rng().choice(frames, size=2, replace=False)
    new_septum = septum.sel(cine="mid", frame=frame) * 2
    _update_segmentation(
        segments.sel(cine="mid"),
        centroid.sel(cine="mid"),
        septum.sel(cine="mid"),
        velocities,
        new_septum=new_septum,
    )
    xr.testing.assert_equal(septum.sel(cine="mid", frame=frame), new_septum)
    assert "mid" not in velocities
    with raises(AssertionError):
        xr.testing.assert_equal(septum.sel(cine="mid", frame=other), new_septum)

    # Checks updating the segmentation of 1 frame
    frame, other = default_rng().choice(frames, size=2, replace=False)
    new_segments = segments.sel(cine="base", frame=frame) * 2
    _update_segmentation(
        segments.sel(cine="base"),
        centroid.sel(cine="base"),
        septum.sel(cine="base"),
        velocities,
        new_segments=new_segments,
    )
    xr.testing.assert_equal(segments.sel(cine="base", frame=frame), new_segments)
    assert "base" not in velocities
    with raises(AssertionError):
        xr.testing.assert_equal(segments.sel(cine="base", frame=other), new_segments)

    # Checks updating the segmentation of all frames
    new_segments = segments.sel(cine="apex") * 2
    _update_segmentation(
        segments.sel(cine="apex"),
        centroid.sel(cine="apex"),
        septum.sel(cine="apex"),
        velocities,
        new_segments=new_segments,
    )
    xr.testing.assert_equal(segments.sel(cine="apex"), new_segments)
    assert not np.isnan(centroid.sel(cine="apex").data).any()
    assert "apex" not in velocities
