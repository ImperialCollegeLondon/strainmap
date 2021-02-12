from pytest import approx, mark


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


@mark.xfail(reason="Not implemented")
def test_new_segmentation():
    assert False


@mark.xfail(reason="Not implemented")
def test_update_segmentation():
    assert False


@mark.xfail(reason="Not implemented")
def test_update_and_find_next():
    assert False


@mark.xfail(reason="Not implemented")
def test_remove_segmentation():
    assert False


@mark.xfail(reason="Not implemented")
def test__drop_cine():
    assert False


@mark.xfail(reason="Not implemented")
def test__get_segment_variables():
    assert False


@mark.xfail(reason="Not implemented")
def test__init_segments():
    assert False


@mark.xfail(reason="Not implemented")
def test__init_septum_and_centroid():
    assert False
