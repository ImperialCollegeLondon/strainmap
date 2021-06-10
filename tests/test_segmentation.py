from pytest import approx
from unittest.mock import patch, MagicMock


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


@patch("strainmap.models.segmentation.SEGMENTATION_METHOD", {"snakes": MagicMock()})
def test_new_segmentation(strainmap_data, initial_segments):
    import numpy as np
    from strainmap.models.segmentation import (
        SEGMENTATION_METHOD,
        _init_septum_and_centroid,
        new_segmentation,
    )

    # Setup
    frame = 0
    frames = strainmap_data.data_files.frames
    cine = strainmap_data.data_files.datasets[0]
    strainmap_data.save = MagicMock()
    new_septum = _init_septum_and_centroid(cine, frames, "septum").sel(cine=cine)
    new_septum.loc[{"frame": frame}] = np.array([0, 0])
    SEGMENTATION_METHOD["snakes"].return_value = initial_segments.expand_dims(
        "frame", 1
    )

    # Test - segment 1 frame
    new_segmentation(
        strainmap_data,
        cine,
        frame=frame,
        initials=initial_segments,
        new_septum=new_septum,
    )
    SEGMENTATION_METHOD["snakes"].assert_called_once()
    strainmap_data.save.assert_called_once_with("segments", "centroid", "septum")
    assert strainmap_data.septum.isnull().sum() > 0

    # Test - segment all frames
    new_segmentation(
        strainmap_data,
        cine,
        frame=None,
        initials=initial_segments,
        new_septum=new_septum,
    )
    assert strainmap_data.septum.isnull().sum() == 0


def test_update_segmentation(strainmap_data, initial_segments):
    import numpy as np
    from strainmap.models.segmentation import (
        _init_septum_and_centroid,
        update_segmentation,
    )

    # Setup
    frame = 0
    frames = strainmap_data.data_files.frames
    cine = strainmap_data.data_files.datasets[0]
    strainmap_data.save = MagicMock()
    strainmap_data.save_all = MagicMock()
    new_septum = _init_septum_and_centroid(cine, frames, "septum").sel(cine=cine)

    # Test update 1 frame
    new_septum.loc[{"frame": frame}] = np.array([0, 0])
    new_segments = initial_segments.expand_dims("frame", 1)
    update_segmentation(strainmap_data, cine, new_segments, new_septum)
    assert not strainmap_data.segments.sel(frame=frame).isnull().any()
    strainmap_data.save.assert_called_once_with("segments", "centroid", "septum")

    # Test update all frames
    new_septum[...] = 0
    new_segments = initial_segments.expand_dims({"frame": frames}, 1)
    update_segmentation(strainmap_data, cine, new_segments, new_septum)
    assert not strainmap_data.segments.isnull().any()
    strainmap_data.save_all.assert_called_once()


@patch("strainmap.models.segmentation.SEGMENTATION_METHOD", {"snakes": MagicMock()})
@patch("strainmap.models.segmentation.update_segmentation", MagicMock())
def test_update_and_find_next(strainmap_data, initial_segments):
    import numpy as np
    from strainmap.models.segmentation import (
        SEGMENTATION_METHOD,
        _init_septum_and_centroid,
        update_segmentation,
        update_and_find_next,
    )

    # Setup
    frame = 1
    frames = strainmap_data.data_files.frames
    cine = strainmap_data.data_files.datasets[0]
    new_septum = _init_septum_and_centroid(cine, frames, "septum").sel(cine=cine)
    new_septum.loc[{"frame": frame}] = np.array([0, 0])
    SEGMENTATION_METHOD["snakes"].return_value = initial_segments.expand_dims(
        "frame", 1
    )

    new_septum.loc[{"frame": frame}] = np.array([0, 0])
    new_segments = initial_segments.expand_dims("frame", 1)
    update_and_find_next(strainmap_data, cine, frame, new_segments, new_septum)
    SEGMENTATION_METHOD["snakes"].assert_called_once()
    assert not strainmap_data.segments.sel(frame=frame).isnull().any()
    update_segmentation.assert_called_once()


def test_remove_segmentation():
    import xarray as xr
    from types import SimpleNamespace as SimpleNM
    from strainmap.models.segmentation import (
        _get_segment_variables,
        remove_segmentation,
    )

    data = SimpleNM(
        data_files=SimpleNM(frames=5),
        segments=SimpleNM(shape=()),
        velocities=SimpleNM(shape=()),
        save_all=MagicMock(),
    )
    cine = "top"
    _get_segment_variables(data, cine)
    remove_segmentation(data, cine)

    expected = xr.DataArray()
    xr.testing.assert_equal(data.segments, expected)
    xr.testing.assert_equal(data.centroid, expected)
    xr.testing.assert_equal(data.septum, expected)
    xr.testing.assert_equal(data.strain, expected)
    xr.testing.assert_equal(data.strain_markers, expected)
    assert not hasattr(data, "masks")
    data.save_all.assert_called_once()


def test__drop_cine():
    import xarray as xr
    from strainmap.models.segmentation import _drop_cine

    data = xr.DataArray([1, 2], dims=("cine",), coords={"cine": ["bottom", "top"]})
    expected = xr.DataArray([2], dims=("cine",), coords={"cine": ["top"]})
    actual = _drop_cine(data, "bottom")
    xr.testing.assert_equal(actual, expected)
    actual = _drop_cine(actual, "top")
    xr.testing.assert_equal(actual, xr.DataArray())


def test__get_segment_variables():
    import xarray as xr
    from types import SimpleNamespace as SimpleNM
    from strainmap.models.segmentation import _get_segment_variables

    data = SimpleNM(data_files=SimpleNM(frames=5), segments=SimpleNM(shape=()))

    cines = ("top", "bottom")
    for cine in cines:
        variables = _get_segment_variables(data, cine)

        for v in variables:
            assert isinstance(v, xr.DataArray)
            assert cine == v.cine
            assert v.sizes["frame"] == 5

    assert cines == tuple(data.segments.cine)
    assert cines == tuple(data.centroid.cine)
    assert cines == tuple(data.septum.cine)


def test__init_segments():
    import numpy as np
    from numpy.testing import assert_array_equal
    from strainmap.models.segmentation import _init_segments

    out = _init_segments("bottom", 10, 200)
    assert out.dims == ("cine", "side", "frame", "coord", "point")
    assert_array_equal(out.coords["cine"].data, np.array(["bottom"]))
    assert_array_equal(out.coords["side"].data, np.array(["endocardium", "epicardium"]))
    assert out.sizes["frame"] == 10
    assert_array_equal(out.coords["coord"].data, np.array(["col", "row"]))
    assert out.sizes["point"] == 200
    assert out.name == "segments"


def test__init_septum_and_centroid():
    import numpy as np
    from strainmap.models.segmentation import _init_septum_and_centroid

    out = _init_septum_and_centroid("bottom", 10, "septum")
    assert out.dims == ("cine", "frame", "coord")
    assert out.coords["cine"].data == np.array(["bottom"])
    assert out.sizes["frame"] == 10
    assert out.name == "septum"


def test__calc_centroids(initial_segments, initial_centroid):
    from strainmap.models.segmentation import _calc_centroids

    actual = _calc_centroids(initial_segments)
    assert actual.data == approx(initial_centroid)


def test__calc_effective_centroids():
    import xarray as xr
    import numpy as np
    from strainmap.models.segmentation import _calc_effective_centroids

    centroids = np.array([[3, 4], [4, 4], [3, 4], [4, 4]])
    segments = xr.DataArray(
        centroids, dims=("frame", "coord"), coords={"coord": ["col", "row"]}
    ).astype(float)
    actual = _calc_effective_centroids(segments)
    expected = np.array([[3.5, 4], [3.5, 4], [3.5, 4], [3.5, 4]])
    assert actual.data == approx(expected, abs=0.1)
