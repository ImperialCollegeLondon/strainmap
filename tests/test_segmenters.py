from pytest import approx, raises
from unittest.mock import MagicMock


def test_update_dict():
    from strainmap.models.segmenters import update_dict

    original = {"a": 1, "b": 2}
    new = {"a": 11, "c": 3}

    expected = {"a": 11, "b": 2}
    actual = update_dict(original, new)

    assert expected == actual


def test_segmenter_base_segment_one_image(void_segmenter):
    from strainmap.models.contour_mask import Circle

    c = Circle()

    method = MagicMock()
    method.__name__ = "None"
    sec = void_segmenter(method=method)
    sec._run = MagicMock(return_value=c)

    c2 = c.dilate(p=1.5)

    expected = (c, {}, {})
    actual = sec.segment_one_image(c.mask, c2)

    assert expected == actual


def test_segmenter_base_segment_series(void_segmenter):
    from strainmap.models.contour_mask import Circle

    c = Circle()

    method = MagicMock()
    method.__name__ = "None"
    sec = void_segmenter(method=method)
    sec._run = MagicMock(return_value=c)

    c2 = c.dilate(p=1.5)
    img = [c.mask] * 5
    params = {i: {} for i in range(len(img))}

    expected = ([c] * 5, params, params)
    actual = sec.segment_series(img, c2)

    for i in range(len(img)):
        assert expected[0][i] == actual[0][i]

    assert expected[1:] == actual[1:]


def test_segmenter_base_segment_series_3d(void_segmenter):
    from strainmap.models.contour_mask import Circle

    c = Circle()

    method = MagicMock()
    method.__name__ = "None"
    sec = void_segmenter(method=method)
    sec._run3d = MagicMock(return_value=[c] * 5)

    c2 = c.dilate(p=1.5)
    img = [c.mask] * 5

    expected = ([c] * 5, {}, {})
    actual = sec.segment_series(img, c2, propagation="3d")

    for i in range(len(img)):
        assert expected[0][i] == actual[0][i]

    assert expected[1:] == actual[1:]


def test_segmenter_base_segment_combined(void_segmenter):
    from strainmap.models.contour_mask import Circle

    c = Circle()

    method = MagicMock()
    method.__name__ = "None"
    sec = void_segmenter(method=method)
    sec._run = MagicMock(return_value=c)

    c2 = c.dilate(p=1.5)
    img = ([c.mask] * 5, [c.mask] * 5)
    params = {i: {} for i in range(len(img[0]))}
    params = (params, params)

    expected = (([c] * 5, [c] * 5), params, params)
    actual = sec.segment_combined(img, (c2, c2))

    for i in range(len(img)):
        assert expected[0][:][i] == actual[0][:][i]

    assert expected[1] == actual[1]
    assert expected[2] == actual[2]


def test_active_contour():
    """ This only tests if methods are called correctly, NOT if the algorithm works."""
    from strainmap.models.segmenters import ActiveContour
    from strainmap.models.contour_mask import Circle

    c = Circle()

    method = MagicMock()
    method.__name__ = "None"
    sec = ActiveContour()
    sec.method = MagicMock(return_value=c.xy)

    img = c.mask
    expected = c
    actual = sec._run(img=img, initial=c, params={})

    assert sec.method.call_count == 1
    assert expected.xy == approx(actual.xy)

    with raises(NotImplementedError):
        sec._run3d(img=img, initial=c, params={})


def test_morphological_geodesic_active_contour():
    """ This only tests if methods are called correctly, NOT if the algorithm works."""
    import numpy as np
    from strainmap.models.segmenters import MorphologicalGeodesicActiveContour
    from strainmap.models.contour_mask import Circle

    c = Circle()

    method = MagicMock()
    method.__name__ = "None"
    sec = MorphologicalGeodesicActiveContour()
    sec.method = MagicMock(return_value=c.xy)

    img = c.mask
    expected = c
    actual = sec._run(img=img, initial=c, params={})

    assert sec.method.call_count == 1
    assert expected.xy == approx(actual.xy)

    sec.method = MagicMock(return_value=np.array([c.xy] * 5))

    img = np.array([c.mask] * 5)
    expected = [c] * 5
    actual = sec._run3d(img=img, initial=c, params={})

    assert sec.method.call_count == 1
    for i in range(len(expected)):
        assert expected[i].xy == approx(actual[i].xy)


def test_morphological_chan_vese():
    """ This only tests if methods are called correctly, NOT if the algorithm works."""
    import numpy as np
    from strainmap.models.segmenters import MorphologicalChanVese
    from strainmap.models.contour_mask import Circle

    c = Circle()

    method = MagicMock()
    method.__name__ = "None"
    sec = MorphologicalChanVese()
    sec.method = MagicMock(return_value=c.xy)

    img = c.mask
    expected = c
    actual = sec._run(img=img, initial=c, params={})

    assert sec.method.call_count == 1
    assert expected.xy == approx(actual.xy)

    sec.method = MagicMock(return_value=np.array([c.xy] * 5))

    img = np.array([c.mask] * 5)
    expected = [c] * 5
    actual = sec._run3d(img=img, initial=c, params={})

    assert sec.method.call_count == 1
    for i in range(len(expected)):
        assert expected[i].xy == approx(actual[i].xy)
