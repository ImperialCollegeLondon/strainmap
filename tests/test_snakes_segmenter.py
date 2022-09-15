from unittest.mock import MagicMock, patch

from pytest import approx, raises


def test_segmenter_global_segmenter():
    from strainmap.models.contour_mask import Contour
    from strainmap.models.snakes_segmenter import Segmenter

    c = Contour.circle()

    model = MagicMock(return_value=c)
    ffilter = MagicMock(side_effect=lambda x, **kwargs: x)

    c2 = c.dilate(p=1.5)

    sec = Segmenter(model, ffilter)

    actual = sec(c.mask, c2)
    assert c.xy == approx(actual.xy)


def test_segmenter_propagated_segmenter():
    import numpy as np

    from strainmap.models.contour_mask import Contour
    from strainmap.models.snakes_segmenter import Segmenter

    c = Contour.circle()

    model = MagicMock(return_value=c)
    ffilter = MagicMock(side_effect=lambda x, **kwargs: x)
    propagator = MagicMock(side_effect=lambda initial, *args, **kwargs: initial)

    c2 = c.dilate(p=1.5)

    sec = Segmenter(model, ffilter, propagator)

    actual = sec(c.mask, c2)
    assert c.xy == approx(actual.xy)

    actual = sec(np.array([c.mask] * 3), c2)
    assert len(actual) == 3
    for act in actual:
        assert c.xy == approx(act.xy)


@patch("skimage.segmentation.active_contour", lambda img, ini, *args, **kwargs: ini)
def test_active_contour():
    """This only tests if methods are called correctly, NOT if the algorithm works."""
    import numpy as np

    from strainmap.models.contour_mask import Contour
    from strainmap.models.snakes_segmenter import active_contour_model

    c = Contour.circle()

    actual = active_contour_model(img=c.mask, initial=c, params={})
    assert c.xy == approx(actual.xy)

    with raises(NotImplementedError):
        active_contour_model(img=np.array([c.mask] * 3), initial=c, params={})


@patch(
    "skimage.segmentation.morphological_geodesic_active_contour",
    lambda img, iterations, init_level_set, *args, **kwargs: init_level_set,
)
def test_morph_gac_model():
    """This only tests if methods are called correctly, NOT if the algorithm works."""
    import numpy as np

    from strainmap.models.contour_mask import Contour
    from strainmap.models.snakes_segmenter import (
        morphological_geodesic_active_contour_model,
    )

    c = Contour.circle()

    actual = morphological_geodesic_active_contour_model(
        img=c.mask, initial=c, params={}
    )
    assert (c.image == actual.image).all()

    actual = morphological_geodesic_active_contour_model(
        img=np.array([c.mask] * 3), initial=c, params={}
    )
    assert len(actual) == 3
    for act in actual:
        assert (c.image == act.image).all()


@patch(
    "skimage.segmentation.morphological_chan_vese",
    lambda img, iterations, init_level_set, *args, **kwargs: init_level_set,
)
def test_morph_cv_model():
    """This only tests if methods are called correctly, NOT if the algorithm works."""
    import numpy as np

    from strainmap.models.contour_mask import Contour
    from strainmap.models.snakes_segmenter import morphological_chan_vese_model

    c = Contour.circle()

    actual = morphological_chan_vese_model(img=c.mask, initial=c, params={})
    assert c.mask == approx(actual.mask)

    actual = morphological_chan_vese_model(
        img=np.array([c.mask] * 3), initial=c, params={}
    )
    assert len(actual) == 3
    for act in actual:
        assert c.mask == approx(act.mask)


def test_replace_single():
    from strainmap.models.contour_mask import Contour
    from strainmap.models.snakes_segmenter import _replace_single

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
    from strainmap.models.snakes_segmenter import _replace_in_list

    c = Contour.circle((0, 0), 10)
    contours = [c, c.dilate(1.5), c.dilate(1.05)]
    expected = [c, c, c]

    actual = _replace_in_list(contours, frame_threshold=2)
    for ex, ac in zip(expected, actual):
        assert ac.xy == approx(ex.xy)
