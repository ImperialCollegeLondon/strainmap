from unittest.mock import MagicMock, patch

from pytest import approx, raises


def test_segmenter_global_segmenter():
    from strainmap.models.contour_mask import Contour
    from strainmap.models.segmenters import Segmenter

    c = Contour.circle()

    model = MagicMock(return_value=c)
    ffilter = MagicMock(side_effect=lambda x, **kwargs: x)

    c2 = c.dilate(p=1.5)

    sec = Segmenter(model, ffilter)

    actual = sec(c.mask, c2)
    assert c.xy == approx(actual.xy)


def test_segmenter_propagated_segmenter():
    from strainmap.models.contour_mask import Contour
    from strainmap.models.segmenters import Segmenter
    import numpy as np

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
    """ This only tests if methods are called correctly, NOT if the algorithm works."""
    from strainmap.models.contour_mask import Contour
    from strainmap.models.segmenters import active_contour_model
    import numpy as np

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
    """ This only tests if methods are called correctly, NOT if the algorithm works."""
    import numpy as np
    from strainmap.models.contour_mask import Contour
    from strainmap.models.segmenters import morphological_geodesic_active_contour_model

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
    """ This only tests if methods are called correctly, NOT if the algorithm works."""
    import numpy as np
    from strainmap.models.contour_mask import Contour
    from strainmap.models.segmenters import morphological_chan_vese_model

    c = Contour.circle()

    actual = morphological_chan_vese_model(img=c.mask, initial=c, params={})
    assert c.mask == approx(actual.mask)

    actual = morphological_chan_vese_model(
        img=np.array([c.mask] * 3), initial=c, params={}
    )
    assert len(actual) == 3
    for act in actual:
        assert c.mask == approx(act.mask)
