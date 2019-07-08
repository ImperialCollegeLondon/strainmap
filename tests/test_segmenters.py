from pytest import approx, raises, mark
from unittest.mock import MagicMock, patch

patch(
    "skimage.segmentation.active_contour", lambda img, ini, *args, **kwargs: ini
).start()
patch(
    "skimage.segmentation.morphological_geodesic_active_contour",
    lambda img, iterations, init_level_set, *args, **kwargs: init_level_set,
).start()
patch(
    "skimage.segmentation.morphological_chan_vese",
    lambda img, iterations, init_level_set, *args, **kwargs: init_level_set,
).start()

from strainmap.models.segmenters import (  # noqa: F403,F401
    morphological_geodesic_active_contour_model,
    morphological_chan_vese_model,
)


def test_segmenter_global_segmenter():
    from strainmap.models.contour_mask import Contour
    from strainmap.models.segmenters import Segmenter

    c = Contour.circle()

    model = MagicMock(return_value=c)
    ffilter = MagicMock(side_effect=lambda x, **kwargs: x)

    c2 = c.dilate(p=1.5)

    sec = Segmenter(model, ffilter, lambda: None)

    actual = sec(c.mask, c2)
    assert c.xy == approx(actual.xy)


def test_segmenter_propagated_segmenter():
    from strainmap.models.contour_mask import Contour
    from strainmap.models.segmenters import Segmenter
    import numpy as np

    c = Contour.circle()

    model = MagicMock(return_value=c)
    ffilter = MagicMock(side_effect=lambda x, **kwargs: x)
    propagator = MagicMock(side_effect=lambda x, *args, **kwargs: x)
    propagator.__name__ = "dummy"

    c2 = c.dilate(p=1.5)

    sec = Segmenter(model, ffilter, propagator)

    actual = sec(c.mask, c2)
    assert c.xy == approx(actual.xy)

    actual = sec(np.array([c.mask] * 3), c2)
    assert len(actual) == 3
    for act in actual:
        assert c.xy == approx(act.xy)


def test_active_contour():
    """ This only tests if methods are called correctly, NOT if the algorithm works."""
    from strainmap.models.segmenters import active_contour_model
    from strainmap.models.contour_mask import Contour
    import numpy as np

    c = Contour.circle()

    expected = c
    actual = active_contour_model(img=c.mask, initial=c, params={})
    assert expected.xy == approx(actual.xy)

    with raises(NotImplementedError):
        active_contour_model(img=np.array([c.mask] * 3), initial=c, params={})


@mark.parametrize(
    "solver",
    [morphological_geodesic_active_contour_model, morphological_chan_vese_model],
)
def test_morphological_geodesic_active_contour(solver):
    """ This only tests if methods are called correctly, NOT if the algorithm works."""
    import numpy as np
    from strainmap.models.contour_mask import Contour

    c = Contour.circle()

    actual = solver(img=c.mask, initial=c, params={})
    assert c.mask == approx(actual.mask)

    actual = solver(img=np.array([c.mask] * 3), initial=c, params={})
    assert len(actual) == 3
    for act in actual:
        assert c.mask == approx(act.mask)
