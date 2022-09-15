from pytest import approx


def test_initial():
    from strainmap.models.contour_mask import Contour
    from strainmap.models.propagators import initial

    c = Contour.circle()
    c2 = c.dilate(p=2)

    actual = initial(initial=c, previous=c2)

    assert c.xy == approx(actual.xy)


def test_previous():
    from strainmap.models.contour_mask import Contour
    from strainmap.models.propagators import previous

    c = Contour.circle()
    c2 = c.dilate(p=2)

    actual = previous(initial=c, previous=c2)

    assert c2.xy == approx(actual.xy)

    c3 = c2.dilate(p=2)

    actual = previous(initial=c, previous=c2, dilation_factor=2)

    assert c3.xy == approx(actual.xy)


def test_weighted():
    import numpy as np

    from strainmap.models.contour_mask import Contour
    from strainmap.models.propagators import weighted

    c = Contour.circle()
    c2 = c.dilate(p=2)
    c3 = c.dilate(p=1.5)

    actual = weighted(initial=c, previous=c2, weight=1)
    assert np.mean(c.polar.r) == approx(np.mean(actual.polar.r), rel=0.01)
    actual = weighted(initial=c, previous=c2, weight=0)
    assert np.mean(c2.polar.r) == approx(np.mean(actual.polar.r), rel=0.01)
    actual = weighted(initial=c, previous=c2, weight=0.5)
    assert np.mean(c3.polar.r) == approx(np.mean(actual.polar.r), rel=0.01)
