from pytest import approx


def test_initial():
    from strainmap.models.contour_mask import Circle
    from strainmap.models.propagators import initial

    c = Circle()
    c2 = c.dilate(p=2)

    actual = initial(initial=c, previous=c2)

    assert c.xy == approx(actual.xy)


def test_previous():
    from strainmap.models.contour_mask import Circle
    from strainmap.models.propagators import previous

    c = Circle()
    c2 = c.dilate(p=2)

    actual = previous(initial=c, previous=c2)

    assert c2.xy == approx(actual.xy)

    c3 = c2.dilate(p=2)

    actual = previous(initial=c, previous=c2, options={"dilation_factor": 2})

    assert c3.xy == approx(actual.xy)


def test_weighted():
    from strainmap.models.contour_mask import Circle
    from strainmap.models.propagators import weighted
    import numpy as np

    c = Circle()
    c2 = c.dilate(p=2)
    c3 = c.dilate(p=1.5)

    actual = weighted(initial=c, previous=c2, options={"weight": 1})
    assert np.mean(c.polar[:, 0]) == approx(np.mean(actual.polar[:, 0]), rel=0.01)
    actual = weighted(initial=c, previous=c2, options={"weight": 0})
    assert np.mean(c2.polar[:, 0]) == approx(np.mean(actual.polar[:, 0]), rel=0.01)
    actual = weighted(initial=c, previous=c2, options={"weight": 0.5})
    assert np.mean(c3.polar[:, 0]) == approx(np.mean(actual.polar[:, 0]), rel=0.01)


def test_initial_combined():
    from strainmap.models.contour_mask import Circle
    from strainmap.models.propagators import initial_combined

    c = (Circle(), Circle())
    c2 = (Circle().dilate(p=2), Circle().dilate(p=2))

    actual = initial_combined(initial=c, previous=c2, step=0)

    assert c[0].xy == approx(actual[0].xy)
    assert c[1].xy == approx(actual[1].xy)
