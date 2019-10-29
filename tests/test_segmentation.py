from pytest import approx


def test_replace_single():
    from strainmap.models.contour_mask import Contour
    from strainmap.models.quick_segmentation import replace_single

    c = Contour.circle((0, 0), 10)
    c2 = c.dilate(1.05)

    out = replace_single(c, c2)
    assert out.xy == approx(c2.xy)

    out = replace_single(c, c2, replace=False)
    assert out.xy == approx(c.xy)

    c3 = c.dilate(1.5)
    out = replace_single(c, c3, replace=False)
    assert out.xy == approx(c3.xy)


def test_replace_in_list():
    from strainmap.models.contour_mask import Contour
    from strainmap.models.quick_segmentation import replace_in_list

    c = Contour.circle((0, 0), 10)
    contours = [c, c.dilate(1.5), c.dilate(1.05)]
    expected = [c, c, c]

    actual = replace_in_list(contours, frame_threshold=2)
    for ex, ac in zip(expected, actual):
        assert ac.xy == approx(ex.xy)
