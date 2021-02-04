import pytest


def test_zero2one():
    from strainmap.models.ai_segmenter import zero2one
    import numpy as np

    data = np.random.random((5, 10, 3))
    norm = zero2one(data)
    assert norm.min() == pytest.approx(0)
    assert norm.max() == pytest.approx(1)


def test_zeromean_unitvar():
    from strainmap.models.ai_segmenter import zeromean_unitvar
    import numpy as np

    data = np.random.random((5, 10, 3))
    norm = zeromean_unitvar(data)
    assert np.mean(norm) == pytest.approx(0)
    assert np.std(norm) == pytest.approx(1)


@pytest.mark.xfail
def test_crop_roi():
    assert False


@pytest.mark.xfail
def test_add_ellipse():
    assert False


@pytest.mark.xfail
def test_get_contours():
    assert False


@pytest.mark.xfail
def test_interpolate_contour():
    assert False


@pytest.mark.xfail
def test_labels_to_contours():
    assert False


class TestDataAugmentation:

    @pytest.mark.xfail
    def test_factory(self):
        assert False

    @pytest.mark.xfail
    def test_transform(self):
        assert False

    @pytest.mark.xfail
    def test_augment(self):
        assert False

    @pytest.mark.xfail
    def test__group(self):
        assert False

    @pytest.mark.xfail
    def test__ungroup(self):
        assert False


class TestNormal:

    @pytest.mark.xfail
    def test_avail(self):
        assert False

    @pytest.mark.xfail
    def test_register(self):
        assert False

    @pytest.mark.xfail
    def test_run(self):
        assert False


class TestUNet:

    @pytest.mark.xfail
    def test_factory(self):
        assert False

    @pytest.mark.xfail
    def test_model(self):
        assert False

    @pytest.mark.xfail
    def test__conv_block(self):
        assert False

    @pytest.mark.xfail
    def test__deconv_block(self):
        assert False

    @pytest.mark.xfail
    def test__modelstruct(self):
        assert False

    @pytest.mark.xfail
    def test_compile_model(self):
        assert False

    @pytest.mark.xfail
    def test_train(self):
        assert False

    @pytest.mark.xfail
    def test_infer(self):
        assert False
