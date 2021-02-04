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


def test_crop_roi():
    assert False


def test_add_ellipse():
    assert False


def test_get_contours():
    assert False


def test_interpolate_contour():
    assert False


def test_labels_to_contours():
    assert False


class TestDataAugmentation:
    def test_factory(self):
        assert False

    def test_transform(self):
        assert False

    def test_augment(self):
        assert False

    def test__group(self):
        assert False

    def test__ungroup(self):
        assert False


class TestNormal:
    def test_avail(self):
        assert False

    def test_register(self):
        assert False

    def test_run(self):
        assert False


class TestUNet:
    def test_factory(self):
        assert False

    def test_model(self):
        assert False

    def test__conv_block(self):
        assert False

    def test__deconv_block(self):
        assert False

    def test__modelstruct(self):
        assert False

    def test_compile_model(self):
        assert False

    def test_train(self):
        assert False

    def test_infer(self):
        assert False
