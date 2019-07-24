from pytest import approx

from unittest.mock import MagicMock


def test_update_and_clear_widgets(segmentation_view, strainmap_data):
    segmentation_view.update_plots = MagicMock()
    segmentation_view.data = strainmap_data

    expected = sorted(strainmap_data.data_files.keys())[0]
    assert segmentation_view.datasets_var.get() == expected

    segmentation_view.update_plots.assert_called_once()


def test_update_plots(segmentation_view, strainmap_data):
    segmentation_view.plot_images = MagicMock()
    segmentation_view.plot_segments = MagicMock()
    segmentation_view.data = strainmap_data

    segmentation_view.plot_images.assert_called_once()
    segmentation_view.plot_segments.assert_called_once()

    assert segmentation_view.ax_mag.get_title("right") == "Magnitude"
    assert segmentation_view.ax_vel.get_title("right") == "Velocity"


def test_plot_images(segmentation_view, strainmap_data):
    segmentation_view.data = strainmap_data

    generators = segmentation_view.fig.actions_manager.ScrollFrames._images_generator

    assert segmentation_view.ax_mag in generators
    assert segmentation_view.ax_vel in generators


def test_get_data_to_segment(segmentation_view, strainmap_data):
    segmentation_view.data = strainmap_data

    dataset = sorted(strainmap_data.data_files.keys())[0]
    expected_vel = strainmap_data.get_images(dataset, "PhaseZ")

    mag, vel = segmentation_view.get_data_to_segment()
    assert expected_vel == approx(vel)
    assert expected_vel.shape == mag.shape


def test_switch_button_text(segmentation_view):
    expected = "Ready"
    actual = segmentation_view.nametowidget("control.segmentFrame.initialEndo")["text"]
    assert expected != actual

    segmentation_view.switch_button_text("endocardium", expected)
    actual = segmentation_view.nametowidget("control.segmentFrame.initialEndo")["text"]
    assert expected == actual


def test_switch_mark_state(segmentation_view):
    expected = "\u2705"
    actual = segmentation_view.nametowidget("control.segmentFrame.readyEndo")["text"]
    assert expected != actual

    segmentation_view.switch_mark_state("endocardium", "ready")
    actual = segmentation_view.nametowidget("control.segmentFrame.readyEndo")["text"]
    assert expected == actual


def test_enter_and_leave_initial_edit_mode(segmentation_view):
    assert segmentation_view.cursors["mag"] is None
    segmentation_view.enter_initial_edit_mode()
    assert segmentation_view.cursors["mag"] is not None
    segmentation_view.leave_initial_edit_mode()
    assert segmentation_view.cursors["mag"] is None


def test_define_initial_contour(segmentation_view):
    segmentation_view.define_initial_contour("endocardium")

    assert "get_contour" in str(
        segmentation_view.fig.actions_manager.DrawContours.contours_updated
    )


def test_get_contour(segmentation_view):
    import numpy as np

    segmentation_view.get_contour([], side="endocardium")
    assert segmentation_view.initial_segments["endocardium"] is None

    contour = np.random.random((2, 5))
    segmentation_view.get_contour([contour], side="endocardium")
    assert segmentation_view.initial_segments["endocardium"] == approx(contour)


def test_segmentation_ready(segmentation_view):
    import numpy as np

    contour = np.random.random((2, 5))
    segmentation_view.get_contour([contour], side="endocardium")
    segmentation_view.segmentation_ready()
    assert (
        segmentation_view.nametowidget("control.runSegmentation")["state"] != "enable"
    )

    segmentation_view.get_contour([contour], side="epicardium")
    segmentation_view.segmentation_ready()
    assert (
        segmentation_view.nametowidget("control.runSegmentation")["state"] == "enable"
    )
