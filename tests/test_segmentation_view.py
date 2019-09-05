from pytest import approx

from unittest.mock import MagicMock


def test_update_and_clear_widgets(segmentation_view, strainmap_data):
    segmentation_view.dataset_changed = MagicMock()
    segmentation_view.data = strainmap_data

    expected = list(strainmap_data.data_files.keys())[0]
    assert segmentation_view.datasets_var.get() == expected

    segmentation_view.dataset_changed.assert_called_once()


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

    generators = segmentation_view.fig.actions_manager.ScrollFrames._scroller

    assert segmentation_view.ax_mag in generators
    assert segmentation_view.ax_vel in generators


def test_get_data_to_segment(segmentation_view, strainmap_data):
    segmentation_view.data = strainmap_data

    dataset = list(strainmap_data.data_files.keys())[0]
    expected_vel = strainmap_data.get_images(dataset, "PhaseZ")

    actual = segmentation_view.get_data_to_segment(dataset)
    assert expected_vel == approx(actual["vel"])
    assert expected_vel.shape == actual["mag"].shape


def test_switch_button_text(segmentation_view):
    expected = "Ready"
    actual = segmentation_view.ini_endo_btn["text"]
    assert expected != actual

    segmentation_view.switch_button_text("endocardium", expected)
    actual = segmentation_view.ini_endo_btn["text"]
    assert expected == actual


def test_switch_mark_state(segmentation_view):
    expected = "\u2705"
    actual = segmentation_view.endo_redy_var.get()
    assert expected != actual

    segmentation_view.switch_mark_state("endocardium", "ready")
    actual = segmentation_view.endo_redy_var.get()
    assert expected == actual


def test_enter_and_leave_initial_edit_mode(segmentation_view):
    assert segmentation_view.cursors["mag"] is None
    segmentation_view.enter_initial_edit_mode()
    assert segmentation_view.cursors["mag"] is not None
    segmentation_view.leave_initial_edit_mode()
    assert segmentation_view.cursors["mag"] is None


def test_define_initial_contour(segmentation_view, strainmap_data):
    from copy import deepcopy

    segmentation_view.data = deepcopy(strainmap_data)
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
    assert segmentation_view.next_btn.instate(["disabled"])

    segmentation_view.get_contour([contour], side="epicardium")
    assert segmentation_view.next_btn.instate(["!disabled"])


def test_plot_segments(segmentation_view, strainmap_data):
    import numpy as np
    from copy import deepcopy

    contour = np.random.random((2, 5))
    dataset = list(strainmap_data.data_files.keys())[0]

    segmentation_view.data = deepcopy(strainmap_data)
    segmentation_view.data.segments[dataset]["endocardium"] = [contour]
    segmentation_view.data.segments[dataset]["epicardium"] = [contour]

    segmentation_view.plot_segments(dataset)

    generators = segmentation_view.fig.actions_manager.ScrollFrames._scroller

    assert segmentation_view.ax_mag in generators
    assert segmentation_view.ax_vel in generators


def test_initial_contour(segmentation_view, strainmap_data):
    import numpy as np
    from copy import deepcopy

    segmentation_view.data = deepcopy(strainmap_data)
    segmentation_view.define_initial_contour("endocardium")
    actual = segmentation_view.ini_endo_btn["text"]
    assert actual == "Cancel"

    segmentation_view.initial_contour("endocardium")
    actual = segmentation_view.ini_endo_btn["text"]
    assert actual == "Define endocardium"

    segmentation_view.define_initial_contour("endocardium")
    contour = np.random.random((2, 5))
    segmentation_view.get_contour([contour], side="endocardium")
    actual = segmentation_view.ini_endo_btn["text"]
    assert actual == "Clear endocardium"

    segmentation_view.clear_segments = MagicMock()
    segmentation_view.initial_contour("endocardium")
    actual = segmentation_view.ini_endo_btn["text"]
    assert actual == "Define endocardium"


def test_clear_segments(segmentation_view):
    import numpy as np

    segmentation_view.plot_segments = MagicMock()
    segmentation_view.plot_initial_segments = MagicMock()

    contour = np.random.random((2, 5))

    segmentation_view.initial_segments["endocardium"] = contour
    segmentation_view.initial_segments["epicardium"] = contour
    segmentation_view.final_segments["endocardium"] = contour
    segmentation_view.final_segments["epicardium"] = contour

    segmentation_view.clear_segments()

    assert segmentation_view.initial_segments["endocardium"] is None
    assert segmentation_view.initial_segments["epicardium"] is None
    assert segmentation_view.final_segments["endocardium"] is None
    assert segmentation_view.final_segments["epicardium"] is None


def test_scroll(segmentation_view):
    import numpy as np

    segmentation_view.images["mag"] = np.random.random((5, 5, 2))
    contour = np.random.random((2, 2, 5))

    segmentation_view.final_segments["endocardium"] = contour
    segmentation_view.final_segments["epicardium"] = contour

    frame, img, (endo, epi, centroid) = segmentation_view.scroll(1, "mag")

    assert frame == 1
    assert img == approx(segmentation_view.images["mag"][1])
    assert endo == approx(contour[1])
    assert epi == approx(contour[1])


def test_contour_edited_and_undo(segmentation_view, strainmap_data):
    import numpy as np

    contour = np.random.random((2, 2, 5))
    dataset = list(strainmap_data.data_files.keys())[0]

    segmentation_view.data = strainmap_data
    segmentation_view.data.segments[dataset]["endocardium"] = contour
    segmentation_view.data.segments[dataset]["epicardium"] = contour

    segmentation_view.plot_segments(dataset)

    contour_mod = 2 * contour[0]
    axes = segmentation_view.fig.axes[0]

    segmentation_view.contour_edited("endocardium", axes, contour_mod)

    assert len(segmentation_view.undo_stack) == 1
    assert segmentation_view.final_segments["endocardium"][0] == approx(contour_mod)
    assert segmentation_view.fig.axes[1].lines[0].get_data() == approx(contour_mod)

    segmentation_view.undo(0)

    assert len(segmentation_view.undo_stack) == 0
    assert segmentation_view.final_segments["endocardium"][0] == approx(contour[0])
    assert segmentation_view.fig.axes[0].lines[0].get_data() == approx(contour[0])
    assert segmentation_view.fig.axes[1].lines[0].get_data() == approx(contour[0])
