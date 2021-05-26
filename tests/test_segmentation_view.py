from unittest.mock import MagicMock, PropertyMock, patch

from pytest import approx


def test_update_and_clear_widgets(segmentation_view, strainmap_data):
    segmentation_view.cine_changed = MagicMock()
    segmentation_view.controller.data = strainmap_data
    segmentation_view.controller.review_mode = False
    segmentation_view.controller.progress = MagicMock()
    segmentation_view.update_widgets()

    expected = strainmap_data.data_files.datasets[0]
    assert segmentation_view.cines_var.get() == expected

    segmentation_view.cine_changed.assert_called_once()


def test_update_plots(segmentation_view, strainmap_data):
    segmentation_view.plot_images = MagicMock()
    segmentation_view.plot_segments = MagicMock()
    segmentation_view.controller.data = strainmap_data
    segmentation_view.controller.review_mode = False
    segmentation_view.update_widgets()

    segmentation_view.plot_images.assert_called_once()
    segmentation_view.plot_segments.assert_called_once()

    assert segmentation_view.ax_mag.get_title("right") == "Magnitude"
    assert segmentation_view.ax_vel.get_title("right") == "Velocity"


def test_plot_images(segmentation_view, strainmap_data):
    segmentation_view.controller.data = strainmap_data
    segmentation_view.controller.review_mode = False
    segmentation_view.update_widgets()

    generators = segmentation_view.fig.actions_manager.ScrollFrames._scroller

    assert segmentation_view.ax_mag in generators
    assert segmentation_view.ax_vel in generators


def test_switch_mark_state(segmentation_view):
    expected = "\u2705 Endocardium"
    actual = segmentation_view.endo_redy_var.get()
    assert expected != actual

    segmentation_view.switch_mark_state("endocardium", "ready")
    actual = segmentation_view.endo_redy_var.get()
    assert expected == actual


def test_define_initial_contour(segmentation_view, strainmap_data):
    from copy import deepcopy

    segmentation_view.controller.data = deepcopy(strainmap_data)
    segmentation_view.controller.review_mode = False
    segmentation_view.update_widgets()

    segmentation_view.set_initial_contour("endocardium")

    assert "get_contour" in str(
        segmentation_view.fig.actions_manager.DrawContours.contours_updated
    )


def test_get_contour(segmentation_view):
    import numpy as np

    segmentation_view.controller.review_mode = False
    segmentation_view.initialization = iter((lambda: None,))
    contour = np.random.random((2, 5))
    points = np.random.random((2, 2))
    segmentation_view.get_contour([contour], points=points, side="endocardium")
    assert (
        segmentation_view.initial_segments.sel(side="endocardium") == approx(contour)
    ).all()


def test_initialize_segmentation(segmentation_view, strainmap_data):
    import numpy as np
    from copy import deepcopy

    segmentation_view.controller.data = deepcopy(strainmap_data)
    segmentation_view.controller.review_mode = False
    segmentation_view.update_widgets()

    segmentation_view.first_frame = MagicMock()
    contour = np.random.random((2, 5))
    points = np.random.random((2, 2))
    assert segmentation_view.next_btn.instate(["disabled"])

    segmentation_view.initialize_segmentation()
    segmentation_view.get_contour([contour], points=points, side="endocardium")
    segmentation_view.get_contour([contour], points=points, side="epicardium")
    segmentation_view.get_septum(None, [np.random.random(2)])
    assert segmentation_view.next_btn.instate(["!disabled"])
    segmentation_view.first_frame.assert_called_once()


@patch(
    "strainmap.gui.segmentation_view.SegmentationTaskView.septum",
    new_callable=PropertyMock,
)
@patch(
    "strainmap.gui.segmentation_view.SegmentationTaskView.centroid",
    new_callable=PropertyMock,
)
def test_quick_segmentation(septum, centroid, segmentation_view, strainmap_data):
    import numpy as np
    from copy import deepcopy

    segmentation_view.controller.data = deepcopy(strainmap_data)
    segmentation_view.controller.review_mode = False
    segmentation_view.update_widgets()

    segmentation_view.first_frame = MagicMock()
    segmentation_view.new_segmentation = MagicMock()
    segmentation_view.go_to_frame = MagicMock()
    centroid.return_value = 0
    septum.return_value = 0
    segmentation_view.septum = np.zeros((3, 2, 2))

    segmentation_view.segment_mode_var.set("Automatic")
    contour = np.random.random((2, 5))
    points = np.random.random((2, 2))

    segmentation_view.initialize_segmentation()
    segmentation_view.get_contour([contour], points=points, side="endocardium")
    segmentation_view.get_contour([contour], points=points, side="epicardium")
    segmentation_view.get_septum(None, [np.random.random(2)])
    assert segmentation_view.next_btn.instate(["disabled"])
    assert segmentation_view.working_frame_var.get() == 2
    segmentation_view.first_frame.assert_not_called()


def test_plot_segments(segmentation_view, strainmap_data):
    import numpy as np
    from copy import deepcopy
    from strainmap.models.segmentation import _init_segments

    contour = np.random.random((2, 1, 2, 5))
    dataset = strainmap_data.data_files.datasets[0]

    segmentation_view.controller.data = deepcopy(strainmap_data)
    segmentation_view.controller.review_mode = False
    segmentation_view.update_widgets()

    segmentation_view.data.segments = _init_segments(cine=dataset, frames=1, points=5)
    segmentation_view.data.segments.loc[{"cine": dataset}] = contour

    segmentation_view.plot_segments(dataset)
    generators = segmentation_view.fig.actions_manager.ScrollFrames._scroller
    assert segmentation_view.ax_mag in generators
    assert segmentation_view.ax_vel in generators


def test_clear_segments(segmentation_view):
    import numpy as np
    import xarray as xr

    segmentation_view.remove_segmentation = MagicMock()
    contour = np.random.random((2, 5))

    segmentation_view.initial_segments = xr.DataArray(contour)
    segmentation_view._segments = xr.DataArray(contour)

    segmentation_view.clear_segment_variables()

    xr.testing.assert_equal(segmentation_view.initial_segments, xr.DataArray())
    xr.testing.assert_equal(segmentation_view._segments, xr.DataArray())
    segmentation_view.remove_segmentation.assert_called_once()


def test_scroll(segmentation_view, strainmap_data):
    import numpy as np
    import xarray as xr
    from strainmap.models.segmentation import _init_septum_and_centroid
    from strainmap.coordinates import Comp

    dataset = strainmap_data.data_files.datasets[0]
    segmentation_view.controller.data = strainmap_data
    segmentation_view.controller.review_mode = False

    contour = np.random.random((2, 2, 2, 5))
    segmentation_view._segments = xr.DataArray(
        contour,
        dims=["side", "frame", "coord", "point"],
        coords={"side": ["endocardium", "epicardium"]},
    )

    # segmentation_view.update_widgets()
    rdata = np.random.random((1, 2, 5, 5))
    segmentation_view.data.data_files.images = lambda x: xr.DataArray(
        rdata, dims=["comp", "frame", "row", "col"], coords={"comp": [Comp.MAG]}
    )
    segmentation_view._septum = _init_septum_and_centroid(
        cine=dataset, frames=2, name="septum"
    ).sel(cine=dataset)
    segmentation_view._septum[...] = np.random.random((2, 2))

    frame, img, (endo, epi, septum_line, septum) = segmentation_view.scroll(1, Comp.MAG)

    assert frame == 1
    assert (img == segmentation_view.images.sel(comp=Comp.MAG, frame=1)).all()
    assert endo == approx(contour[0, 1])
    assert epi == approx(contour[1, 1])
    assert approx(septum) == segmentation_view._septum.sel(frame=1)
    assert septum_line == approx(np.array([septum, segmentation_view.centroid]).T)


def test_contour_edited_and_undo(segmentation_view, strainmap_data):
    from strainmap.models.segmentation import _init_septum_and_centroid, _init_segments
    import numpy as np

    contour = np.random.random((2, 2, 2, 5))
    dataset = strainmap_data.data_files.datasets[0]

    segmentation_view.controller.data = strainmap_data
    segmentation_view.controller.review_mode = False
    segmentation_view.update_widgets()

    segmentation_view.data.segments = _init_segments(cine=dataset, frames=2, points=5)
    segmentation_view.data.septum = _init_septum_and_centroid(
        cine=dataset, frames=2, name="septum"
    )
    segmentation_view.data.septum.loc[{"cine": dataset}] = np.random.random((2, 2))
    segmentation_view.data.segments.loc[{"cine": dataset}] = contour

    segmentation_view.plot_segments(dataset)
    segmentation_view.plot_septum(dataset)

    contour_mod = 2 * contour[0, 0]
    axes = segmentation_view.fig.axes[0]

    segmentation_view.contour_edited("endocardium", axes, contour_mod)

    assert len(segmentation_view.undo_stack) == 1
    assert segmentation_view._segments.sel(side="endocardium", frame=0).data == approx(
        contour_mod
    )

    for ax in segmentation_view.fig.axes:
        for line in ax.lines:
            if line.get_label() == "endocardium":
                assert line.get_data() == approx(contour_mod)

    segmentation_view.undo(0)

    assert len(segmentation_view.undo_stack) == 0
    assert segmentation_view._segments.sel(side="endocardium", frame=0).data == approx(
        contour[0, 0]
    )
    for ax in segmentation_view.fig.axes:
        for line in ax.lines:
            if line.get_label() == "endocardium":
                assert line.get_data() == approx(contour[0, 0])


def test_next_frames(segmentation_view, strainmap_data):
    segmentation_view.controller.data = strainmap_data
    segmentation_view.new_segmentation = MagicMock()
    segmentation_view.refresh_data = MagicMock()
    segmentation_view.replot = MagicMock()
    segmentation_view.update_and_find_next = MagicMock()
    segmentation_view.go_to_frame = MagicMock()

    # First frame
    segmentation_view.first_frame()
    segmentation_view.new_segmentation.assert_called_with(0, unlock=False)
    assert segmentation_view.go_to_frame.call_count == 1

    # Other frames
    segmentation_view.next()
    segmentation_view.update_and_find_next.assert_called()
    assert segmentation_view.go_to_frame.call_count == 2


def test_finish_segmentation(segmentation_view):
    segmentation_view.controller.update_segmentation = MagicMock()

    segmentation_view.finish_segmentation()
    segmentation_view.controller.update_segmentation.assert_called()
    assert segmentation_view.completed
