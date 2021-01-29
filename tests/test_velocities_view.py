from unittest.mock import MagicMock
from pytest import approx


def test_update_widgets(velocities_view, segmented_data, data_with_velocities):
    velocities_view.calculate_velocities = MagicMock()
    velocities_view.update_velocities_list = MagicMock()
    velocities_view.replot = MagicMock()

    velocities_view.controller.data = segmented_data
    velocities_view.update_widgets()

    expected = segmented_data.data_files.datasets[0]
    assert velocities_view.datasets_var.get() == expected
    assert velocities_view.calculate_velocities.call_count == 2
    assert velocities_view.replot.call_count == 1

    velocities_view.controller.data = data_with_velocities
    velocities_view.update_widgets()
    velocities_view.update_velocities_list.assert_called_once()
    assert velocities_view.replot.call_count == 2


def test_bg_changed(velocities_view, data_with_velocities):
    velocities_view.calculate_velocities = MagicMock()
    velocities_view.controller.data = data_with_velocities
    velocities_view.update_widgets()
    velocities_view.bg_var.set("None")
    velocities_view.bg_changed()
    velocities_view.calculate_velocities.assert_called_once()


def test_reversal_checked(velocities_view, data_with_velocities):
    velocities_view.calculate_velocities = MagicMock()
    velocities_view.controller.data = data_with_velocities
    velocities_view.update_widgets()

    velocities_view.reversal_checked()
    assert velocities_view.update_vel_btn.instate(["disabled"])

    velocities_view.reverse_vel_var[0].set(True)
    velocities_view.reversal_checked()
    assert velocities_view.update_vel_btn.instate(["!disabled"])


def test_recalculate_velocities(velocities_view, data_with_velocities):
    velocities_view.calculate_velocities = MagicMock()
    velocities_view.controller.data = data_with_velocities
    velocities_view.update_widgets()
    velocities_view.recalculate_velocities()
    velocities_view.calculate_velocities.assert_called_once()


def test_scroll(velocities_view, data_with_velocities):
    velocities_view.calculate_velocities = MagicMock()
    velocities_view.controller.data = data_with_velocities
    velocities_view.update_widgets()
    vel = list(list(data_with_velocities.velocities.values())[0].keys())[1]
    velocities_view.velocities_var.set(vel)

    assert velocities_view.current_region == 0
    region, _, _ = velocities_view.scroll()
    assert region == 1


def test_color_plot(velocities_view, data_with_velocities):
    velocities_view.calculate_velocities = MagicMock()
    velocities_view.controller.data = data_with_velocities
    velocities_view.update_widgets()
    velocities_view.velocities_var.set("angular x24 - Estimated")
    velocities_view.replot()

    expected = list(data_with_velocities.velocities.values())[0][
        "angular x24 - Estimated"
    ][:, 0, :]
    actual = velocities_view.fig.axes[0].images[0].get_array().data

    assert len(velocities_view.fig.axes) == 6  # 3 axes + 3 colorbars
    assert actual == approx(expected)


def test_change_orientation(velocities_view):
    velocities_view.data.orientation = "CCW"
    velocities_view.data.set_orientation = MagicMock()
    velocities_view.populate_tables = MagicMock()

    velocities_view.change_orientation()
    assert velocities_view.data.set_orientation.called_once()
    assert velocities_view.populate_tables.called_once()
