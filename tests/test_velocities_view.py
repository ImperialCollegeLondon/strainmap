from unittest.mock import MagicMock


def test_update_widgets(velocities_view, segmented_data, data_with_velocities):
    velocities_view.calculate_velocities = MagicMock()
    velocities_view.update_velocities_list = MagicMock()
    velocities_view.switch_velocity = MagicMock()

    velocities_view.data = segmented_data
    expected = list(segmented_data.data_files.keys())[0]
    assert velocities_view.datasets_var.get() == expected
    velocities_view.calculate_velocities.assert_called_once()

    velocities_view.data = data_with_velocities
    velocities_view.update_velocities_list.assert_called_once()
    velocities_view.switch_velocity.assert_called_once()


def test_bg_changed(velocities_view, data_with_velocities):
    velocities_view.calculate_velocities = MagicMock()
    velocities_view.data = data_with_velocities
    velocities_view.bg_var.set("None")
    velocities_view.bg_changed()
    velocities_view.calculate_velocities.assert_called_once()


def test_reversal_checked(velocities_view, data_with_velocities):
    velocities_view.calculate_velocities = MagicMock()
    velocities_view.data = data_with_velocities

    velocities_view.reversal_checked()
    assert velocities_view.update_vel_btn.instate(["disabled"])

    velocities_view.reverse_vel_var[0].set(True)
    velocities_view.reversal_checked()
    assert velocities_view.update_vel_btn.instate(["!disabled"])


def test_recalculate_velocities(velocities_view, data_with_velocities):
    velocities_view.calculate_velocities = MagicMock()
    velocities_view.data = data_with_velocities
    velocities_view.recalculate_velocities()
    velocities_view.calculate_velocities.assert_called_once()


def test_scroll(velocities_view, data_with_velocities):
    velocities_view.calculate_velocities = MagicMock()
    velocities_view.data = data_with_velocities
    vel = list(list(data_with_velocities.velocities.values())[0].keys())[1]
    velocities_view.velocities_var.set(vel)

    assert velocities_view.current_region == 0
    region, _, _ = velocities_view.scroll()
    assert region == 1
