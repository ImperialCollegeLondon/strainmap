from unittest.mock import MagicMock


def test_creation(control_with_mock_window):
    from strainmap.controller import Requisites

    assert control_with_mock_window.achieved == Requisites.NONE
    assert control_with_mock_window.window.add.call_count == 1

    control_with_mock_window.window.reset_mock()


def test_unlock_lock(control_with_mock_window):
    from strainmap.controller import Requisites

    control_with_mock_window.unlock(Requisites.DATALOADED)
    assert control_with_mock_window.achieved == Requisites.NONE | Requisites.DATALOADED
    assert control_with_mock_window.window.add.call_count == 3

    control_with_mock_window.lock(Requisites.DATALOADED)
    assert control_with_mock_window.achieved == Requisites.NONE

    control_with_mock_window.window.reset_mock()


def test_update_views(control_with_mock_window):

    view = control_with_mock_window.registered_views[0]
    control_with_mock_window.window.views = [MagicMock(view)]
    control_with_mock_window.update_views()
    assert control_with_mock_window.window.views[0].update_widgets.called


def test_load_data(control_with_mock_window, dicom_data_path, data_view):

    control_with_mock_window.unlock = MagicMock()
    data_view.update_widgets = MagicMock()

    control_with_mock_window.load_data_from_folder(data_files=dicom_data_path)

    assert control_with_mock_window.unlock.call_count == 1
    assert len(control_with_mock_window.data.data_files.datasets) > 0

    control_with_mock_window.window.reset_mock()


def test_clear_data(control_with_mock_window):
    control_with_mock_window.data = "Dummy"
    control_with_mock_window.lock = MagicMock()
    control_with_mock_window.clear_data()

    assert control_with_mock_window.lock.call_count == 3
    assert control_with_mock_window.data is None

    control_with_mock_window.window.reset_mock()
