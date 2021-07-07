from unittest.mock import MagicMock

from .conftest import patch_dialogs


@patch_dialogs
def test_load_data_button(data_view):
    data_view.controller.load_data_from_folder = MagicMock()

    data_view.nametowidget("control.chooseDataFolder").invoke()
    assert data_view.controller.load_data_from_folder.call_count == 1
    assert data_view.data_folder.get() == ""


@patch_dialogs
def test_clear_data_button(data_view, dicom_data_path):
    data_view.controller.clear_data = MagicMock()

    data_view.nametowidget("control.clearAllData").invoke()
    assert data_view.controller.clear_data.call_count == 1


def test_update_and_clear_widgets(data_view, strainmap_data):
    data_view.controller.data = strainmap_data
    data_view.update_widgets()

    assert data_view.nametowidget("visualise.notebook")
    assert len(data_view.notebook.tabs()) == 2
