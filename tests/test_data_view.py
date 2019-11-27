from unittest.mock import MagicMock

from .conftest import patch_dialogs


@patch_dialogs
def test_load_data_button(data_view, dicom_data_path):
    from strainmap.gui.base_window_and_task import EVENTS

    EVENTS["load_data_from_folder"] = MagicMock()

    data_view.nametowidget("control.chooseDataFolder").invoke()
    assert EVENTS["load_data_from_folder"].call_count == 1
    assert data_view.data_folder.get() == str(dicom_data_path)


@patch_dialogs
def test_clear_data_button(data_view, dicom_data_path):
    from strainmap.gui.base_window_and_task import EVENTS

    EVENTS["clear_data"] = MagicMock()

    data_view.nametowidget("control.clearAllData").invoke()
    assert EVENTS["clear_data"].call_count == 1


def test_update_and_clear_widgets(data_view, strainmap_data):
    data_view._controller().data = strainmap_data
    data_view.update_widgets()

    assert data_view.nametowidget("visualise.notebook")
    assert len(data_view.notebook.tabs()) == 2
    assert data_view.nametowidget("control.chooseOutputFile")["state"] == "enable"
