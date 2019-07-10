from unittest.mock import MagicMock

from pytest import mark

from conftest import patch_dialogs


@patch_dialogs
def test_load_data_button(data_view, dicom_data_path):
    from strainmap.gui.base_window_and_task import EVENTS

    EVENTS["load_data"] = MagicMock()

    data_view.nametowidget("control.chooseDataFolder").invoke()
    assert EVENTS["load_data"].call_count == 1
    assert data_view.data_folder.get() == str(dicom_data_path)


@patch_dialogs
def test_clear_data_button(data_view, dicom_data_path):
    from strainmap.gui.base_window_and_task import EVENTS

    EVENTS["clear_data"] = MagicMock()

    data_view.nametowidget("control.clearAllData").invoke()
    assert EVENTS["clear_data"].call_count == 1


@mark.skip("Functionality ot implemented, yet.")
def test_resume_button(data_view):
    pass


@mark.skip("Functionality ot implemented, yet.")
def test_output_button(data_view):
    pass


def test_update_and_clear_widgets(data_view, strainmap_data):

    data_view.data = strainmap_data

    assert data_view.nametowidget("visualise.notebook")
    assert len(data_view.notebook.tabs()) == 2
    assert data_view.nametowidget("control.chooseOutputFile")["state"] == "enable"

    data_view.data = None

    assert not data_view.notebook
    assert (
        str(data_view.nametowidget("control.chooseOutputFile")["state"]) == "disabled"
    )


def test_update_visualization_plot(data_view, strainmap_data):

    assert not data_view.anim
    data_view.data = strainmap_data
    assert data_view.anim

    # Stop animation when the timestep is manually changed
    data_view.update_visualization(step_changed=True)
    assert data_view.anim and not data_view.anim.event_source

    # Resume animation if a new variable (or slice) is selected
    data_view.update_visualization()
    assert data_view.anim and data_view.anim.event_source


def test_update_visualization_dicom(data_view, strainmap_data):

    data_view.update_tree = MagicMock()
    data_view.data = strainmap_data

    data_view.notebook.hide(data_view.notebook.select())

    # Update the information tree when a new timestep is selected
    data_view.update_visualization(step_changed=True)
    assert data_view.update_tree.call_count == 1

    # And also if a new variable (or slice) is selected
    data_view.update_visualization()
    assert data_view.update_tree.call_count == 2
