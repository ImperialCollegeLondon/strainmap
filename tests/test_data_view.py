from pytest import mark


def test_load_data_button(data_view, dicom_data_path):

    data_view.nametowidget("control.chooseDataFolder").invoke()

    assert data_view.actions.load_data.call_count == 1
    assert data_view.data_folder.get() == str(dicom_data_path)

    data_view.actions.load_data.reset_mock()


def test_clear_data_button(data_view):

    data_view.nametowidget("control.clearAllData").invoke()
    assert data_view.actions.clear_data.call_count == 1

    data_view.actions.clear_data.reset_mock()


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
    assert data_view.data_folder.get() == ""
