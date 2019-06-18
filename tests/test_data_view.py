from pytest import mark


def test_load_data_button(data_view, dicom_data_path):

    data_view.control.children["newSeries"].children["chooseDataFolder"].invoke()
    assert data_view.actions.load_data.call_count == 1
    assert data_view.data_folder.get() == str(dicom_data_path)

    data_view.actions.load_data.reset_mock()


@mark.skip("Functionality ot implemented, yet.")
def test_resume_button(data_view):
    pass


@mark.skip("Functionality ot implemented, yet.")
def test_output_button(data_view):
    pass


def test_clear_data_button(data_view):

    data_view.control.children["clearAllData"].invoke()
    assert data_view.actions.clear_data.call_count == 1

    data_view.actions.clear_data.reset_mock()
