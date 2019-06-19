from unittest.mock import MagicMock


def test_load_data(control_with_mock_window, dicom_data_path):
    from strainmap.actions import load_data

    control_with_mock_window.unlock = MagicMock()

    load_data(control_with_mock_window, data_files=dicom_data_path)

    assert control_with_mock_window.unlock.call_count == 1
    assert len(control_with_mock_window.data.data_files.keys()) == 3

    control_with_mock_window.window.reset_mock()


def test_clear_data(control_with_mock_window):
    from strainmap.actions import clear_data

    control_with_mock_window.data = "Dummy"
    control_with_mock_window.lock = MagicMock()

    clear_data(control_with_mock_window)

    assert control_with_mock_window.lock.call_count == 1
    assert control_with_mock_window.data is None

    control_with_mock_window.window.reset_mock()
