from pathlib import Path
from unittest.mock import patch

from pytest import fixture

patch("tkinter.filedialog.askdirectory", lambda *x, **y: _dicom_data_path()).start()
patch("tkinter.messagebox.askokcancel", lambda *x, **y: "ok").start()
patch("tkinter.messagebox.showinfo", lambda *x, **y: "ok").start()


def _dicom_data_path():
    """ Returns the DICOM data path. """
    return Path(__file__).parent / "data" / "SUB1"


@fixture(scope="session")
def dicom_data_path():
    """ Returns the DICOM data path. """
    return _dicom_data_path()


@fixture(scope="session")
def data_tree(dicom_data_path):
    """ Returns the DICOM directory data tree. """
    from strainmap.models.readers import read_dicom_directory_tree

    return read_dicom_directory_tree(dicom_data_path)


@fixture(scope="session")
def strainmap_data(dicom_data_path):
    """ Returns a loaded StrainMapData object. """
    from strainmap.models.strainmap_data_model import factory

    return factory(data_files=dicom_data_path)


@fixture(scope="session")
def dicom_bg_data_path():
    """ Returns the DICOM background data path. """
    return Path(__file__).parent / "data" / "SUB1_BG"


@fixture
def registered_views():
    from strainmap.gui.data_view import DataTaskView
    from strainmap.gui.segmentation_view import SegmentationTaskView

    return [DataTaskView, SegmentationTaskView]


@fixture
@patch("strainmap.gui.base_window_and_task.MainWindow", autospec=True)
def control_with_mock_window(MockWindow, registered_views):
    from strainmap.controller import StrainMap

    StrainMap.registered_views = registered_views
    return StrainMap()


@fixture(scope="session")
def main_window():
    from strainmap.gui.base_window_and_task import MainWindow

    root = MainWindow()
    root.withdraw()

    return root


@fixture
def empty_view():
    from strainmap.gui.base_window_and_task import TaskViewBase

    class TestView(TaskViewBase):
        def __init__(self, root):
            super().__init__(root)

        def update_widgets(self):
            pass

        def clear_widgets(self):
            pass

    return TestView


@fixture
def data_view(main_window):
    from strainmap.gui.data_view import DataTaskView

    return DataTaskView(main_window)
