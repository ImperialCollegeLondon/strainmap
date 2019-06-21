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
    from strainmap.gui.data_view import DataView
    from strainmap.gui.segmentation_view import SegmentationView

    return [DataView, SegmentationView]


@fixture
@patch("strainmap.gui.base_classes.MainWindow", autospec=True)
def control_with_mock_window(MockWindow, registered_views):
    from strainmap.main import StrainMap

    StrainMap.registered_views = registered_views
    return StrainMap()


@fixture(scope="session")
def main_window():
    from strainmap.gui.base_classes import MainWindow

    root = MainWindow()
    root.withdraw()

    return root


@fixture
def empty_view():
    from strainmap.gui.base_classes import ViewBase

    class TestView(ViewBase):
        def __init__(self, root, actions):
            super().__init__(root, actions)

        def update_widgets(self):
            pass

        def clear_widgets(self):
            pass

    return TestView


@fixture
def data_view(main_window):
    from strainmap.gui.data_view import DataView
    from unittest.mock import MagicMock

    return DataView(main_window, {"load_data": MagicMock(), "clear_data": MagicMock()})
