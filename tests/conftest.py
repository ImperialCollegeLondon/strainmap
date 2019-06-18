from pathlib import Path
from unittest.mock import patch

from pytest import fixture


@fixture
def dicom_data_path():
    """ Returns the DICOM data path. """
    return Path(__file__).parent / "data" / "SUB1"


@fixture
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


@fixture
def main_window():
    from strainmap.gui.base_classes import MainWindow

    return MainWindow()


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
