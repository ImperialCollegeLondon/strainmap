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
def mock_registered_views():
    from strainmap.gui.data_view import DataView
    from strainmap.gui.segmentation_view import SegmentationView

    return [DataView, SegmentationView]


@fixture
@patch("strainmap.gui.base_classes.MainWindow", autospec=True)
def control_with_mock_window(MockWindow, mock_registered_views):
    from strainmap.main import StrainMap

    StrainMap.registered_views = mock_registered_views
    return StrainMap()
