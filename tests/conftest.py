from pathlib import Path

from pytest import fixture


@fixture
def dicom_data_path():
    """ Returns the DICOM data path. """
    return Path(__file__).parent / "data" / "SUB1"


@fixture
def dicom_bg_data_path():
    """ Returns the DICOM background data path. """
    return Path(__file__).parent / "data" / "SUB1_BG"
