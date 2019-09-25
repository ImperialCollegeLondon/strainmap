from pathlib import Path
from unittest.mock import patch

from pytest import fixture


def patch_dialogs(function):
    from functools import wraps

    @wraps(function)
    def decorated(*args, **kwargs):

        with patch(
            "tkinter.filedialog.askdirectory", lambda *x, **y: _dicom_data_path()
        ):
            with patch("tkinter.messagebox.askokcancel", lambda *x, **y: "ok"):
                with patch("tkinter.messagebox.showinfo", lambda *x, **y: "ok"):
                    return function(*args, **kwargs)

    return decorated


def _dicom_data_path():
    """Returns the DICOM data path."""
    return Path(__file__).parent / "data" / "SUB1"


@fixture(scope="session")
def dicom_data_path():
    """Returns the DICOM data path."""
    return _dicom_data_path()


@fixture(scope="session")
def data_tree(dicom_data_path):
    """Returns the DICOM directory data tree."""
    from strainmap.models.readers import read_dicom_directory_tree

    return read_dicom_directory_tree(dicom_data_path)


@fixture
def strainmap_data(dicom_data_path):
    """Returns a loaded StrainMapData object."""
    from strainmap.models.strainmap_data_model import factory

    return factory(data_files=dicom_data_path)


@fixture
def segmented_data(strainmap_data):
    """Returns a StrainMapData object with segmented data."""
    from strainmap.models.contour_mask import Contour
    from strainmap.models.quick_segmentation import find_segmentation

    dataset = list(strainmap_data.data_files.keys())[0]
    image = strainmap_data.get_images(dataset, "MagX")

    # Create the initial contour
    init_epi = Contour.circle(center=(310, 280), radius=60, shape=image.shape).xy
    init_endo = Contour.circle(center=(310, 280), radius=40, shape=image.shape).xy

    # Launch the segmentation process
    data = find_segmentation(
        data=strainmap_data,
        dataset_name=dataset,
        images={"endocardium": image, "epicardium": image},
        frame=None,
        initials={"epicardium": init_epi, "endocardium": init_endo},
    )

    return data


@fixture(scope="session")
def dicom_bg_data_path():
    """Returns the DICOM background data path."""
    return Path(__file__).parent / "data" / "SUB1_BG"


@fixture
def registered_views():
    from strainmap.gui.data_view import DataTaskView
    from strainmap.gui.segmentation_view import SegmentationTaskView

    return [DataTaskView, SegmentationTaskView]


@fixture
def control_with_mock_window(registered_views):
    from strainmap.controller import StrainMap

    StrainMap.registered_views = registered_views
    with patch("strainmap.gui.base_window_and_task.MainWindow", autospec=True):
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


@fixture
def segmentation_view(main_window):
    from strainmap.gui.segmentation_view import SegmentationTaskView

    return SegmentationTaskView(main_window)


@fixture
def velocities_view(main_window):
    from strainmap.gui.velocities_view import VelocitiesTaskView

    return VelocitiesTaskView(main_window)


@fixture
def actions_manager():
    from matplotlib.pyplot import figure
    from strainmap.gui.figure_actions_manager import FigureActionsManager

    fig = figure()
    return FigureActionsManager(fig)


@fixture
def action():
    from strainmap.gui.figure_actions_manager import (
        ActionBase,
        TriggerSignature,
        Location,
        MouseAction,
        Button,
    )

    s1 = TriggerSignature(Location.EDGE, Button.LEFT, MouseAction.MOVE)
    s2 = TriggerSignature(Location.N, Button.LEFT, MouseAction.MOVE)
    s3 = TriggerSignature(Location.CENTRE, Button.LEFT, MouseAction.MOVE)

    class DummyAction(ActionBase):
        def __init__(self):
            super().__init__(
                {s1: lambda *args: None, s2: lambda *args: None, s3: lambda *args: None}
            )

    return DummyAction


@fixture
def figure():
    import matplotlib.pyplot as plt

    fig = plt.figure()
    fig.add_subplot()

    yield fig

    plt.close(fig)


@fixture
def markers():
    return (
        {
            "PS": [5, 5.0, 145.83333333333334],
            "PD": [20, -10.0, 486.8421052631579],
            "PAS": [38, -5.0, 794.7368421052631],
        },
        {
            "PS": [5, 2.0000000000000124, 145.83333333333334],
            "PD": [20, -10.0, 486.8421052631579],
            "PAS": [38, -5.0, 794.7368421052631],
            "ES": [12, 3.116736209871314e-07, 350.0],
        },
        {
            "PC1": [3, -1.9990386376492923, 87.5],
            "PC2": [8, 1.9990386376492724, 233.33333333333331],
            "PC3": [18, -2.9999999999999867, 452.63157894736844],
        },
    )


@fixture
def velocity(markers):
    import numpy as np

    velocities = np.zeros((3, 50))
    idx = np.arange(0, 50)
    for i in range(3):
        for centre, amplitude, _ in markers[i].values():
            velocities[i] += amplitude * np.exp(-(idx - centre) ** 2 / 3)

    return velocities
