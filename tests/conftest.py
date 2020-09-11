from pathlib import Path
from unittest.mock import patch, MagicMock

from pytest import fixture
import pandas as pd
import xarray as xr


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
    return Path(__file__).parent / "data" / "CM1"


@fixture(scope="session")
def dicom_data_path():
    """Returns the DICOM data path."""
    return _dicom_data_path()


@fixture
def h5_file_path(tmpdir):
    """Returns the DICOM data path."""
    from shutil import copyfile

    destination = tmpdir / "CM1_analysis.h5"
    copyfile(Path(__file__).parent / "data" / "CM1_analysis.h5", destination)
    return destination


@fixture
def data_tree(strainmap_data) -> xr.DataArray:
    """Returns the DICOM directory data tree."""
    return strainmap_data.data_files.files


@fixture
def strainmap_data(dicom_data_path):
    """Returns a loaded StrainMapData object."""
    from strainmap.models.strainmap_data_model import StrainMapData

    return StrainMapData.from_folder(data_files=dicom_data_path)


@fixture
def segmented_data(strainmap_data):
    """Returns a StrainMapData object with segmented data."""
    from strainmap.models.contour_mask import Contour
    from strainmap.models.quick_segmentation import find_segmentation
    import numpy as np

    dataset = strainmap_data.data_files.datasets[0]
    image = strainmap_data.data_files.mag(dataset)

    # Create the initial contour
    init_epi = Contour.circle(center=(270, 308), radius=50, shape=image.shape).xy
    init_endo = Contour.circle(center=(270, 308), radius=30, shape=image.shape).xy

    # Launch the segmentation process
    find_segmentation(
        data=strainmap_data,
        dataset_name=dataset,
        images={"endocardium": image, "epicardium": image},
        frame=None,
        initials={"epicardium": init_epi, "endocardium": init_endo},
    )

    strainmap_data.zero_angle[dataset][..., 0] = np.array([260, 230])
    return strainmap_data


@fixture
def registered_views():
    from strainmap.gui.data_view import DataTaskView
    from strainmap.gui.segmentation_view import SegmentationTaskView
    from strainmap.gui.atlas_view import AtlasTaskView

    return [DataTaskView, SegmentationTaskView, AtlasTaskView]


@fixture
def control_with_mock_window(registered_views):
    from strainmap.controller import StrainMap

    StrainMap.registered_views = registered_views
    with patch("strainmap.gui.base_window_and_task.MainWindow", autospec=True):
        return StrainMap()


@fixture
def main_window():
    from strainmap.gui.base_window_and_task import MainWindow

    root = MainWindow()
    root.withdraw()

    yield root
    root.destroy()


@fixture
def data_view(main_window):
    from strainmap.gui.data_view import DataTaskView
    from strainmap.controller import StrainMap
    import weakref

    return DataTaskView(main_window, weakref.ref(StrainMap))


@fixture
def segmentation_view(main_window):
    from strainmap.gui.segmentation_view import SegmentationTaskView
    from strainmap.controller import StrainMap
    import weakref

    return SegmentationTaskView(main_window, weakref.ref(StrainMap))


@fixture
def velocities_view(main_window):
    from strainmap.gui.velocities_view import VelocitiesTaskView
    from strainmap.controller import StrainMap
    import weakref

    return VelocitiesTaskView(main_window, weakref.ref(StrainMap))


@fixture
def atlas_view(main_window):
    from strainmap.gui.atlas_view import AtlasTaskView
    from strainmap.controller import StrainMap
    import weakref

    StrainMap.data = None
    return AtlasTaskView(main_window, weakref.ref(StrainMap))


@fixture
def actions_manager():
    import matplotlib

    matplotlib.use("Agg")
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
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure()
    fig.add_subplot()

    yield fig

    plt.close(fig)


@fixture
def markers():
    import numpy as np

    return np.array(
        [
            [
                [5, 5.0, 145.83333333333334],
                [20, -10.0, 486.8421052631579],
                [38, -5.0, 794.7368421052631],
                [0, 0, 0],
            ],
            [
                [5, 2.0000000000000124, 145.83333333333334],
                [20, -10.0, 486.8421052631579],
                [38, -5.0, 794.7368421052631],
                [14, 3.116736209871314e-07, 350.0],
            ],
            [
                [3, -1.9990386376492923, 87.5],
                [8, 1.9990386376492724, 233.33333333333331],
                [18, -2.9999999999999867, 452.63157894736844],
                [0, 0, 0],
            ],
        ]
    )


@fixture
def velocity(markers):
    import numpy as np

    velocities = np.zeros((3, 50))
    idx = np.arange(0, 50)
    for i in range(3):
        for centre, amplitude, _ in markers[i]:
            velocities[i] += amplitude * np.exp(-((idx - centre) ** 2) / 3)

    return velocities


@fixture
def data_with_velocities(segmented_data):
    from strainmap.models.velocities import calculate_velocities
    from copy import deepcopy

    dataset_name = segmented_data.data_files.datasets[0]
    output = deepcopy(segmented_data)
    calculate_velocities(
        output,
        dataset_name,
        global_velocity=True,
        angular_regions=(6, 24),
        radial_regions=(4,),
    )
    return output


@fixture
def larray_np():
    from strainmap.models.sm_data import LabelledArray
    import numpy as np

    dims = ("rows", "cols", "depth")
    coords = {"cols": ["x", "y", "y"], "depth": ["shallow", "mid", "deep", "very deep"]}
    values = np.random.random((3, 3, 4))
    values[values < 0.5] = 0

    return LabelledArray(dims, coords, values)


@fixture
def larray_coo(larray_np):
    from strainmap.models.sm_data import LabelledArray
    import sparse

    values = sparse.COO.from_numpy(larray_np.values)

    return LabelledArray(larray_np.dims, larray_np.coords, values)


@fixture(params=["np", "COO"])
def larray(request, larray_np, larray_coo):
    return larray_np if request.param == "np" else larray_coo


@fixture
def dummy_data() -> pd.DataFrame:
    """Create dataframe with some dummy data."""
    from strainmap.gui.atlas_view import SLICES, COMP, REGIONS, validate_data
    import numpy as np

    M = 30
    N = 9 * 7 * M

    Record = pd.Series(np.random.randint(1, 20+1, N))
    Slice = pd.Series(np.random.choice(SLICES, size=N))
    Component = pd.Series(np.random.choice(COMP, size=N))
    Region = pd.Series(np.random.choice([a for a in REGIONS if a != ""], size=N))
    PSS = pd.Series(np.random.random(N))
    ESS = pd.Series(np.random.random(N))
    PS = pd.Series(np.random.random(N))
    Included = pd.Series([True] * N)

    return validate_data(
        pd.DataFrame(
            {
                "Record": Record,
                "Slice": Slice,
                "Region": Region,
                "Component": Component,
                "PSS": PSS,
                "ESS": ESS,
                "PS": PS,
                "Included": Included,
            }
        )
    )


@fixture
def atlas_view_with_data(atlas_view, dummy_data):
    atlas_view.save_atlas = MagicMock()
    atlas_view.update_plots = MagicMock()
    atlas_view.atlas_data = dummy_data
    atlas_view.update_table(atlas_view.atlas_data)
    return atlas_view
