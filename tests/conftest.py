from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import xarray as xr
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
    from strainmap.models.segmentation import new_segmentation
    import numpy as np
    from strainmap.coordinates import Comp

    cine = strainmap_data.data_files.datasets[0]
    image = strainmap_data.data_files.images(cine).sel(comp=Comp.MAG)

    # Create the initial contour
    initial_segments = xr.DataArray(
        np.array(
            [
                Contour.circle(center=(270, 308), radius=50, shape=image.shape).xy.T,
                Contour.circle(center=(270, 308), radius=30, shape=image.shape).xy.T,
            ]
        ),
        dims=("side", "coord", "point"),
        coords={"side": ["endocardium", "epicardium"], "coord": ["row", "col"]},
    )

    # Create septum
    septum = xr.DataArray(
        np.full((image.sizes["frame"], 2), np.nan),
        dims=("frame", "coord"),
        coords={"coord": ["row", "col"], "frame": np.arange(image.sizes["frame"])},
    )

    # Launch the segmentation process
    new_segmentation(
        data=strainmap_data,
        cine=cine,
        frame=None,
        initials=initial_segments,
        new_septum=septum,
    )

    strainmap_data.septum.loc[{"cine": cine}] = np.array([260, 230])
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
    from tkinter import _default_root

    if _default_root is not None:
        _default_root.destroy()
        _default_root = None

    root = MainWindow()
    root.withdraw()

    return root


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
def velocities_view(main_window, data_with_velocities):
    from strainmap.gui.velocities_view import VelocitiesTaskView
    from strainmap.controller import StrainMap
    import weakref

    StrainMap.data = data_with_velocities
    StrainMap.window = main_window
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

    Record = pd.Series(np.random.randint(1, 20 + 1, N))
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


@fixture
def segments_arrays():
    import numpy as np
    import xarray as xr
    from strainmap.models.segmentation import _init_septum_and_centroid, _init_segments

    frames, points = np.random.randint(20, 51, 2)
    segments = xr.concat(
        [
            _init_segments("apex", frames, points),
            _init_segments("mid", frames, points),
            _init_segments("base", frames, points),
        ],
        dim="cine",
    )
    septum = xr.concat(
        [
            _init_septum_and_centroid("apex", frames, "septum"),
            _init_septum_and_centroid("mid", frames, "septum"),
            _init_septum_and_centroid("base", frames, "septum"),
        ],
        dim="cine",
    )
    centroid = xr.concat(
        [
            _init_septum_and_centroid("apex", frames, "centroid"),
            _init_septum_and_centroid("mid", frames, "centroid"),
            _init_septum_and_centroid("base", frames, "centroid"),
        ],
        dim="cine",
    )
    return segments, septum, centroid


@fixture
def theta0(segmented_data):
    from strainmap.models.velocities import theta_origin

    centroid = segmented_data.centroid.isel(cine=0)
    septum = segmented_data.septum.isel(cine=0)
    return theta_origin(centroid, septum)


@fixture
def masks(segmented_data, theta0):
    from strainmap.models.velocities import find_masks

    cine = segmented_data.data_files.datasets[0]
    image = segmented_data.data_files.images(cine)
    shape = image.sizes["row"], image.sizes["col"]
    segments = segmented_data.segments.isel(cine=0)
    centroid = segmented_data.centroid.isel(cine=0)
    return find_masks(segments, centroid, theta0, shape)


@fixture
def empty_markers(regions):
    from strainmap.models.velocities import Mark
    from strainmap.coordinates import Comp
    import numpy as np
    components = [Comp.LONG, Comp.RAD, Comp.CIRC]
    quantities = ["frame", "velocity", "time"]
    return xr.DataArray(
        np.full((len(regions), len(components), len(Mark), len(quantities)), np.nan),
        dims=["region", "comp", "marker", "quantity"],
        coords={
            "region": regions,
            "comp": components,
            "marker": list(Mark),
            "quantity": quantities,
        },
    )


@fixture
def regions():
    from itertools import chain
    from strainmap.coordinates import Region
    return list(chain.from_iterable(([r] * r.value for r in Region)))


@fixture
def velocities(regions):
    from strainmap.coordinates import Comp
    import numpy as np

    components = [Comp.LONG, Comp.RAD, Comp.CIRC]
    frames = 50
    output = xr.DataArray(
        np.full((len(regions), frames, len(components)), np.nan),
        dims=["region", "frame", "comp"],
        coords={
            "region": regions,
            "frame": np.arange(frames),
            "comp": components,
        },
    )
    output[...] = np.sin(np.linspace(0, 2 * np.pi, frames))[None, :, None]
    return output
