from pathlib import Path
from unittest.mock import patch

import numpy as np
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
    import numpy as np

    from strainmap.coordinates import Comp
    from strainmap.models.contour_mask import Contour
    from strainmap.models.segmentation import new_segmentation

    cine = strainmap_data.data_files.datasets[0]
    image = strainmap_data.data_files.images(cine).sel(comp=Comp.MAG.name)

    # Create the initial contour
    initial_segments = xr.DataArray(
        np.array(
            [
                Contour.circle(center=(270, 308), radius=30, shape=image.shape).xy.T,
                Contour.circle(center=(270, 308), radius=50, shape=image.shape).xy.T,
            ]
        ),
        dims=("side", "coord", "point"),
        coords={"side": ["endocardium", "epicardium"], "coord": ["col", "row"]},
    )

    # Create septum
    septum = xr.DataArray(
        np.full((image.sizes["frame"], 2), np.nan),
        dims=("frame", "coord"),
        coords={"coord": ["col", "row"], "frame": np.arange(image.sizes["frame"])},
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

    return [DataTaskView, SegmentationTaskView]


@fixture
def control_with_mock_window(registered_views):
    from strainmap.controller import StrainMap

    StrainMap.registered_views = registered_views
    with patch("strainmap.gui.base_window_and_task.MainWindow", autospec=True):
        return StrainMap()


@fixture
def main_window():
    from tkinter import _default_root

    from strainmap.gui.base_window_and_task import MainWindow

    if _default_root is not None:
        _default_root.destroy()
        _default_root = None

    root = MainWindow()
    root.withdraw()

    return root


@fixture
def data_view(main_window):
    import weakref

    from strainmap.controller import StrainMap
    from strainmap.gui.data_view import DataTaskView

    return DataTaskView(main_window, weakref.ref(StrainMap))


@fixture
def segmentation_view(main_window):
    import weakref

    from strainmap.controller import StrainMap
    from strainmap.gui.segmentation_view import SegmentationTaskView

    return SegmentationTaskView(main_window, weakref.ref(StrainMap))


@fixture
def velocities_view(main_window, data_with_velocities):
    import weakref

    from strainmap.controller import StrainMap
    from strainmap.gui.velocities_view import VelocitiesTaskView

    StrainMap.data = data_with_velocities
    StrainMap.window = main_window
    return VelocitiesTaskView(main_window, weakref.ref(StrainMap))


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
        Button,
        Location,
        MouseAction,
        TriggerSignature,
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
    from copy import deepcopy

    from strainmap.models.velocities import calculate_velocities

    dataset_name = segmented_data.data_files.datasets[0]
    output = deepcopy(segmented_data)
    calculate_velocities(output, dataset_name)
    return output


@fixture
def segments_arrays():
    import numpy as np
    import xarray as xr

    from strainmap.models.segmentation import (
        _init_segments,
        _init_septum_and_centroid,
    )

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
    from strainmap.models.transformations import theta_origin

    centroid = segmented_data.centroid.isel(cine=0)
    septum = segmented_data.septum.isel(cine=0)
    return theta_origin(centroid, septum)


@fixture
def masks(segmented_data, theta0):
    from strainmap.models.velocities import find_masks

    segments = segmented_data.segments.isel(cine=0)
    centroid = segmented_data.centroid.isel(cine=0)
    return find_masks(segments, centroid, theta0)


@fixture
def empty_markers(regions):
    import numpy as np

    from strainmap.coordinates import Comp
    from strainmap.models.velocities import Mark

    components = [Comp.LONG.name, Comp.RAD.name, Comp.CIRC.name]
    quantities = ["frame", "velocity", "time"]
    return xr.DataArray(
        np.full((len(regions), len(components), len(Mark), len(quantities)), np.nan),
        dims=["region", "comp", "marker", "quantity"],
        coords={
            "region": regions,
            "comp": components,
            "marker": [m.name for m in Mark],
            "quantity": quantities,
        },
    )


@fixture
def regions():
    from itertools import chain

    from strainmap.coordinates import Region

    return list(chain.from_iterable(([r.name] * r.value for r in Region)))


@fixture
def velocities(regions):
    import numpy as np

    from strainmap.coordinates import Comp

    components = [Comp.LONG.name, Comp.RAD.name, Comp.CIRC.name]
    frames = 50
    output = xr.DataArray(
        np.full((len(regions), frames, len(components)), np.nan),
        dims=["region", "frame", "comp"],
        coords={"region": regions, "frame": np.arange(frames), "comp": components},
    )
    output[...] = np.sin(np.linspace(0, 2 * np.pi, frames))[None, :, None]
    return output


@fixture
def initial_centroid():
    import numpy as np

    return np.array([4, 4])


@fixture
def initial_segments(initial_centroid):
    import numpy as np
    import xarray as xr

    t = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    xy0 = np.array([np.cos(t), np.sin(t)])
    return xr.DataArray(
        np.array(
            [initial_centroid[:, None] + 2 * xy0, initial_centroid[:, None] + 3 * xy0]
        ),
        dims=("side", "coord", "point"),
        coords={"side": ["endocardium", "epicardium"], "coord": ["col", "row"]},
    )


@fixture
def mask_shape():
    return [5, 30, 30]


@fixture
def radial_mask(mask_shape) -> xr.DataArray:
    """DataArray defining 3 radial regions."""
    from skimage.draw import disk

    region, frame, row, col = shape = [3] + mask_shape
    data = np.full(shape, False)

    midr = row / 2
    midc = col / 2
    for r in range(region - 1, -1, -1):
        rr, cc = disk((midr, midc), row / (5 - r), shape=(row, col))
        data[r, :, rr, cc] = True
        if r < region - 1:
            data[r + 1, :, rr, cc] = False

    rr, cc = disk((midr, midc), row / 8, shape=(row, col))
    data[0, :, rr, cc] = False

    return xr.DataArray(
        data,
        dims=["region", "frame", "row", "col"],
        coords={
            "frame": np.arange(0, frame),
            "row": np.arange(0, row),
            "col": np.arange(0, col),
        },
    )


@fixture
def angular_mask(mask_shape) -> xr.DataArray:
    """DataArray defining 4 angular regions"""
    from skimage.draw import rectangle

    region, frame, row, col = shape = [4] + mask_shape
    data = np.full(shape, False)

    mid = int(row / 2)
    for i in range(2):
        for j in range(2):
            rr, cc = rectangle((i * mid, j * mid), extent=(mid, mid), shape=(row, col))
            data[int(i + 2 * j), :, rr, cc] = True

    return xr.DataArray(
        data,
        dims=["region", "frame", "row", "col"],
        coords={
            "frame": np.arange(0, frame),
            "row": np.arange(0, row),
            "col": np.arange(0, col),
        },
    )


@fixture
def expanded_radial(radial_mask):
    return (radial_mask * (radial_mask.region + 1)).sum("region").expand_dims(cine=[0])


@fixture
def expanded_angular(angular_mask):
    return (
        (angular_mask * (angular_mask.region + 1)).sum("region").expand_dims(cine=[0])
    )


@fixture
def reduced_radial(radial_mask, angular_mask):
    frame = angular_mask.sizes["frame"]

    return xr.DataArray(
        np.tile(
            np.arange(1, radial_mask.sizes["region"] + 1),
            (angular_mask.sizes["region"], frame, 1),
        ).transpose([1, 2, 0])[None, ...],
        dims=["cine", "frame", "radius", "angle"],
        coords={"cine": [0], "frame": np.arange(0, frame)},
    ).astype(float)


@fixture
def reduced_angular(radial_mask, angular_mask):
    frame = angular_mask.sizes["frame"]

    return xr.DataArray(
        np.tile(
            np.arange(1, angular_mask.sizes["region"] + 1),
            (radial_mask.sizes["region"], frame, 1),
        ).transpose([1, 0, 2])[None, ...],
        dims=["cine", "frame", "radius", "angle"],
        coords={"cine": [0], "frame": np.arange(0, frame)},
    ).astype(float)
