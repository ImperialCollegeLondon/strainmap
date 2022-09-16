from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import openpyxl as xlsx
import pandas as pd
import xarray as xr

from ..coordinates import Comp, Region
from .strainmap_data_model import StrainMapData


def velocity_to_xlsx(filename, data, cine) -> None:
    """Exports the velocity of the chosen cine to an Excel file.

    - Sheet 1 includes some metadata, like patient information, dataset, background
    subtraction method, etc. and the table with the markers information.
    - From sheet 2, includes all the velocities for the Global, Angular and Radial
    regions, one per sheet, with the 3 components of the velocity in each case.

    Args:
        filename: The name of the file to save the data. If it exist, it will be
            overwritten.
        data: A StrainMapData object with all the relevant information.
        cine: The cine of interest.
    """
    regions = [r.name for r in Region]
    region_names = ["AS", "A", "AL", "IL", "I", "IS"]
    if data.orientation == "CCW":
        region_names = region_names[::-1]

    # First we extract the metadata from the data object
    meta = data.metadata(cine)
    meta["Orientation"] = data.orientation
    meta["Timeshift (ms)"] = data.timeshift * 1000
    meta.update(
        {
            f"Sign reversal - {d}": v.item() < 0
            for d, v in zip(("X", "Y", "Z"), data.sign_reversal)
        }
    )

    row = len(meta) + 3

    markers = data.markers.assign_coords(
        quantity=["Frame", "Velocity (cm/s)", "Norm. Time (s)"]
    )
    with pd.ExcelWriter(filename) as writer:

        # The metadata is saved at the top of the first sheet
        pd.DataFrame(meta, index=[0]).T.to_excel(
            writer, sheet_name="Parameters", header=False
        )

        # Different regions need to be treated slightly differently to get the
        # formatting right, in addition to be saved in a different sheet for the case
        # of velocities
        for region in regions:
            num = Region[region].value
            mark = (
                markers.sel(cine=cine, region=region)
                .drop("cine")
                .stack(new_comp=["comp", "marker"])
                .dropna("new_comp", how="all")
            )
            vel = data.velocities.sel(cine=cine, region=region)

            if region == "GLOBAL":
                mark = mark.expand_dims(region=[region]).stack(
                    new_q=["quantity", "region"]
                )
            else:
                # We need to do some relabelling of the coordinates for the markers
                labels = (
                    [f"{region} - {n}" for n in range(num)]
                    if num != 6
                    else region_names
                )
                mark = mark.assign_coords(region=labels).stack(
                    new_q=["quantity", "region"]
                )
                vel = vel.stack(cols=["comp", "region"]).reset_index("region", True)

            # Now is when we actually save the markers and the velocities for the
            # current region
            mark.T.to_pandas().to_excel(writer, sheet_name="Parameters", startrow=row)
            vel.to_pandas().to_excel(writer, sheet_name=region)
            row += mark.sizes["new_q"] + 5


def export_superpixel(data, cine, filename):
    """Export superpixel velocity data.

    TODO: Remove in final version.
    """
    from .transformations import coordinates
    from .velocities import superpixel_velocity_curves

    svel = (
        superpixel_velocity_curves(
            data.cylindrical.sel(cine=cine).drop_sel(comp=Comp.LONG.name),
            data.masks.sel(cine=cine, region=Region.RADIAL_x3.name),
            data.masks.sel(cine=cine, region=Region.ANGULAR_x24.name),
        )
        .assign_coords(
            radius=["endo", "mid", "epi"], angle=range(Region.ANGULAR_x24.value)
        )
        .stack(loc=["radius", "angle"])
    )
    loc = (
        coordinates(
            data.centroid.sel(cine=cine),
            data.septum.sel(cine=cine),
            data.masks.sel(cine=cine, region=Region.RADIAL_x3.name),
            data.masks.sel(cine=cine, region=Region.ANGULAR_x24.name),
            data.data_files.pixel_size(cine),
        )
        .assign_coords(
            radius=["endo", "mid", "epi"], angle=range(Region.ANGULAR_x24.value)
        )
        .stack(loc=["radius", "angle"])
    )

    vel_sheet_names = ("Radial velocity", "Circumferential velocity")
    loc_sheet_names = ("Radial location", "Angular location")
    description = {
        "Radial velocity": "Radial velocity as a function of frame for each superpixel (cm/s).",  # noqa: E501
        "Circumferential velocity": "Circumferential velocity as a function of frame for each superpixel (cm/s).",  # noqa: E501
        "Radial location": "Radial location of each superpixel (cm).",
        "Angular location": "Angular location of each superpixel (radian).",
    }
    with pd.ExcelWriter(filename) as writer:
        for comp, name in zip(svel.comp, vel_sheet_names):
            pd.DataFrame({"Description": description[name]}, index=[0]).T.to_excel(
                writer, sheet_name=name, header=False
            )
            svel.sel(comp=comp).to_pandas().to_excel(
                writer, sheet_name=name, startrow=2
            )
        for comp, name in zip(loc.comp, loc_sheet_names):
            pd.DataFrame({"Description": description[name]}, index=[0]).T.to_excel(
                writer, sheet_name=name, header=False
            )
            loc.sel(comp=comp).to_pandas().to_excel(writer, sheet_name=name, startrow=2)


def rotation_to_xlsx(data, filename):
    """Exports rotation data to an excel file."""

    wb = xlsx.Workbook()

    ws = wb.active
    ws.title = "Mean angular velocity"
    ws.append(["Mean angular velocity (radian/s) as a function of frame and cine."])
    ws.append([""])
    ws.append(data.twist.coords["dataset"])
    ws.append(
        tuple((data.data_files.time_interval(d) for d in data.twist.coords["dataset"]))
    )
    ws.append(())
    for values in (
        data.twist.sel(item="angular_velocity").transpose("frame", "dataset").values
    ):
        ws.append(tuple(values))

    ws = wb.create_sheet("Mean radius")
    ws.append(["Mean radius of myocardium (cm) as a function of frame and cine."])
    ws.append([""])
    ws.append(data.twist.coords["dataset"])
    ws.append(())
    ws.append(())
    for values in data.twist.sel(item="radius").transpose("frame", "dataset").values:
        ws.append(tuple(values))

    wb.save(filename)
    wb.close()


def terminal(msg: str, value: Optional[float] = None):
    if value is not None:
        msg = f"{msg}. Progress: {min(1., value) * 100}%"
    print(msg)


def save_group(
    filename: Path, data: xr.DataArray, name: str, to_int=False, overwrite=False
) -> None:
    """Save a Datarray as a group in the netCDF file.

    If the group already exists, it is deleted first.

    Args:
        - filename: The filename of the file to save the data into.
        - data: Data to save.
        - name: Name of the group.
        - to_int: Flag to indicate if the data should be encoded as integer.
        - overwrite: Flag to indicate if the file content should be overwritten.

    Returns:
        None
    """
    mode = "w"
    if filename.is_file() and not overwrite:
        with h5py.File(filename, "a") as f:
            if name in f:
                del f[name]
        mode = "a"

    encoding = {}
    if to_int:
        scale = np.abs(data).max().item() / np.iinfo(np.int32).max
        encoding = {
            name: {
                "dtype": "int32",
                "scale_factor": scale,
                "_FillValue": np.iinfo(np.int32).min,
            }
        }

    data.to_netcdf(filename, mode=mode, group=name, encoding=encoding)


def save_attribute(filename: Path, **kwargs) -> None:
    """Save a scalar value as attribute in the netCDF file.

    Args:
        - filename: The filename of the file to save the data into.
        - kwargs: Data to be saved as attributes.

    Returns:
        None
    """
    f = h5py.File(filename, mode="a")
    for name, value in kwargs.items():
        if name in f.attrs:
            f.attrs.modify(name, value)
        else:
            f.attrs.create(name, value)
    f.close()


def write_netcdf_file(filename: Path, **kwargs) -> None:
    """Writes all kwargs to a netcdf file.

    Those arguments of type DataArray will be stored as groups. Any other will be stored
    as attributes.

    Args:
        - filename: The filename of the file to save the data into.
        - kwargs: The information to save in the file as keyword arguments.

    Return:
        None
    """
    if filename.suffix != ".nc":
        raise ValueError(
            f"'{filename.suffix}' is an invalid extension for a "
            "netCDF file. It must be '.nc'."
        )

    attr = dict()
    for name, value in kwargs.items():
        if isinstance(value, xr.DataArray):
            if value.shape != ():
                save_group(filename, value, name, value.dtype == float)
        else:
            attr[name] = value

    save_attribute(filename, **attr)


def export_for_training(destination: Path, data: StrainMapData) -> None:
    """Exports image and mask data to netCDF to train a new AI down the line.

    Args:
        destination (Path): Location where to save the data.
        data (StrainMapData): A StrainMapData object with segmentations.
    """
    from datetime import datetime as dt

    from tqdm import tqdm

    now = round(dt.now().timestamp())
    dataset_name = "stacked"
    encoding = {dataset_name: {"zlib": True, "complevel": 9}}
    for num, stacked in enumerate(
        tqdm(data.stack_masks(), total=data.segments.sizes["cine"])
    ):
        name = data.metadata()["Patient Name"].replace(" ", "")
        filename = destination / f"{name}_{num+1}_{now}_train.nc"
        stacked.drop_vars("cine").to_dataset(name=dataset_name).to_netcdf(
            filename, encoding=encoding
        )
