from itertools import chain
from pathlib import Path
from typing import Optional
import subprocess

import h5py
import numpy as np
import openpyxl as xlsx
import xarray as xr


def velocity_to_xlsx(filename, data, dataset, vel_label):
    """Exports the chosen velocity to an Excel file.

    It includes 2 or more sheets:

    - Sheet 1 includes some metadata, like patient information, dataset, background
    subtraction method, etc. and the table with the markers information.
    - From sheet 2, includes all the velocities calculated with the same background
        subtraciton method, one per sheet, with the 3 components of the velocity for
        each of the regions.
    """
    wb = xlsx.Workbook()
    if len(vel_label.split(" - ")) == 2:
        background = vel_label.split(" - ")[-1]
        labels = [label for label in data.velocities[dataset] if background in label]
    else:
        labels = list(data.velocities[dataset].keys())

    params_ws = wb.active
    params_ws.title = "Parameters"
    add_metadata(data.metadata(dataset), params_ws)
    params_ws.append(("Orientation", "", data.orientation))
    add_sign_reversal(data.sign_reversal, params_ws)
    params_ws.append(("Timeshift (ms)", "", data.timeshift * 1000))

    colnames = (
        "Parameter",
        "Region",
        "PS",
        "PD",
        "PAS",
        "PS",
        "PD",
        "PAS",
        "ES",
        "PC1",
        "PC2",
        "PC3",
    )

    p = ("Frame", "Velocity (cm/s)", "Norm. Time (s)")
    region_names = "AS", "A", "AL", "IL", "I", "IS"
    if data.orientation == "CCW":
        region_names = region_names[::-1]

    for ll in labels:
        title = ll.split(" - ")[0]
        try:
            add_markers(
                data.markers[dataset][ll],
                params_ws,
                colnames=colnames,
                p=p,
                region_names=region_names,
                title=title,
            )
        except KeyError:
            print(f"Ignoring key '{ll}' when exporting velocity markers.")

        add_velocity(data.velocities[dataset][ll], region_names, wb.create_sheet(title))

    wb.save(filename)
    wb.close()


def strain_to_xlsx(filename, data, dataset, vel_label):
    """Exports the chosen velocity to an Excel file.

    It includes 2 or more sheets:

    - Sheet 1 includes some metadata, like patient information, dataset, background
    subtraction method, etc. and the table with the markers information.
    - From sheet 2, includes all the velocities calculated with the same background
        subtraciton method, one per sheet, with the 3 components of the velocity for
        each of the regions.
    """
    wb = xlsx.Workbook()
    if len(vel_label.split(" - ")) == 2:
        background = vel_label.split(" - ")[-1]
        labels = [
            label
            for label in data.strain[dataset]
            if background in label and "cylindrical" not in label
        ]
    else:
        labels = [
            label for label in data.velocities[dataset] if "cylindrical" not in label
        ]

    params_ws = wb.active
    params_ws.title = "Parameters"
    add_metadata(data.metadata(dataset), params_ws)
    params_ws.append(("Orientation", "", data.orientation))
    add_sign_reversal(data.sign_reversal, params_ws)
    params_ws.append(("Timeshift (ms)", "", data.timeshift * 1000))

    colnames = (
        "Parameter",
        "Region",
        "PS",
        "ES",
        "P",
        "PS",
        "ES",
        "P",
        "PS",
        "ES",
        "P",
    )

    p = ("Frame", "Strain (%)", "Time (s)")
    region_names = "AS", "A", "AL", "IL", "I", "IS"
    if data.orientation == "CCW":
        region_names = region_names[::-1]

    for ll in labels:
        title = ll.split(" - ")[0]
        try:
            add_strain_markers(
                data.strain_markers[dataset][ll],
                params_ws,
                colnames=colnames,
                p=p,
                region_names=region_names,
                title=title,
            )
        except KeyError:
            print(f"Ignoring key '{ll}' when exporting strain markers.")

        add_velocity(data.strain[dataset][ll], region_names, wb.create_sheet(title))

    wb.save(filename)
    wb.close()


def add_metadata(metadata, ws):
    """Prepares the metadata of interest to be exported."""
    ws.column_dimensions["A"].width = 15
    for key, value in metadata.items():
        ws.append((key, "", value))


def add_sign_reversal(sign_reversal, ws):
    """Adds the sign reversal information."""
    ws.append(("Sign reversal",))
    for i, key in enumerate(("X", "Y", "Z")):
        ws.append(("", key, str(sign_reversal[i])))


def add_markers(markers, ws, colnames, p, region_names, title=None):
    """Adds the markers parameters to the sheet."""
    row = ws.max_row + 2
    ws.merge_cells(start_row=row, start_column=3, end_row=row, end_column=12)
    ws.cell(column=3, row=row, value=title)

    row += 1
    ws.merge_cells(start_row=row, start_column=3, end_row=row, end_column=5)
    ws.merge_cells(start_row=row, start_column=6, end_row=row, end_column=9)
    ws.merge_cells(start_row=row, start_column=10, end_row=row, end_column=12)
    ws.cell(column=3, row=row, value="Longitudinal")
    ws.cell(column=6, row=row, value="Radial")
    ws.cell(column=10, row=row, value="Circumferential")

    ws.append(colnames)

    mask = np.ones(12, dtype=bool)
    mask[[3, 11]] = False
    m = np.array(markers).transpose((3, 0, 1, 2)).reshape((-1, 12))[:, mask]

    reg = len(markers)
    divider = int(np.ceil(reg / 6))

    for j in range(len(m)):
        rg = region_names[(j % reg) // divider] if reg > 1 else "Global"
        data = m[j].astype(int) if j // reg == 0 else np.around(m[j], 3)
        row_data = [p[j // reg] if j % reg == 0 else "", rg] + list(data)
        ws.append(row_data)


def add_strain_markers(markers, ws, colnames, p, region_names, title=None):
    """Adds the markers parameters to the sheet."""
    row = ws.max_row + 2
    ws.merge_cells(start_row=row, start_column=3, end_row=row, end_column=9)
    ws.cell(column=3, row=row, value=title)

    row += 1
    ws.merge_cells(start_row=row, start_column=3, end_row=row, end_column=5)
    ws.merge_cells(start_row=row, start_column=6, end_row=row, end_column=8)
    ws.merge_cells(start_row=row, start_column=9, end_row=row, end_column=11)
    ws.cell(column=3, row=row, value="Longitudinal")
    ws.cell(column=6, row=row, value="Radial")
    ws.cell(column=9, row=row, value="Circumferential")

    ws.append(colnames)

    m = np.array(markers).transpose((3, 0, 1, 2)).reshape((-1, 9))

    reg = len(markers)
    divider = int(np.ceil(reg / 6))

    for j in range(len(m)):
        rg = region_names[(j % reg) // divider] if reg > 1 else "Global"
        data = m[j].astype(int) if j // reg == 0 else np.around(m[j], 3)
        row_data = [p[j // reg] if j % reg == 0 else "", rg] + list(data)
        ws.append(row_data)


def add_velocity(velocity, region_names, ws):
    """Adds the velocities to the sheet."""
    reg = velocity.shape[0]
    frames = velocity.shape[-1]
    headers = ("Longitudinal (cm/s)", "Radial (cm/s)", "Circumferential (cm/s)")
    for i, header in enumerate(headers):
        ws.cell(column=i * reg + 1, row=1, value=header)

    rg = region_names
    if reg == 1:
        rg = ["Global"]
    elif reg == 3:
        rg = list(range(1, reg + 1))
    elif reg == 24:
        rg = list(chain.from_iterable(([r] * (reg // len(rg)) for r in rg)))

    labels = ("z_{}", "r_{}", "theta_{}")
    ws.append([label.format(r) for label in labels for r in rg])

    for f in range(frames):
        ws.append([velocity[r, i, f] for i in range(len(labels)) for r in range(reg)])

    row = 1
    for i in range(len(headers)):
        ws.insert_cols((i + 1) * (reg + 1))
        if reg > 1:
            ws.merge_cells(
                start_row=1, start_column=row, end_row=1, end_column=row + reg - 1
            )
            row = row + reg + 1


def export_superpixel(data, dataset, filename):
    """Export superpixel velocity data.

    TODO: Remove in final version.
    """
    from .velocities import px_velocity_curves
    from .strain import coordinates

    vels = px_velocity_curves(data, dataset)
    loc = np.take(
        coordinates(data, (dataset,), resample=False, use_frame_zero=False), 0, axis=2
    )

    rad = ["endo", "mid", "epi"]
    label = "{} {}"

    def add_superpixel_data(values, ws, description):
        ws.append([description])
        ws.append([""])
        ws.append(
            [
                label.format(rad[r], a + 1)
                for r in range(values.shape[1])
                for a in range(values.shape[2])
            ]
        )
        for f in range(values.shape[0]):
            ws.append(
                [
                    values[f, r, a]
                    for r in range(values.shape[1])
                    for a in range(values.shape[2])
                ]
            )
        for i in range(values.shape[1]):
            ws.insert_cols((i + 1) * (values.shape[2] + 1))

    wb = xlsx.Workbook()
    ws = wb.active
    ws.title = "Radial velocity"
    msg = "Radial velocity as a function of frame for each superpixel (cm/s)."
    add_superpixel_data(vels[1], ws, msg)
    msg = "Circumferential velocity as a function of frame for each superpixel (cm/s)."
    add_superpixel_data(vels[2], wb.create_sheet("Circumferential velocity"), msg)
    msg = "Radial location of each superpixel (cm)."
    add_superpixel_data(loc[1], wb.create_sheet("Radial location"), msg)
    msg = "Angular location of each superpixel (radian)."
    add_superpixel_data(loc[2], wb.create_sheet("Angular location"), msg)

    wb.save(filename)
    wb.close()


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

    If the gorup already exists, it is deleted first.

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
        f = h5py.File(filename, "a")
        if name in f:
            del f[name]
        f.close()
        mode = "a"

    encoding = {}
    if to_int:
        scale = np.abs(data).max().item() / np.iinfo(np.int16).max
        encoding = {
            name: {
                "dtype": "int16",
                "scale_factor": scale,
                "_FillValue": np.iinfo(np.int16).min,
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
        if value != f.attrs.get(name, None):
            f.attrs[name] = value
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


def repack_file(
    filename: Path, target: Optional[Path] = None, overwrite: bool = False
) -> None:
    """Repacks a nc or h5 file to remove unused space.

    The HDF5 file space management is peculiar and files might look enormous even though
    there are just a few things inside. To recover that missing space, the file needs to
    be re-packed:

    https://support.hdfgroup.org/HDF5/doc/Advanced/FileSpaceManagement/FileSpaceManagement.pdf

    Args:
        - filename: The filename of the file repack.
        - target: Name of the repacked file. If None, it is set to the name of the file
            preceded by ~ or, if it was already like that, removes the tilde.
        - overwrite: If target exists and overwrite is true, it is overwritten.

    Returns:
        None
    """
    if filename.suffix not in (".nc", ".h5"):
        raise ValueError("Only '.nc' and '.h5' files can be repacked.")

    if target is None:
        if "~" in filename.name:
            name = filename.name.strip("~")
        else:
            name = f"~{filename.name}"
        target = filename.parent / name

    if target.is_file() and not overwrite:
        raise RuntimeError(
            f"Target file for repack {target}. Already exists."
            "Use 'overwrite=True' to overwrite."
        )

    subprocess.run(["h5repack", "--enable-error-stack", str(filename), str(target)])
