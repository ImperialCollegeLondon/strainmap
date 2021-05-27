import os
from itertools import chain
from pathlib import Path, PurePath, PurePosixPath
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import openpyxl as xlsx

import h5py
import sparse

from .sm_data import LabelledArray


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

    for l in labels:
        title = l.split(" - ")[0]
        try:
            add_markers(
                data.markers[dataset][l],
                params_ws,
                colnames=colnames,
                p=p,
                region_names=region_names,
                title=title,
            )
        except KeyError:
            print(f"Ignoring key '{l}' when exporting velocity markers.")

        add_velocity(data.velocities[dataset][l], region_names, wb.create_sheet(title))

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

    for l in labels:
        title = l.split(" - ")[0]
        try:
            add_strain_markers(
                data.strain_markers[dataset][l],
                params_ws,
                colnames=colnames,
                p=p,
                region_names=region_names,
                title=title,
            )
        except KeyError:
            print(f"Ignoring key '{l}' when exporting strain markers.")

        add_velocity(data.strain[dataset][l], region_names, wb.create_sheet(title))

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


def write_hdf5_file(data, filename: Union[h5py.File, str]):
    """Writes the contents of the StrainMap data object to a HDF5 file."""
    f = filename if isinstance(filename, h5py.File) else h5py.File(filename, "a")

    metadata_to_hdf5(f, data.metadata())

    for s in data.stored:
        if s in ("sign_reversal", "orientation"):
            if s in f:
                f[s][...] = getattr(data, s)
            else:
                f.create_dataset(s, data=getattr(data, s))
        elif s == "timeshift":
            f.attrs[s] = getattr(data, s)
        elif "files" in s:
            if getattr(data, s) is not None:
                paths_to_hdf5(f, f.filename, s, getattr(data, s).files)
        else:
            write_data_structure(f, s, getattr(data, s))


def metadata_to_hdf5(g, metadata):
    """"""
    for key, value in metadata.items():
        g.attrs[key] = value


def write_data_structure(g, name, structure):
    """Recursively populates the hdf5 file with a nested dictionary.

    If any dataset already exist, it gets updated with the new values, otherwise it
    is created.
    """
    group = g[name] if name in g else g.create_group(name, track_order=True)

    for n, struct in structure.items():
        if isinstance(struct, dict):
            write_data_structure(group, n, struct)
        else:
            if n in group:
                del group[n]
            group.create_dataset(n, data=struct, track_order=True)


def to_relative_paths(master: str, paths: List[str]) -> list:
    """Finds the relative paths of "paths" with respect to "master"."""
    import sys

    if sys.platform == "win32":
        return []

    try:
        filenames = [
            str(
                PurePosixPath(
                    Path(os.path.relpath(PurePath(p), PurePath(master).parent))
                )
            ).encode("ascii", "ignore")
            for p in paths
        ]
    except ValueError:
        filenames = []

    return filenames


def paths_to_hdf5(
    g: Union[h5py.File, h5py.Group], master: str, name: str, structure: dict
) -> None:
    """Saves a dictionary with paths as values after calculating the relative path."""
    group = g[name] if name in g else g.create_group(name, track_order=True)

    for n, struct in structure.items():
        if isinstance(struct, dict):
            paths_to_hdf5(group, master, n, struct)
        else:
            if n in group:
                del group[n]
            paths = to_relative_paths(master, struct)
            group.create_dataset(n, data=paths, track_order=True)


def labels_to_group(
    group: h5py.Group, dims: Sequence[str], coords: Dict[str, Sequence]
) -> None:
    """Save the dimensions and coordinates of a labelled array as a h5 group."""
    for d in dims:
        if coords[d] is not None:
            group.create_dataset(d, data=np.array(coords[d], dtype="S10"))
        else:
            group.create_dataset(d, dtype=h5py.Empty("f"))


def labelled_array_to_group(group: h5py.Group, larray: LabelledArray) -> None:
    """Save a LabelledArray as a h5 group."""
    labels_to_group(
        group.create_group("labels", track_order=True), larray.dims, larray.coords
    )
    if isinstance(larray.values, sparse.COO):
        group["values"] = larray.values.data
        group["coords"] = larray.values.coords
        group.attrs["shape"] = larray.values.shape
        group.attrs["fill_value"] = larray.values.fill_value
    else:
        group["values"] = larray.values


def terminal(msg: str, value: Optional[float] = None):
    if value is not None:
        msg = f"{msg}. Progress: {min(1., value) * 100}%"
    print(msg)
