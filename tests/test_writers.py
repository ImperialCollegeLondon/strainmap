import sys

from pytest import approx, mark


def test_add_metadata(strainmap_data):
    from strainmap.models.writers import add_metadata
    import openpyxl as xlsx

    dataset = strainmap_data.data_files.datasets[0]

    wb = xlsx.Workbook()
    ws = wb.create_sheet("Parameters")
    assert "Parameters" in wb.sheetnames

    add_metadata(strainmap_data.metadata(dataset), ws)
    assert ws.max_row == 8


@mark.xfail(reason="Refactoring in progress")
def test_add_markers(markers):
    from strainmap.models.writers import add_markers
    import numpy as np
    import openpyxl as xlsx

    wb = xlsx.Workbook()
    ws = wb.create_sheet("Parameters")

    colnames = (
        "Parameter",
        "Region",
        "P",
        "S",
        "PSS",
        "P",
        "S",
        "PSS",
        "ES",
        "P",
        "S",
        "PSS",
    )

    p = ("Frame", "Strain (%)", "Time (s)")
    region_names = "AS", "A", "AL", "IL", "I", "IS"
    add_markers(
        markers[None, :, :, :],
        ws,
        colnames=colnames,
        p=p,
        region_names=region_names,
        title="Global",
    )
    m = markers[None, :, :, :].transpose((3, 0, 1, 2)).reshape((-1, 12))
    expected = 5 + len(m)
    assert ws.max_row == expected
    assert ws["A5"].value == "Parameter"
    assert ws["L8"].value == approx(np.around(markers[2, 2, 2], 3))


@mark.xfail(reason="Refactoring in progress")
def test_add_velocity(velocities):
    from strainmap.models.writers import add_velocity
    import numpy as np
    import openpyxl as xlsx

    wb = xlsx.Workbook()
    ws = wb.create_sheet("Velocity")
    region_names = "AS", "A", "AL", "IL", "I", "IS"

    add_velocity(velocities[None, :, :], region_names, ws)
    assert ws.max_row == 52
    assert ws["A52"].value == approx(np.around(velocities[0, -1], 3))
    assert ws["C52"].value == approx(np.around(velocities[1, -1], 3))
    assert ws["E52"].value == approx(np.around(velocities[2, -1], 3))


def test_metadata_to_hdf5(strainmap_data, tmpdir):
    from strainmap.models.writers import metadata_to_hdf5
    import h5py

    filename = tmpdir / "strain_map_file.h5"
    f = h5py.File(filename, "a")

    metadata_to_hdf5(f, strainmap_data.metadata())

    assert "Patient Name" in f.attrs
    assert "Patient DOB" in f.attrs
    assert "Date of Scan" in f.attrs


def test_write_data_structure(segmented_data, tmpdir):
    from strainmap.models.writers import write_data_structure
    import h5py

    cine = segmented_data.segments.cine[0].item()
    filename = tmpdir / "strain_map_file.h5"
    f = h5py.File(filename, "a")

    write_data_structure(f, "segments", segmented_data.segments)
    assert "segments" in f
    assert cine in f["segments"]
    assert "endocardium" in f["segments"][cine]
    assert "epicardium" in f["segments"][cine]
    assert segmented_data.segments.sel(cine=cine, side="endocardium").values == approx(
        f["segments"][cine]["endocardium"][:]
    )

    segmented_data.segments.loc[{"cine": cine, "side": "endocardium"}] *= 2
    write_data_structure(f, "segments", segmented_data.segments)
    assert segmented_data.segments.sel(cine=cine, side="endocardium").values == approx(
        f["segments"][cine]["endocardium"][:]
    )


def test_write_hdf5_file(segmented_data, tmpdir):
    from strainmap.models.writers import write_hdf5_file

    filename = tmpdir / "strain_map_file.h5"
    write_hdf5_file(segmented_data, filename)


@mark.skipif(sys.platform == "win32", reason="does not run on windows in Azure")
def test_to_relative_paths():
    from strainmap.models.writers import to_relative_paths
    from pathlib import Path

    master = Path("home/data/my_file.h5").resolve()
    paths = [
        str(Path("home").resolve()),
        str(Path("home/data").resolve()),
        str(Path("home/data/cars").resolve()),
        str(Path("home/data/cars/Tesla").resolve()),
    ]
    expected = [b"..", b".", b"cars", b"cars/Tesla"]
    actual = to_relative_paths(master, paths)
    assert actual == expected


@mark.skipif(sys.platform == "win32", reason="does not run on windows in Azure")
def test_paths_to_hdf5(strainmap_data, tmpdir):
    from strainmap.models.writers import paths_to_hdf5, to_relative_paths
    from strainmap.coordinates import Comp
    import h5py

    cine = strainmap_data.data_files.datasets[0]
    filename = tmpdir / "strain_map_file.h5"

    mag = strainmap_data.data_files.vars[Comp.MAG]
    abs_paths = strainmap_data.data_files.files.sel(cine=cine, raw_comp=mag).values
    rel_paths = to_relative_paths(filename, abs_paths)

    f = h5py.File(filename, "a")
    paths_to_hdf5(f, filename, strainmap_data.data_files.files)

    assert "data_files" in f
    assert cine in f["data_files"]
    assert mag in f["data_files"][cine]
    assert rel_paths == f["data_files"][cine][mag][...].tolist()

    strainmap_data.data_files.files.loc[
        {"cine": cine, "raw_comp": mag, "frame": 0}
    ] = "/my new path"
    abs_paths = strainmap_data.data_files.files.sel(cine=cine, raw_comp=mag).values
    rel_paths = to_relative_paths(filename, abs_paths)
    paths_to_hdf5(f, filename, strainmap_data.data_files.files)

    assert rel_paths == f["data_files"][cine][mag][...].tolist()
