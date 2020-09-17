from pytest import approx, mark
import sys


def test_add_metadata(strainmap_data):
    from strainmap.models.writers import add_metadata
    import openpyxl as xlsx

    dataset = strainmap_data.data_files.datasets[0]

    wb = xlsx.Workbook()
    ws = wb.create_sheet("Parameters")
    assert "Parameters" in wb.sheetnames

    add_metadata(strainmap_data.metadata(dataset), "Phantom", ws)
    assert ws.max_row == 5
    assert ws["C5"].value == "Phantom"


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

    add_markers(markers[None, :, :, :], ws, colnames=colnames, p=p, title="Global")
    m = markers[None, :, :, :].transpose((3, 0, 1, 2)).reshape((-1, 12))
    expected = 5 + len(m)
    assert ws.max_row == expected
    assert ws["A5"].value == "Parameter"
    assert ws["L8"].value == approx(np.around(markers[2, 2, 2], 3))


def test_add_velocity(velocity):
    from strainmap.models.writers import add_velocity
    import numpy as np
    import openpyxl as xlsx

    wb = xlsx.Workbook()
    ws = wb.create_sheet("Velocity")

    add_velocity(velocity[None, :, :], ws)
    assert ws.max_row == 52
    assert ws["A52"].value == approx(np.around(velocity[0, -1], 3))
    assert ws["C52"].value == approx(np.around(velocity[1, -1], 3))
    assert ws["E52"].value == approx(np.around(velocity[2, -1], 3))


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
    import h5py

    cine = strainmap_data.data_files.datasets[0]
    filename = tmpdir / "strain_map_file.h5"

    mag = strainmap_data.data_files.vars["mag"]
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


def test_labels_to_group(larray, tmpdir):
    from strainmap.models.writers import labels_to_group
    import numpy as np
    import h5py

    filename = tmpdir / "strainmap_file.h5"
    f = h5py.File(filename, "a")
    labels_to_group(
        f.create_group("labels", track_order=True), larray.dims, larray.coords
    )

    assert tuple(f["labels"].keys()) == larray.dims
    for d in larray.dims:
        if larray.coords[d] is None:
            assert f["labels"][d].shape is None
        else:
            assert list(f["labels"][d][...].astype(np.str)) == larray.coords[d]


def test_labelled_array_to_group(larray, tmpdir):
    from strainmap.models.writers import labelled_array_to_group
    import numpy as np
    import sparse
    import h5py

    filename = tmpdir / "strainmap_file.h5"
    f = h5py.File(filename, "a")
    labelled_array_to_group(f.create_group("array"), larray)

    assert tuple(f["array"]["labels"].keys()) == larray.dims
    for d in larray.dims:
        if larray.coords[d] is None:
            assert f["array"]["labels"][d].shape is None
        else:
            assert list(f["array"]["labels"][d][...].astype(np.str)) == larray.coords[d]

    if isinstance(larray.values, sparse.COO):
        assert f["array"]["values"][...] == approx(larray.values.data)
        assert f["array"]["coords"][...] == approx(larray.values.coords)
        assert f["array"].attrs["shape"] == approx(larray.values.shape)
        assert f["array"].attrs["fill_value"] == larray.values.fill_value
    else:
        assert f["array"]["values"][...] == approx(larray.values)
