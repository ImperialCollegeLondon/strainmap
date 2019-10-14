from pytest import approx, mark
import sys


def test_add_metadata(strainmap_data):
    from strainmap.models.writers import add_metadata
    import openpyxl as xlsx

    dataset = list(strainmap_data.data_files.keys())[0]

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

    add_markers(markers[None, :, :, :], ws, title="Global")
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

    dataset_name = list(segmented_data.segments.keys())[0]
    filename = tmpdir / "strain_map_file.h5"
    f = h5py.File(filename, "a")

    write_data_structure(f, "segments", segmented_data.segments)
    assert "segments" in f
    assert dataset_name in f["segments"]
    assert "endocardium" in f["segments"][dataset_name]
    assert "epicardium" in f["segments"][dataset_name]
    assert segmented_data.segments[dataset_name]["endocardium"] == approx(
        f["segments"][dataset_name]["endocardium"][:]
    )

    segmented_data.segments[dataset_name]["endocardium"] *= 2
    write_data_structure(f, "segments", segmented_data.segments)
    assert segmented_data.segments[dataset_name]["endocardium"] == approx(
        f["segments"][dataset_name]["endocardium"][:]
    )


@mark.skipif(
    sys.platform.startswith("win"),
    reason="Relative paths across units fail under Windows.",
)
def test_write_hdf5_file(segmented_data, tmpdir):
    from strainmap.models.velocities import calculate_velocities
    from strainmap.models.writers import write_hdf5_file

    dataset_name = list(segmented_data.segments.keys())[0]
    calculate_velocities(
        segmented_data, dataset_name, global_velocity=True, angular_regions=[6]
    )

    filename = tmpdir / "strain_map_file.h5"
    write_hdf5_file(segmented_data, filename)


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
    expected = [b"..", b".", b"cars", str(Path("cars/Tesla")).encode("ascii", "ignore")]
    actual = to_relative_paths(master, paths)

    assert actual == expected


@mark.skipif(
    sys.platform.startswith("win"),
    reason="Relative paths across units fail under Windows.",
)
def test_paths_to_hdf5(strainmap_data, tmpdir):
    from strainmap.models.writers import paths_to_hdf5, to_relative_paths
    import h5py

    dataset_name = list(strainmap_data.data_files.keys())[0]
    filename = tmpdir / "strain_map_file.h5"

    abs_paths = strainmap_data.data_files[dataset_name]["MagX"]
    rel_paths = to_relative_paths(filename, abs_paths)

    f = h5py.File(filename, "a")
    paths_to_hdf5(f, filename, "data_files", strainmap_data.data_files)

    assert "data_files" in f
    assert dataset_name in f["data_files"]
    assert "MagX" in f["data_files"][dataset_name]
    assert rel_paths == f["data_files"][dataset_name]["MagX"][...].tolist()

    strainmap_data.data_files[dataset_name]["MagX"][0] = "my new path"
    abs_paths = strainmap_data.data_files[dataset_name]["MagX"]
    rel_paths = to_relative_paths(filename, abs_paths)
    paths_to_hdf5(f, filename, "data_files", strainmap_data.data_files)

    assert b"my new path" in rel_paths[0]
    assert rel_paths == f["data_files"][dataset_name]["MagX"][...].tolist()
