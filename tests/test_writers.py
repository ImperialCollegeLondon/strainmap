from pytest import approx


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


def test_to_hdf5(segmented_data, tmpdir):
    from strainmap.models.velocities import calculate_velocities
    from strainmap.models.writers import to_hdf5

    dataset_name = list(segmented_data.segments.keys())[0]
    calculate_velocities(
        segmented_data, dataset_name, global_velocity=True, angular_regions=[6]
    )

    filename = tmpdir / "strain_map_file.h5"
    to_hdf5(segmented_data, filename)
