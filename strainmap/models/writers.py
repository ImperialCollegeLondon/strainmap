import xlsxwriter as xlsx
import numpy as np


def velocity_to_xlsx(filename, data, dataset, vel_label):
    """Exports the chosen velocity to an Excel file.

    It includes 3 sheets:

    - Sheet 1 includes some metadata, like patient information, dataset, background
    subtraction method, etc.
    - Sheet 2 has the table with the markers information.
    - Sheet 3 has the 3 components of the velocity for each of the regions.
    """
    wb = xlsx.Workbook(filename)
    add_metadata(data, dataset, vel_label, wb.add_worksheet("Information"))
    add_markers(data.markers[dataset][vel_label], wb.add_worksheet("Parameters"))
    add_velocity(data.velocities[dataset][vel_label], wb.add_worksheet("Velocity"))
    wb.close()


def add_metadata(data, dataset, vel_label, ws):
    """Prepares the metadata of interests to be exported."""
    vel_type, background = vel_label.split(" - ")
    patient_data = data.read_dicom_file_tags(dataset, "MagZ", 0)
    metadata = {
        "Patient Name": str(patient_data.get("PatientName", "")),
        "Patient DOB": str(patient_data.get("PatientBirthDate", "")),
        "Date of Scan": str(patient_data.get("StudyDate", "")),
        "Dataset": dataset,
        "Velocity Type": vel_type,
        "Background Correction": background,
    }
    for i, key in enumerate(metadata):
        ws.write_row(i, 0, (key, "", metadata[key]))


def add_markers(markers, ws):
    """Adds the markers parameters to the sheet."""
    ws.merge_range(0, 2, 0, 4, "Longitudinal")
    ws.merge_range(0, 5, 0, 8, "Radial")
    ws.merge_range(0, 9, 0, 11, "Circumferential")

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

    ws.write_row(1, 0, colnames)
    ws.set_column(0, 0, 15)

    mask = np.ones(12, dtype=bool)
    mask[[3, 11]] = False
    m = np.array(markers).transpose((3, 0, 1, 2)).reshape((-1, 12))[:, mask]

    p = ("Frame", "Velocity (cm/s)", "Norm. Time (s)")
    reg = len(markers)
    for j in range(len(m)):
        data = m[j].astype(int) if j // reg == 0 else np.around(m[j], 3)
        row = [p[j // reg] if j % reg == 0 else "", j % reg + 1] + list(data)
        ws.write_row(j + 2, 0, row)


def add_velocity(velocity, ws):
    """Adds the velocities to the sheet."""
    regions = velocity.shape[0]
    header = ("Longitudinal", "Radial", "Circumferential")
    labels = ("z_Reg{}", "r_Reg{}", "theta_Reg{}")

    col = 0
    for i in range(3):
        ws.merge_range(0, i * (regions + 1), 0, (i + 1) * regions + i - 1, header[i])

        for r in range(regions):
            ws.write_column(
                1, col, [labels[i].format(r + 1)] + list(np.around(velocity[r, i], 3))
            )
            col += 1
        col += 1

    ws.set_column(0, col, 10)
