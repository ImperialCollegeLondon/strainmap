import openpyxl as xlsx
import numpy as np


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
    background = vel_label.split(" - ")[-1]
    labels = [label for label in data.velocities[dataset] if background in label]

    params_ws = wb.active
    params_ws.title = "Parameters"
    add_metadata(data, dataset, background, params_ws)

    for l in labels:
        title = l.split(" - ")[0]
        add_markers(data.markers[dataset][l], params_ws, title=title)
        add_velocity(data.velocities[dataset][l], wb.create_sheet(title))

    wb.save(filename)
    wb.close()


def add_metadata(data, dataset, background, ws):
    """Prepares the metadata of interest to be exported."""
    patient_data = data.read_dicom_file_tags(dataset, "MagZ", 0)
    metadata = {
        "Patient Name": str(patient_data.get("PatientName", "")),
        "Patient DOB": str(patient_data.get("PatientBirthDate", "")),
        "Date of Scan": str(patient_data.get("StudyDate", "")),
        "Dataset": dataset,
        "Background Correction": background,
    }

    ws.column_dimensions["A"].width = 15
    for i, key in enumerate(metadata):
        ws.append((key, "", metadata[key]))


def add_markers(markers, ws, title=None):
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

    ws.append(colnames)

    mask = np.ones(12, dtype=bool)
    mask[[3, 11]] = False
    m = np.array(markers).transpose((3, 0, 1, 2)).reshape((-1, 12))[:, mask]

    p = ("Frame", "Velocity (cm/s)", "Norm. Time (s)")
    reg = len(markers)

    for j in range(len(m)):
        data = m[j].astype(int) if j // reg == 0 else np.around(m[j], 3)
        row_data = [p[j // reg] if j % reg == 0 else "", j % reg + 1] + list(data)
        ws.append(row_data)


def add_velocity(velocity, ws):
    """Adds the velocities to the sheet."""
    reg = velocity.shape[0]
    frames = velocity.shape[-1]
    headers = ("Longitudinal", "Radial", "Circumferential")
    for i, header in enumerate(headers):
        ws.cell(column=i * reg + 1, row=1, value=header)

    labels = ("z_Reg{}", "r_Reg{}", "theta_Reg{}")
    ws.append([label.format(r + 1) for label in labels for r in range(reg)])

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
