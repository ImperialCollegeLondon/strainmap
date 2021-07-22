import os
import tkinter as tk
import tkinter.filedialog
from datetime import datetime
from functools import partial
from pathlib import Path
from tkinter import messagebox, ttk
from traceback import print_exc
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .base_window_and_task import TaskViewBase, register_view
from .figure_actions import BrightnessAndContrast, ScrollFrames, ZoomAndPan
from .figure_actions_manager import FigureActionsManager


@register_view
class DataTaskView(TaskViewBase):
    def __init__(self, root, controller):

        super().__init__(
            root, controller, button_text="Data", button_image="save.gif", button_row=0
        )
        self.rowconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)

        self.data_folder = tk.StringVar(value="")
        self.output_file = tk.StringVar(value="")
        self.datafolder_entry = None
        self.outputfile_entry = None
        self.current_dir = os.path.expanduser("~")
        self.control = None
        self.dataselector = None
        self.patient_info = None

        self.visualise = None
        self.notebook = None
        self.treeview = None
        self.fig = None
        self.datasets_var = tk.StringVar(value="")
        self.maps_var = tk.StringVar(value="MAG")
        self.anim = False

        self.create_controls()

    def create_controls(self) -> None:
        """Creates the controls to load and save data."""
        self.control = ttk.Frame(master=self, width=300, name="control")
        self.control.grid(column=0, row=0, sticky=tk.NSEW, pady=5)
        self.control.rowconfigure(50, weight=1)
        self.control.grid_propagate(flag=False)
        self.control.columnconfigure(1, weight=1)

        self.visualise = ttk.Frame(master=self, name="visualise")
        self.visualise.grid(column=1, row=0, sticky=tk.NSEW, padx=5, pady=5)
        self.visualise.columnconfigure(0, weight=1)
        self.visualise.rowconfigure(0, weight=1)
        self.visualise.grid_propagate(flag=False)

        ttk.Button(
            master=self.control,
            name="chooseDataFolder",
            text="New analysis",
            command=self.load_data,
        ).grid(sticky=tk.NSEW, columnspan=2)

        ttk.Button(
            master=self.control,
            name="openStrainMapFile",
            text="Review analysis",
            command=self.open_existing_file,
            width=25,
        ).grid(sticky=tk.NSEW, columnspan=2, pady=5)

        ttk.Button(
            master=self.control,
            name="chooseOutputFile",
            text="Save analysis as...",
            command=self.select_output_file,
            state="disabled",
        ).grid(sticky=tk.NSEW, columnspan=2)

        ttk.Label(master=self.control, text="Data folder: ").grid(
            row=4, sticky=tk.W, pady=5
        )

        self.datafolder_entry = ttk.Entry(
            master=self.control, textvariable=self.data_folder, state="readonly"
        )
        self.datafolder_entry.grid(
            row=5, column=0, columnspan=2, sticky=tk.NSEW, pady=5
        )
        ttk.Label(master=self.control, text="Output file: ").grid(row=6, sticky=tk.W)

        self.outputfile_entry = ttk.Entry(
            master=self.control, textvariable=self.output_file, state="readonly"
        )
        self.outputfile_entry.grid(row=7, column=0, columnspan=2, sticky=tk.NSEW)

        ttk.Button(
            master=self.control,
            name="clearAllData",
            text="Clear all data",
            command=self.clear_data,
        ).grid(row=99, columnspan=2, sticky=tk.NSEW)

    def create_data_selector(self):
        """Creates the selector for the data."""
        values, texts, var_values, patient_data = self.get_data_information()
        self.datasets_var.set(values[0])

        patient_name = patient_data.get("Patient Name", "")
        patient_dob = patient_data.get("Patient DOB", "")
        patient_study_date = patient_data.get("Date of Scan", "")

        self.dataselector = ttk.Frame(self.control, name="dataSelector")
        self.dataselector.grid(row=40, columnspan=2, sticky=tk.NSEW, pady=5)
        self.dataselector.columnconfigure(0, weight=1)

        patient_frame = ttk.Labelframe(self.dataselector, text="Patient Information:")
        patient_frame.grid(column=0, sticky=tk.NSEW, padx=5, pady=5)
        patient_frame.columnconfigure(1, weight=1)

        ttk.Label(master=patient_frame, text="Name:").grid(sticky=tk.W, padx=10)
        ttk.Label(master=patient_frame, text="DoB:").grid(sticky=tk.W, padx=10)
        ttk.Label(master=patient_frame, text="Date of scan:").grid(sticky=tk.W, padx=10)

        ttk.Label(master=patient_frame, text=patient_name).grid(
            column=1, row=0, sticky=tk.W
        )
        ttk.Label(master=patient_frame, text=patient_dob).grid(
            column=1, row=1, sticky=tk.W
        )
        ttk.Label(master=patient_frame, text=patient_study_date).grid(
            column=1, row=2, sticky=tk.W
        )

        dataset_frame = ttk.Labelframe(self.dataselector, text="Datasets:")
        dataset_frame.grid(column=0, sticky=tk.NSEW, padx=5, pady=5)
        dataset_frame.columnconfigure(0, weight=1)

        datasets_box = ttk.Combobox(
            master=dataset_frame,
            textvariable=self.datasets_var,
            values=values,
            state="readonly" if len(patient_data) > 0 else "disabled",
        )
        datasets_box.bind("<<ComboboxSelected>>", self.update_visualization)
        datasets_box.grid(column=0, sticky=tk.NSEW, padx=5, pady=5)

        for t, v in zip(texts, var_values):
            ttk.Radiobutton(
                master=dataset_frame,
                variable=self.maps_var,
                text=t,
                value=v,
                command=self.update_visualization,
            ).grid(sticky=tk.NSEW, padx=5, pady=5)

    def create_data_viewer(self):
        """Creates the viewer for the data, including the animation and the DICOM."""
        self.notebook = ttk.Notebook(self.visualise, name="notebook")
        self.notebook.grid(sticky=tk.NSEW)
        self.notebook.columnconfigure(0, weight=1)
        self.notebook.rowconfigure(0, weight=1)

        self.notebook.add(self.create_animation_viewer(), text="Animation")
        data_frame, self.treeview = self.create_dicom_viewer()
        self.notebook.add(data_frame, text="DATA dicom")

    def create_animation_viewer(self):
        """Creates the animation plot area."""
        self.fig = Figure()
        ax = self.fig.add_subplot()
        ax.set_position((0.05, 0.05, 0.9, 0.9))

        animation_frame = ttk.Frame(self.notebook)
        animation_frame.columnconfigure(0, weight=1)
        animation_frame.rowconfigure(0, weight=1)

        canvas = FigureCanvasTkAgg(self.fig, master=animation_frame)
        canvas.draw()
        canvas.get_tk_widget().grid(sticky=tk.NSEW, padx=5, pady=5)

        self.fig.actions_manager = FigureActionsManager(
            self.fig, ZoomAndPan, BrightnessAndContrast, ScrollFrames
        )

        return animation_frame

    def create_dicom_viewer(self):

        frame = ttk.Frame(self.notebook)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        treeview = ttk.Treeview(frame, selectmode="browse")
        vsb = ttk.Scrollbar(frame, orient="vertical", command=treeview.yview)
        treeview.configure(yscrollcommand=vsb.set)
        treeview.grid(column=0, row=0, sticky=tk.NSEW, padx=5, pady=5)
        vsb.grid(column=1, row=0, sticky=tk.NSEW)

        treeview["columns"] = ("1", "2")
        treeview["show"] = "headings"
        treeview.column("1", width=300, stretch=False)
        treeview.heading("1", text="Tags")
        treeview.heading("2", text="Values")

        return frame, treeview

    def load_data(self, refresh: bool = True):
        """Loads new data into StrainMap"""
        path = tk.filedialog.askdirectory(
            title="Select DATA directory", initialdir=self.current_dir
        )

        if path == "":
            return

        self.controller.load_data_from_folder(data_files=path)
        if self.data is not None:
            self.controller.review_mode = False
            self.current_dir = path
            self.data_folder.set(self.current_dir)
            self.output_file.set(None)
            self.nametowidget("control.chooseOutputFile")["state"] = "enable"
            msg = f"Data loaded from {path}."
        else:
            msg = "No data was loaded."

        self.controller.window.progress(msg)
        if refresh:
            self.update_widgets()

    def open_existing_file(self):
        """Opens an existing StrainMap file."""
        path = tk.filedialog.askopenfilename(
            title="Select existing StrainMap file",
            initialdir=self.current_dir,
            filetypes=(("StrainMap files", "*.nc"), ("Legacy StrainMap files", "*.h5")),
        )

        if path == "":
            return

        self.load_data(refresh=False)

        if self.data is None:
            return

        try:
            self.controller.load_data_from_file(strainmap_file=path)
            self.controller.review_mode = False
            self.output_file.set(self.data.filename)
            self.current_dir = str(Path(path).parent)
            self.nametowidget("control.chooseOutputFile")["state"] = "enable"
            msg = f"Data loaded from {path}."

        except Exception as err:
            log = (
                Path.home()
                / f"StrainMap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            )
            with log.open("a") as f:
                print_exc(file=f)
            self.controller.clear_data()
            msg = f"{err} - Log info in {str(log)}"

        self.controller.window.progress(msg)
        self.update_widgets()

    def select_output_file(self):
        """Selects an output file in which to store the current data."""
        meta = self.data.metadata()
        name, date = [meta[key] for key in ["Patient Name", "Date of Scan"]]
        init = f"{name}_{date}.nc"

        path = tk.filedialog.asksaveasfilename(
            title="Introduce new StrainMap filename.",
            initialfile=init,
            initialdir=self.current_dir,
            filetypes=(("StrainMap files", "*.nc"),),
            defaultextension="nc",
        )

        if path == "":
            return
        elif self.controller.add_file(strainmap_file=path):
            self.output_file.set(path)

    def clear_data(self):
        """Clears all data from memory."""
        clear = messagebox.askokcancel(
            "Warning!",
            "This will erase all data from memory\nDo you want to continue?",
            icon="warning",
        )
        if clear:
            self.controller.clear_data()
            self.clear_widgets()

    def get_data_information(self):
        """Gets some information related to the available datasets, frames, etc."""
        values = ["No suitable datasets available."]
        texts = []
        var_values = []
        patient_data = {}
        if self.data is not None:
            values = self.data.data_files.datasets
            texts = [
                "Magnitude",
                "Through-plane velocity map (Z)",
                "In-plane velocity map (X)",
                "In-plane velocity map (Y)",
            ]
            var_values = ["MAG", "Z", "X", "Y"]
            patient_data = self.data.metadata()

        return values, texts, var_values, patient_data

    @property
    def images(self) -> xr.DataArray:
        variable = self.maps_var.get()
        series = self.datasets_var.get()
        return self.data.data_files.images(series).sel(comp=variable)

    def update_visualization(self, *args):
        """Updates the visualization whenever the data selected changes."""
        self.update_plot()
        self.update_dicom_data_view()

    def update_plot(self):
        """Updates the data contained in the plot."""
        self.fig.actions_manager.ScrollFrames.clear()

        ax = self.fig.axes[-1]
        clim = xlim = ylim = None
        if len(ax.images) > 0:
            clim = ax.images[0].get_clim()
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

        ax.clear()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        if len(self.images) == 0:
            return

        ax.imshow(self.images[0], cmap=plt.get_cmap("binary_r"))

        self.fig.actions_manager.ScrollFrames.set_scroller(
            partial(scroll, self.images.values), ax
        )

        self.fig.canvas.draw_idle()
        if clim is not None:
            ax.images[0].set_clim(*clim)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)

        self.fig.canvas.draw()

    def update_dicom_data_view(self):
        """Updates the treeview with data from the selected options.

        Only data for cine = 0 is loaded.
        """
        variable = self.maps_var.get()
        series = self.datasets_var.get()
        self.treeview.delete(*self.treeview.get_children())

        data = self.data.data_files.tags(series, variable) if series != "" else []

        for d in data:
            value = data.get(d)
            if isinstance(value, bytes):
                value = "{binary data}"
            self.treeview.insert("", tk.END, values=(d, value))

    def stop_animation(self):
        """Stops an animation, if there is one running."""
        if hasattr(self.fig, "actions_manager") and hasattr(
            self.fig.actions_manager, "ScrollFrames"
        ):
            self.fig.actions_manager.ScrollFrames.stop_animation()

    def update_widgets(self):
        """Updates widgets after an update in the data var."""
        self.create_data_selector()
        self.create_data_viewer()
        if self.data is not None and self.data.data_files is not None:
            self.update_visualization()
        self.datafolder_entry.xview(len(self.data_folder.get()))
        self.outputfile_entry.xview(len(self.output_file.get()))

    def clear_widgets(self):
        """Clear widgets after removing the data."""
        self.data_folder.set("")
        self.output_file.set("")
        self.nametowidget("control.chooseOutputFile")["state"] = "disabled"

        if self.notebook:
            self.dataselector.grid_remove()
            self.notebook.destroy()
            self.notebook = None
            self.dataselector = None


def scroll(images: np.ndarray, frame: int) -> Tuple[int, Optional[np.ndarray], None]:
    """Provides the next image when scrolling."""
    if len(images) == 0:
        return 0, None, None

    current_frame = frame % images.shape[0]
    return current_frame, images[current_frame], None
