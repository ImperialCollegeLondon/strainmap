import tkinter as tk
import tkinter.filedialog
from tkinter import messagebox, ttk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .base_window_and_task import TaskViewBase, register_view, trigger_event
from .figure_actions import (
    BrightnessAndContrast,
    ScrollFrames,
    ZoomAndPan,
    data_generator,
)
from .figure_actions_manager import FigureActionsManager


@register_view
class DataTaskView(TaskViewBase):
    def __init__(self, root):

        super().__init__(root, button_text="Data", button_image="save.gif")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)

        # Control-related attributes
        self.data_folder = tk.StringVar(value="")
        self.output_file = tk.StringVar(value="")
        self.phantom_check = tk.BooleanVar(value=False)
        self.control = None
        self.dataselector = None
        self.patient_info = None

        # Visualization-related attributes
        self.visualise = None
        self.notebook = None
        self.treeview = None
        self.time_step = None
        self.fig = None
        self.datasets_var = tk.StringVar(value="")
        self.maps_var = tk.StringVar(value="MagZ")
        self.cine_frame_var = tk.IntVar(value=0)
        self.phantoms_box = None
        self.anim = False

        self.create_controls()

    def create_controls(self) -> None:
        """ Creates the controls to load and save data."""
        self.control = ttk.Frame(master=self, width=300, name="control")
        self.control.grid(column=0, row=0, sticky=tk.NSEW, padx=10, pady=10)
        self.control.rowconfigure(50, weight=1)
        self.control.grid_propagate(flag=False)
        self.control.columnconfigure(1, weight=1)

        self.visualise = ttk.Frame(master=self, name="visualise")
        self.visualise.grid(column=1, row=0, sticky=tk.NSEW, padx=10, pady=10)
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
            text="Resume analysis",
            command=self.open_existing_file,
            width=25,
        ).grid(sticky=tk.NSEW, columnspan=2)

        ttk.Button(
            master=self.control,
            name="chooseOutputFile",
            text="Save analysis as...",
            command=self.select_output_file,
            state="disabled",
        ).grid(sticky=tk.NSEW, columnspan=2)

        ttk.Label(master=self.control, text="Data folder: ").grid(row=4, sticky=tk.W)

        ttk.Entry(
            master=self.control, textvariable=self.data_folder, state="disabled"
        ).grid(row=4, column=1, sticky=tk.NSEW)

        ttk.Label(master=self.control, text="Output file: ").grid(row=5, sticky=tk.W)

        ttk.Entry(
            master=self.control, textvariable=self.output_file, state="disabled"
        ).grid(row=5, column=1, sticky=tk.NSEW)

        ttk.Button(
            master=self.control,
            name="clearAllData",
            text="Clear all data",
            command=self.clear_data,
        ).grid(row=99, columnspan=2, sticky=tk.NSEW)

    def create_data_selector(self):
        """ Creates the selector for the data. """
        values, texts, var_values, cine_frames, patient_data = (
            self.get_data_information()
        )
        self.datasets_var.set(values[0])
        self.cine_frame_var.set(cine_frames[0])

        patient_name = patient_data.get("PatientName", "")
        patient_dob = patient_data.get("PatientBirthDate", "")
        patient_study_date = patient_data.get("StudyDate", "")

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

        ttk.Checkbutton(
            master=self.dataselector,
            text="Use phantom subtraction.",
            command=self.load_phantom,
            variable=self.phantom_check,
            onvalue=True,
            offvalue=False,
            state="enable" if len(patient_data) > 0 else "disabled",
        ).grid(sticky=tk.NSEW, padx=5, pady=5)

        self.phantoms_box = ttk.Combobox(
            master=self.dataselector, values=[], state="readonly"
        )

    def create_data_viewer(self):
        """ Creates the viewer for the data, including the animation and the DICOM. """
        self.notebook = ttk.Notebook(self.visualise, name="notebook")
        self.notebook.grid(sticky=tk.NSEW)
        self.notebook.columnconfigure(0, weight=1)
        self.notebook.rowconfigure(0, weight=1)

        self.notebook.add(self.create_animation_viewer(), text="Animation")
        self.notebook.add(self.create_dicom_viewer(), text="DICOM Data")

    def create_animation_viewer(self):
        """ Creates the animation plot area. """

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

        dicom_frame = ttk.Frame(self.notebook)
        dicom_frame.columnconfigure(0, weight=1)
        dicom_frame.rowconfigure(0, weight=1)

        self.treeview = ttk.Treeview(dicom_frame, selectmode="browse")
        vsb = ttk.Scrollbar(
            dicom_frame, orient="vertical", command=self.treeview.yview()
        )
        self.treeview.configure(yscrollcommand=vsb.set)
        self.treeview.grid(column=0, row=0, sticky=tk.NSEW, padx=5, pady=5)
        vsb.grid(column=1, row=0, sticky=tk.NSEW)

        self.treeview["columns"] = ("1", "2")
        self.treeview["show"] = "headings"
        self.treeview.column("1", width=300, stretch=False)
        self.treeview.heading("1", text="Tags")
        self.treeview.heading("2", text="Values")

        return dicom_frame

    @trigger_event
    def load_data(self):
        """Loads new data into StrainMap"""
        path = tk.filedialog.askdirectory(title="Select DATA directory")
        self.data_folder.set(path)

        output = {}
        if path != "":
            output = dict(data_files=path)

        return output

    @trigger_event(name="load_data")
    def load_phantom(self):
        """Loads phantom data into a data structure."""

        result = {}
        if self.phantom_check.get():
            path = tk.filedialog.askdirectory(title="Select PHANTOM directory")
            if path == "":
                self.phantom_check.set(False)
            else:
                result = dict(bg_files=path, data=self.data)
        else:
            self.phantoms_box.grid_remove()
            result = dict(bg_files="", data=self.data)

        return result

    @trigger_event
    def clear_data(self):
        """ Clears all data from memory."""
        clear = messagebox.askokcancel(
            "Warning!",
            "This will erase all data from memory\nDo you want to continue?",
            icon="warning",
        )
        return {"clear": clear}

    def open_existing_file(self):
        """ Opens an existing StrainMap file."""
        messagebox.showinfo(message="This functionality is not implemented, yet.")
        self.output_file.set("")

    def select_output_file(self):
        """ Selects an output file in which to store the analysis."""
        messagebox.showinfo(message="This functionality is not implemented, yet.")
        self.output_file.set("")

    def get_data_information(self):
        """ Gets some information related to the available datasets, frames, etc. """
        values = sorted(self.data.data_files.keys())
        if len(values) > 0:
            texts = [
                "Magnitude",
                "Through-plane velocity map (Z)",
                "In-plane velocity map (X)",
                "In-plane velocity map (Y)",
            ]
            var_values = ["MagZ", "PhaseZ", "PhaseX", "PhaseY"]
            cine_frames = [*range(len(self.data.data_files[values[0]]["MagZ"]))]
            patient_data = self.data.read_dicom_file_tags(values[0], "MagZ", 0)
        else:
            values = ["No suitable datasets available."]
            texts = []
            var_values = []
            cine_frames = [""]
            patient_data = {}

        return values, texts, var_values, cine_frames, patient_data

    def update_visualization(self, *args):
        """ Updates the visualization whenever the data selected changes. """
        series = self.datasets_var.get()
        variable = self.maps_var.get()
        data = self.data.get_images(series, variable)

        if len(data) == 0:
            return

        self.update_plot(data)
        self.update_dicom_data_view(series, variable)

    def update_plot(self, data):
        """Updates the data contained in the plot."""
        self.fig.actions_manager.ScrollFrames.clear()

        ax = self.fig.axes[-1]
        ax.clear()

        ax.imshow(data[0], cmap=plt.get_cmap("binary_r"))

        self.fig.actions_manager.ScrollFrames.set_generators(data_generator(data), ax)

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        self.fig.canvas.draw()

    def update_dicom_data_view(self, series, variable):
        """ Updates the treeview with data from the selected options.

        Only data for cine = 0 is loaded."""

        self.treeview.delete(*self.treeview.get_children())

        data = self.data.read_dicom_file_tags(series, variable, 0)
        for d in data:
            self.treeview.insert("", tk.END, values=(d, data.get(d)))

    def stop_animation(self):
        """Stops an animation, if there is one running."""
        self.fig.actions_manager.ScrollFrames.stop_animation()

    def update_widgets(self):
        """ Updates widgets after an update in the data variable. """
        self.nametowidget("control.chooseOutputFile")["state"] = "enable"
        self.create_data_selector()
        self.create_data_viewer()
        self.update_visualization()

        values = sorted(self.data.bg_files.keys())
        if self.phantom_check.get() and len(values) > 0:
            self.phantoms_box["values"] = values
            self.phantoms_box.current(0)
            self.phantoms_box.grid(column=0, sticky=tk.NSEW, padx=5, pady=5)
        else:
            self.phantom_check.set(False)

    def clear_widgets(self):
        """ Clear widgets after removing the data. """
        self.data_folder.set("")
        self.output_file.set("")
        self.nametowidget("control.chooseOutputFile")["state"] = "disabled"

        if self.notebook:
            self.dataselector.grid_remove()
            self.phantom_check.set(False)
            self.notebook.destroy()
            self.notebook = None
            self.dataselector = None
