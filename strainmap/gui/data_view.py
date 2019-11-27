import tkinter as tk
import tkinter.filedialog
from tkinter import messagebox, ttk
import os
from functools import partial

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .base_window_and_task import TaskViewBase, register_view, trigger_event
from .figure_actions import BrightnessAndContrast, ScrollFrames, ZoomAndPan
from .figure_actions_manager import FigureActionsManager


@register_view
class DataTaskView(TaskViewBase):
    def __init__(self, root, controller):

        super().__init__(root, controller, button_text="Data", button_image="save.gif")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)

        # Control-related attributes
        self.data_folder = tk.StringVar(value="")
        self.output_file = tk.StringVar(value="")
        self.current_dir = os.path.expanduser("~")
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

        ttk.Entry(
            master=self.control, textvariable=self.data_folder, state="disabled"
        ).grid(row=4, column=1, sticky=tk.NSEW, pady=5)

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

    @trigger_event(name="load_data_from_folder")
    def load_data(self):
        """Loads new data into StrainMap"""
        path = tk.filedialog.askdirectory(
            title="Select DATA directory", initialdir=self.current_dir
        )

        output = {}
        if path != "":
            self.current_dir = path
            output = dict(data_files=path)
            self.data_folder.set(self.current_dir)

        return output

    @trigger_event(name="add_paths")
    def load_phantom(self):
        """Loads phantom data into a data structure."""

        result = {}
        if self.phantom_check.get():
            path = tk.filedialog.askdirectory(
                title="Select PHANTOM directory", initialdir=self.current_dir
            )
            if path == "":
                self.phantom_check.set(False)
            else:
                result = dict(bg_files=path)
        else:
            self.phantoms_box.grid_remove()
            result = dict(bg_files="")

        return result

    @trigger_event(name="add_paths")
    def load_missing_data(self, data_missing=False, phantom_missing=False):

        data_path = phantom_path = None
        if data_missing:
            data_path = tk.filedialog.askdirectory(
                title="Select DATA directory", initialdir=self.current_dir
            )
        if phantom_missing:
            phantom_path = tk.filedialog.askdirectory(
                title="Select PHANTOM directory", initialdir=self.current_dir
            )

        if data_path is not None:
            self.current_dir = data_path
            self.data_folder.set(self.current_dir)

        return dict(data_files=data_path, bg_files=phantom_path)

    @trigger_event(name="load_data_from_file")
    def open_existing_file(self):
        """ Opens an existing StrainMap file."""
        path = tk.filedialog.askopenfilename(
            title="Select existing StrainMap file (HDF5 format)",
            initialdir=self.current_dir,
            filetypes=(("StrainMap files", "*.h5"),),
        )

        output = {}
        if path != "":
            output = dict(strainmap_file=path)
            self.output_file.set(path)

        return output

    @trigger_event(name="add_h5_file")
    def select_output_file(self):
        """ Selects an output file in which to store the current data."""
        meta = self.data.metadata()
        name, date = [meta[key] for key in ["Patient Name", "Date of Scan"]]
        init = f"{name}_{date}.h5"

        path = tk.filedialog.asksaveasfilename(
            title="Introduce new StrainMap filename.",
            initialfile=init,
            initialdir=self.current_dir,
            filetypes=(("StrainMap files", "*.h5"),),
            defaultextension="h5",
        )

        output = {}
        if path != "":
            output = dict(strainmap_file=path)
            self.output_file.set(path)

        return output

    @trigger_event
    def clear_data(self):
        """ Clears all data from memory."""
        clear = messagebox.askokcancel(
            "Warning!",
            "This will erase all data from memory\nDo you want to continue?",
            icon="warning",
        )
        return {"clear": clear}

    def get_data_information(self):
        """ Gets some information related to the available datasets, frames, etc. """
        values = list(self.data.data_files.keys())
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

        self.update_plot(series, variable)
        self.update_dicom_data_view(series, variable)

    def update_plot(self, series, variable):
        """Updates the data contained in the plot."""
        images = self.data.get_images(series, variable)

        self.fig.actions_manager.ScrollFrames.clear()

        ax = self.fig.axes[-1]
        ax.clear()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        if len(images) == 0:
            return

        ax.imshow(images[0], cmap=plt.get_cmap("binary_r"))

        self.fig.actions_manager.ScrollFrames.set_scroller(
            partial(self.scroll, images), ax
        )

        self.fig.canvas.draw()

    @staticmethod
    def scroll(images, frame):
        """Provides the next images when scrolling."""
        if len(images) == 0:
            return 0, None, None

        current_frame = frame % images.shape[0]
        return current_frame, images[current_frame], None

    def update_dicom_data_view(self, series, variable):
        """ Updates the treeview with data from the selected options.

        Only data for cine = 0 is loaded."""

        self.treeview.delete(*self.treeview.get_children())

        data = self.data.read_dicom_file_tags(series, variable, 0)
        for d in data:
            self.treeview.insert("", tk.END, values=(d, data.get(d)))

    def stop_animation(self):
        """Stops an animation, if there is one running."""
        if hasattr(self.fig, "actions_manager") and hasattr(
            self.fig.actions_manager, "ScrollFrames"
        ):
            self.fig.actions_manager.ScrollFrames.stop_animation()

    def update_widgets(self):
        """ Updates widgets after an update in the data variable. """
        values = list(self.data.data_files.keys())
        if len(values) > 0:
            self.nametowidget("control.chooseOutputFile")["state"] = "enable"
        else:
            self.nametowidget("control.chooseOutputFile")["state"] = "disable"

        data_missing = (
            len(values) > 0 and len(self.data.data_files[values[0]]["MagZ"]) == 0
        )
        values = list(self.data.bg_files.keys())
        phantom_missing = (
            len(values) > 0 and len(self.data.bg_files[values[0]]["MagZ"]) == 0
        )

        if data_missing or phantom_missing:
            messagebox.showwarning(
                "DICOM data not found! " * data_missing
                + "PHANTOM data not found!" * phantom_missing,
                message="Data paths found in the StrainMap data file do not exist."
                "Choose an alternative folder for the files.",
            )
            self.load_missing_data(
                data_missing=data_missing, phantom_missing=phantom_missing
            )

        filename = (
            self.data.strainmap_file.filename
            if self.data.strainmap_file is not None
            else None
        )
        self.output_file.set(filename)
        self.create_data_selector()
        self.create_data_viewer()
        self.update_visualization()

        if len(values) > 0:
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
