import tkinter as tk
import tkinter.filedialog
from pathlib import Path
from tkinter import messagebox, ttk

from .base_classes import ViewBase, register_view


@register_view
class DataView(ViewBase):

    actions = ("load_data", "clear_data")

    def __init__(self, root, actions):

        super().__init__(root, actions, button_text="Data", button_image="save.gif")

        self.data_folder = tk.StringVar()
        self.output_file = tk.StringVar()
        self.notebook = None

        self.create_controls()

    def create_controls(self) -> None:

        self.control.columnconfigure(0, weight=1)

        ttk.Button(
            master=self.control,
            name="chooseDataFolder",
            text="New analysis from data folder",
            padding=5,
            command=self.load_data,
        ).grid(sticky=tk.NSEW, padx=5, pady=5)

        ttk.Button(
            master=self.control,
            name="openStrainMapFile",
            text="Resume analysis from StrainMap file",
            padding=5,
            command=self.open_existing_file,
        ).grid(sticky=tk.NSEW, padx=5, pady=5)

        ttk.Button(
            master=self.control,
            name="chooseOutputFile",
            text="Save analysis as...",
            padding=5,
            command=self.select_output_file,
            state="disabled",
        ).grid(sticky=tk.NSEW, padx=5, pady=5)

        # Current data folder widgets -----------
        info = ttk.Labelframe(self.control, text="Current data folder:")
        info.grid(row=60, sticky=(tk.EW, tk.N), padx=5, pady=5)
        info.columnconfigure(0, weight=1)

        ttk.Entry(
            master=info,
            textvariable=self.data_folder,
            state="disabled",
            justify="center",
        ).grid(sticky=tk.NSEW, padx=5)

        # Current output file widgets -----------
        info = ttk.Labelframe(self.control, text="Current output file:")
        info.grid(sticky=(tk.EW, tk.N), padx=5, pady=5)
        info.columnconfigure(0, weight=1)

        ttk.Entry(
            master=info,
            textvariable=self.output_file,
            state="disabled",
            justify="center",
        ).grid(sticky=tk.NSEW, padx=5)

        # Clear data widget -------------
        ttk.Button(
            master=self.control,
            name="clearAllData",
            text="Clear all data",
            padding=5,
            command=self.clear_data,
        ).grid(sticky=tk.NSEW, padx=5, pady=5)

    def load_data(self):

        path = tk.filedialog.askdirectory(title="Select DATA directory")

        if path != "":
            self.actions.load_data(data_files=path)
            self.data_folder.set(Path(path))

    def clear_data(self):
        """ Clears all data from memory."""

        answer = messagebox.askokcancel(
            "Warning!",
            "This will erase all data from memory\nDo you want to continue?",
            icon="warning",
        )
        if answer:
            self.actions.clear_data()

    def open_existing_file(self):
        """ Opens an existing StrainMap file."""
        messagebox.showinfo(message="This functionality is not implemented, yet.")
        self.output_file.set("")

    def select_output_file(self):
        """ Selects an output file in which to store the analysis."""
        messagebox.showinfo(message="This functionality is not implemented, yet.")
        self.output_file.set("")

    def create_tabs(self):
        """ Loads the child tabs that are available after loading the data."""
        from .animate_beating import Animation
        from .view_dicom_data import DICOMData

        self.notebook = ttk.Notebook(self.visualise, name="notebook")
        self.notebook.grid(sticky=tk.NSEW)
        self.notebook.columnconfigure(0, weight=1)
        self.notebook.rowconfigure(0, weight=1)

        anim = Animation(self.notebook, self.data)
        dicom = DICOMData(self.notebook, self.data)

        self.notebook.add(anim, text="Animation")
        self.notebook.add(dicom, text="DICOM Data")

    def update_widgets(self):
        """ Updates widgets after an update in the data variable. """
        self.nametowidget("control.chooseOutputFile")["state"] = "enable"
        self.create_tabs()

    def clear_widgets(self):
        """ Clear widgets after removing the data. """
        self.data_folder.set("")
        self.output_file.set("")
        self.nametowidget("control.chooseOutputFile")["state"] = "disabled"

        if self.notebook:
            self.notebook.destroy()
            self.notebook = None
