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

        self.output_file = tk.StringVar()
        self.data_folder = tk.StringVar()
        self.bg_folder = tk.StringVar()
        self.notebook = None

        self.create_controls()

    def create_controls(self) -> None:

        self.control.columnconfigure(0, weight=1)

        # New series widgets -----------
        new_series = ttk.Labelframe(self.control, text="New series analysis")
        new_series.grid(sticky=(tk.EW, tk.N), padx=5, pady=10)
        new_series.columnconfigure(0, weight=1)

        ttk.Button(
            master=new_series,
            text="Choose data folder",
            padding=5,
            command=self.load_data,
        ).grid(sticky=tk.NSEW, padx=5, pady=5)

        # Resume analysis widgets -------
        resume = ttk.Labelframe(self.control, text="Resume analysis")
        resume.grid(sticky=(tk.EW, tk.N), padx=5, pady=15)
        resume.columnconfigure(0, weight=1)

        ttk.Button(
            master=resume,
            text="Open StrainMap file",
            padding=5,
            command=self.open_existing_file,
        ).grid(column=0, row=0, sticky=tk.NSEW, padx=5, pady=5)

        ttk.Button(
            master=new_series,
            text="Choose output file",
            padding=5,
            command=self.select_output_file,
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

        # Current background folder widgets -----------
        info = ttk.Labelframe(self.control, text="Current background folder:")
        info.grid(sticky=(tk.EW, tk.N), padx=5, pady=5)
        info.columnconfigure(0, weight=1)

        ttk.Entry(
            master=info, textvariable=self.bg_folder, state="disabled", justify="center"
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
            text="Clear all data",
            padding=5,
            command=self.clear_data,
        ).grid(sticky=(tk.EW, tk.S), padx=5, pady=5)

    def load_data(self):

        path = tk.filedialog.askdirectory(title="Select DATA directory")
        if path != "":
            self.actions.load_data(data_files=path)
            self.data_folder.set(Path(path).parent)

    def open_existing_file(self):
        """ Opens an existing StrainMap file."""
        messagebox.showinfo(message="This functionality is not implemented, yet.")
        self.output_file.set("")

    def select_output_file(self):
        """ Selects an output file in which to store the analysis."""
        messagebox.showinfo(message="This functionality is not implemented, yet.")
        self.output_file.set("")

    def clear_data(self):
        """ Clears all data from memory."""

        answer = messagebox.askokcancel(
            "Warning!",
            "This will erase all data from memory\nDo you want to continue?",
            icon="warning",
        )
        if answer:
            self.actions.clear_data()

    def create_tabs(self):
        """ Loads the child tabs that are available after loading the data."""
        from .animate_beating import Animation
        from .view_dicom_data import DICOMData

        self.notebook = ttk.Notebook(self.visualise)
        self.notebook.grid(sticky=tk.NSEW)
        self.notebook.columnconfigure(0, weight=1)
        self.notebook.rowconfigure(0, weight=1)

        anim = Animation(self.notebook, self.data)
        dicom = DICOMData(self.notebook, self.data)

        self.notebook.add(anim, text="Animation")
        self.notebook.add(dicom, text="DICOM Data")

    def update_widgets(self):
        """ Updates widgets after an update in the data variable. """
        self.create_tabs()

    def clear_widgets(self):
        """ Clear widgets after removing the data. """
        self.data_folder.set("")
        self.bg_folder.set("")
        self.output_file.set("")

        if self.notebook:
            self.notebook.destroy()
            self.notebook = None
