import tkinter as tk
import tkinter.filedialog
from pathlib import Path
from tkinter import messagebox, ttk

from ..base_classes import TaskBase, register_task, DataLoaded


@register_task
class RawData(TaskBase):
    def __init__(self, root):

        super().__init__(root, button_text="Raw data", button_image="save.gif")

        self.series_types_var = tk.StringVar()
        self.bg_types_var = tk.StringVar()
        self.output_file = tk.StringVar()
        self.data_folder = tk.StringVar()
        self.bg_folder = tk.StringVar()
        self.series_types_combobox = None
        self.phantom_button = None

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
            command=self.skim_series_data,
        ).grid(sticky=tk.NSEW, padx=5, pady=5)

        ttk.Label(master=new_series, text="Available series:").grid(
            sticky=tk.NSEW, padx=5
        )

        self.series_types_combobox = ttk.Combobox(
            master=new_series,
            textvariable=self.series_types_var,
            values=[],
            state="disabled",
        )
        self.series_types_combobox.grid(column=0, sticky=tk.NSEW, padx=5, pady=5)

        ttk.Label(master=new_series, text="Background subtraction method:").grid(
            sticky=tk.NSEW, padx=5
        )
        cbox = ttk.Combobox(
            master=new_series,
            textvariable=self.bg_types_var,
            values=["None", "Phantom", "Method"],
            state="readonly",
        )
        cbox.grid(sticky=tk.NSEW, padx=5, pady=5)
        cbox.bind("<<ComboboxSelected>>", self.bg_subtraction_selected)
        cbox.set("None")

        self.phantom_button = ttk.Button(
            master=new_series,
            text="Choose background folder",
            padding=5,
            command=self.skim_bg_data,
            state="disabled",
        )
        self.phantom_button.grid(sticky=tk.NSEW, padx=5, pady=5)

        ttk.Button(
            master=new_series,
            text="Choose output file",
            padding=5,
            command=self.select_output_file,
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

        self.controls_created = True

    def skim_series_data(self):

        path = tk.filedialog.askdirectory()
        values = sorted(self.data.fill_data_files(path).keys())
        self.data_folder.set(Path(path).parent)

        self.series_types_combobox["values"] = values
        if len(values) > 0:
            self.series_types_var.set(values[0])
            self.series_types_combobox["state"] = "enable"
            self.series_types_combobox.event_generate(DataLoaded.UNLOCK)
        else:
            self.series_types_var.set("")
            self.series_types_combobox["state"] = "disabled"
            self.series_types_combobox.event_generate(DataLoaded.LOCK)

    def skim_bg_data(self):

        path = tk.filedialog.askdirectory()
        self.data.fill_bg_files(path)
        self.bg_folder.set(Path(path).parent)

    def bg_subtraction_selected(self, event):

        bg_type = event.widget.get()
        if bg_type == "Phantom":
            self.phantom_button["state"] = "enable"
        else:
            self.phantom_button["state"] = "disabled"

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
            self.data.clear()
            self.series_types_combobox["values"] = []
            self.series_types_var.set("")
            self.bg_types_var.set("None")
            self.data_folder.set("")
            self.bg_folder.set("")
            self.output_file.set("")
            self.phantom_button["state"] = "disabled"
            self.series_types_combobox.event_generate(DataLoaded.LOCK)
