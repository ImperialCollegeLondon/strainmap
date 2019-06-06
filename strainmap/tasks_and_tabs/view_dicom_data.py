import tkinter as tk
from pathlib import Path
from tkinter import ttk

from ..base_classes import TabBase, register_tab, DataLoaded


@register_tab
class DICOMData(TabBase):

    pre_requisites = {DataLoaded}

    def __init__(self, root):

        super().__init__(root, tab_text="DICOM Data")

        first_series = sorted(self.data.data_files.keys())[0]
        self.series_types_var = tk.StringVar(value=first_series)
        self.variables_var = tk.StringVar(value="MagZ")
        self.files_box = None

    def create_tab_contents(self):
        """ Creates the contents of the tab. """
        for i in range(3):
            self.tab.columnconfigure(i, weight=1)
        self.tab.rowconfigure(1, weight=1)

        series = ttk.Labelframe(self.tab, text="Available series:")
        series.grid(column=0, row=0, sticky=(tk.EW, tk.N), padx=5, pady=5)
        series.columnconfigure(0, weight=1)

        cbox1 = ttk.Combobox(
            master=series,
            textvariable=self.series_types_var,
            values=sorted(self.data.data_files.keys()),
            state="enable",
        )
        cbox1.grid(sticky=tk.NSEW, padx=5, pady=5)
        cbox1.bind("<<ComboboxSelected>>", self.update_files_box)

        variable = ttk.Labelframe(self.tab, text="Select variable:")
        variable.grid(column=1, row=0, sticky=(tk.EW, tk.N), padx=5, pady=5)
        variable.columnconfigure(0, weight=1)

        cbox2 = ttk.Combobox(
            master=variable,
            textvariable=self.variables_var,
            values=["MagZ", "PhaseZ", "MagX", "PhaseX", "MagY", "PhaseY"],
            state="enable",
        )
        cbox2.grid(sticky=tk.NSEW, padx=5, pady=5)
        cbox2.bind("<<ComboboxSelected>>", self.update_files_box)

        files = ttk.Labelframe(self.tab, text="Select variable:")
        files.grid(column=2, row=0, sticky=(tk.EW, tk.N), padx=5, pady=5)
        files.columnconfigure(0, weight=1)

        self.files_box = ttk.Combobox(master=files, values=[], state="enable")
        self.files_box.grid(sticky=tk.NSEW, padx=5, pady=5)
        self.files_box.bind("<<ComboboxSelected>>", self.update_tree)

        tree = ttk.Frame(self.tab)
        tree.grid(column=0, row=1, columnspan=3, sticky=tk.NSEW, padx=5, pady=5)
        tree.columnconfigure(0, weight=1)
        tree.rowconfigure(0, weight=1)

        self.treeview = ttk.Treeview(tree, selectmode="browse")
        vsb = ttk.Scrollbar(tree, orient="vertical", command=self.treeview.yview())
        self.treeview.configure(yscrollcommand=vsb.set)
        self.treeview.grid(column=0, row=0, sticky=tk.NSEW, padx=5, pady=5)
        vsb.grid(column=1, row=0, sticky=tk.NSEW)

        self.treeview["columns"] = ("1", "2")
        self.treeview["show"] = "headings"
        self.treeview.column("1", width=300, stretch=False)
        self.treeview.heading("1", text="Tags")
        self.treeview.heading("2", text="Values")

        self.update_files_box()

    def update_tree(self, *args):
        """ Updates the treeview with data from the selected options. """

        self.treeview.delete(*self.treeview.get_children())

        series = self.series_types_var.get()
        variable = self.variables_var.get()
        idx = self.files_box.current()

        data = self.data.get_DICOM_file(series, variable, idx)
        for i, d in enumerate(data.dir()):
            self.treeview.insert("", tk.END, values=(d, getattr(data, d)))

    def update_files_box(self, *args):
        """ Updates the contents of the files combobox when others changes. """
        series = self.series_types_var.get()
        variable = self.variables_var.get()

        self.files_box["values"] = [
            Path(f).name for f in self.data.data_files[series][variable]
        ]
        self.files_box.current(0)

        self.update_tree()
