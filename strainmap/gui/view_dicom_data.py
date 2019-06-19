import tkinter as tk
from pathlib import Path
from tkinter import ttk


class DICOMData(ttk.Frame):
    def __init__(self, root, data):

        super().__init__(root)

        self.data = data

        self.series_types_var = tk.StringVar()
        self.variables_var = tk.StringVar(value="MagZ")
        self.files_box = None

        self.create_tab_contents()

    def create_tab_contents(self):
        """ Creates the contents of the tab. """
        for i in range(3):
            self.columnconfigure(i, weight=1)
        self.rowconfigure(1, weight=1)

        series = ttk.Labelframe(self, text="Available series:", name="series")
        series.grid(column=0, row=0, sticky=(tk.EW, tk.N), padx=5, pady=5)
        series.columnconfigure(0, weight=1)

        values = sorted(self.data.data_files.keys())
        self.series_types_var.set(values[0])
        cbox1 = ttk.Combobox(
            master=series,
            name="seriesBox",
            textvariable=self.series_types_var,
            values=values,
            state="readonly",
        )
        cbox1.grid(sticky=tk.NSEW, padx=5, pady=5)
        cbox1.bind("<<ComboboxSelected>>", self.update_files_box)

        variable = ttk.Labelframe(self, text="Select variable:", name="variables")
        variable.grid(column=1, row=0, sticky=(tk.EW, tk.N), padx=5, pady=5)
        variable.columnconfigure(0, weight=1)

        cbox2 = ttk.Combobox(
            master=variable,
            name="variablesBox",
            textvariable=self.variables_var,
            values=["MagZ", "PhaseZ", "MagX", "PhaseX", "MagY", "PhaseY"],
            state="readonly",
        )
        cbox2.grid(sticky=tk.NSEW, padx=5, pady=5)
        cbox2.bind("<<ComboboxSelected>>", self.update_files_box)

        files = ttk.Labelframe(self, text="Select file:", name="files")
        files.grid(column=2, row=0, sticky=(tk.EW, tk.N), padx=5, pady=5)
        files.columnconfigure(0, weight=1)

        self.files_box = ttk.Combobox(
            master=files, name="filesBox", values=[], state="enable"
        )
        self.files_box.grid(sticky=tk.NSEW, padx=5, pady=5)
        self.files_box.bind("<<ComboboxSelected>>", self.update_tree)

        tree = ttk.Frame(self)
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

        data = self.data.read_dicom_file_tags(series, variable, idx)
        for d in data:
            self.treeview.insert("", tk.END, values=(d, data.get(d)))

    def update_files_box(self, *args):
        """ Updates the contents of the files combobox when others changes. """
        series = self.series_types_var.get()
        variable = self.variables_var.get()

        self.files_box["values"] = [
            Path(f).name for f in self.data.data_files[series][variable]
        ]
        self.files_box.current(0)

        self.update_tree()
