from __future__ import annotations

import tkinter as tk
import tkinter.filedialog
from tkinter import messagebox, ttk
import os
from functools import partial
from pathlib import Path
from typing import Dict, Callable, Union, Optional

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np


from .base_window_and_task import TaskViewBase, register_view


@register_view
class AtlasTaskView(TaskViewBase):
    def __init__(self, root, controller):

        super().__init__(
            root,
            controller,
            button_text="Atlas",
            button_image="atlas.gif",
            button_row=99,
        )
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.notebook = ttk.Notebook(self, name="notebook")
        self.notebook.grid(sticky=tk.NSEW)
        self.notebook.columnconfigure(0, weight=1)
        self.notebook.rowconfigure(0, weight=1)

        self.path = None
        self.atlas_data = self.load_atlas_data(self.path)
        self.pss = self.create_plot("PSS", self.atlas_data)
        self.ess = self.create_plot("ESS", self.atlas_data)
        self.ps = self.create_plot("PS", self.atlas_data)
        self.table = self.create_data_tab(self.atlas_data)

    def create_plot(self, label: str, data: pd.DataFrame) -> GridPlot:
        """Creates a grid plot and put it as new tab in the notebook.

        Args:
            label (str): Label for the notebook tab and strain value to use.
            data (pd.DataFrame): Dataframe containing all the data.

        Returns:
            A GridPlot object.
        """
        plot = GridPlot(label, data)

        frame = ttk.Frame(self.notebook)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        canvas = FigureCanvasTkAgg(plot.fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().grid(sticky=tk.NSEW, padx=5, pady=5)

        self.notebook.add(frame, text=label)
        return plot

    def update_plots(self, data: pd.DataFrame):
        """Updates the plots with the new data."""
        self.pss.update_plot(data)
        self.ess.update_plot(data)
        self.ps.update_plot(data)

    def create_data_tab(self, data: pd.DataFrame) -> ttk.Treeview:
        """Creates and populates the table in the data tab and controls.

        Args:
            data (pd.DataFrame): Dataframe containing all the data.

        Returns:
            A Treeview object.
        """
        frame = ttk.Frame(self.notebook)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(0, weight=1)

        control_frame = ttk.Frame(frame, width=300)
        control_frame.grid(row=0, column=0, sticky=tk.NSEW, padx=10, pady=10)
        control_frame.grid_propagate(flag=False)
        control_frame.columnconfigure(0, weight=1)

        button_frame = buttons_frame(
            control_frame,
            {
                "Add external record": self.add_record,
                "Add current patient": self.add_record,
                "Remove selected record": self.remove_record,
                "Include/Exclude selected record": self.include_exclude_record,
                "Include/Exclude selected row": self.include_exclude_selected,
                "Load atlas": self.load_atlas,
                "Save atlas as...": self.save_as_atlas,
            },
            text="Manage atlas:",
            width=300,
        )
        button_frame.grid(row=0, column=0, sticky=tk.NSEW)

        table_frame = ttk.Frame(frame)
        table_frame.grid(row=0, column=1, sticky=tk.NSEW, padx=10, pady=10)
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)

        treeview = ttk.Treeview(table_frame, selectmode="browse")
        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=treeview.yview())
        treeview.configure(yscrollcommand=vsb.set)
        treeview.grid(column=0, row=0, sticky=tk.NSEW, padx=5, pady=5)
        vsb.grid(column=1, row=0, sticky=tk.NSEW)

        treeview["columns"] = tuple(data.columns)
        treeview["show"] = "headings"
        for v in treeview["columns"]:
            treeview.column(v, width=80, stretch=True)
            treeview.heading(v, text=v)

        for i, row in data.iterrows():
            treeview.insert("", tk.END, values=tuple(row))

        self.notebook.add(frame, text="Data")
        return treeview

    def update_table(self, data: pd.DataFrame) -> None:
        """Updates the table with the new data.

        Args:
            data (pd.DataFrame): Dataframe containing all the data.
        """
        self.table.delete(*self.table.get_children())
        for i, row in data.iterrows():
            self.table.insert("", tk.END, values=tuple(row))

    def add_record(self, *args):
        pass

    def add_current(self, *args):
        pass

    def remove_record(self, *args):
        if not self.table.selection():
            return

        i = self.table.index(self.table.selection()[0])
        record = self.atlas_data.loc[i, "Record"]
        msg = (
            f"Record {record} will be permanently removed from atlas. "
            f"Do you want to continue?"
        )

        if not messagebox.askyesno(message=msg, title="Confirm remove record"):
            return

        self.atlas_data = self.atlas_data.loc[self.atlas_data.Record != record]

        self.save_atlas(self.path)
        self.update_table(self.atlas_data)
        self.update_plots(self.atlas_data)

    def include_exclude_record(self, *args):
        if not self.table.selection():
            return
        i = self.table.index(self.table.selection()[0])
        record = self.atlas_data.loc[i, "Record"]
        self.atlas_data.loc[
            self.atlas_data.Record == record, "Included"
        ] = ~self.atlas_data.loc[self.atlas_data.Record == record, "Included"]

        self.save_atlas(self.path)
        self.update_table(self.atlas_data)
        self.update_plots(self.atlas_data)

    def include_exclude_selected(self, *args):
        if not self.table.selection():
            return
        i = self.table.index(self.table.selection()[0])
        self.atlas_data.loc[i, "Included"] = not self.atlas_data.loc[i, "Included"]

        self.save_atlas(self.path)
        self.update_table(self.atlas_data)
        self.update_plots(self.atlas_data)

    def load_atlas(self, *args, path: Optional[str] = None):
        if not path:
            filename = tk.filedialog.askopenfilename(
                title="Select strain atlas file",
                initialdir=Path.home(),
                filetypes=(("CSV file", "*.csv"),),
            )
            if filename == "":
                return
            path = Path(filename)

        self.atlas_data = self.load_atlas_data(path)

        self.update_table(self.atlas_data)
        self.update_plots(self.atlas_data)

    def save_as_atlas(self, *args):
        filename = tk.filedialog.asksaveasfilename(
            title="Introduce new name for the strain atlas.",
            initialfile="strain_atlas.csv",
            initialdir=Path.home(),
            filetypes=(("CSV file", "*.csv"),),
            defaultextension="csv",
        )
        if filename == "":
            return

        self.save_atlas(filename)

    def save_atlas(self, filename: Path):
        try:
            self.atlas_data.to_csv(filename, index=False)
            self.path = Path(filename)
        except ValueError as err:
            self.controller.progress(f"Error found when saving: {str(err)}.")

    def load_atlas_data(self, path: Optional[Path] = None):
        """Loads the atlas data from the csv file.

        If path is invalid or empty, an empty atlas data structure is returned and an
        an informative message is provided.
        """
        if not path:
            return empty_data()
        elif not path.is_file():
            self.controller.progress(f"Unknown file {str(path)}.")
            self.path = None
            return empty_data()

        self.controller.progress("")
        try:
            data = validate_data(pd.read_csv(path), self.controller.progress)
            # data = dummy_data()
            self.path = path

        except ValueError as err:
            self.controller.progress(str(err))
            data = empty_data()
            self.path = None

        return data

    def update_widgets(self):
        pass

    def clear_widgets(self):
        pass


def buttons_frame(
    master: Union[ttk.Frame, ttk.LabelFrame],
    buttons: Dict[str, Callable],
    vert=True,
    text: Optional[str] = None,
    **kwargs,
) -> tk.Frame:
    """Creates a frame with buttons in the vertical or horizontal direction."""
    if text is not None:
        frame = ttk.LabelFrame(master, text=text, **kwargs)
    else:
        frame = ttk.Frame(master, **kwargs)

    if vert:
        frame.columnconfigure(0, weight=1)
    else:
        frame.rowconfigure(0, weight=1)

    for i, (text, callback) in enumerate(buttons.items()):
        row, col = (i, 0) if vert else (i, 0)
        ttk.Button(master=frame, text=text, command=callback).grid(
            row=row, column=col, sticky=tk.NSEW, padx=5, pady=5,
        )

    return frame


class GridPlot:

    slices = ("Base", "Mid", "Apex")
    comp = ("Longitudinal", "Radial", "Circumferential")

    def __init__(self, label: str, data: pd.Dataframe):
        self.label = label
        self.fig, self.ax = plt.subplots(3, 3, sharey="all", tight_layout=True)
        self.update_plot(data)

    def update_plot(self, data: pd.DataFrame):
        pad = 5

        for i, s in enumerate(self.slices):
            for j, c in enumerate(self.comp):
                self.ax[i][j].clear()
                d = data.loc[
                    (data["Slice"] == s) & (data["Component"] == c) & data["Included"]
                ]
                d.boxplot(
                    column=[self.label], by=["Region"], ax=self.ax[i][j], grid=False
                )

                self.ax[i][j].xaxis.set_label_text("")
                self.ax[i][j].xaxis.label.set_visible(False)
                self.ax[i][j].set_title(c if i == 0 else "", size=15)
                self.ax[i][j].yaxis.set_label_text("Strain (%)")
                self.ax[i][j].yaxis.set_tick_params(labelleft=True)

                if j == 0:
                    self.ax[i][j].annotate(
                        s,
                        xy=(0, 0.5),
                        xytext=(-self.ax[i][j].yaxis.labelpad - pad, 0),
                        xycoords=self.ax[i][j].yaxis.label,
                        textcoords="offset points",
                        size=15,
                        ha="right",
                        va="center",
                        rotation=90,
                    )
        self.fig.suptitle("")
        self.fig.canvas.draw_idle()


def validate_data(
    data: pd.DataFrame, callback: Optional[Callable] = None
) -> pd.DataFrame:
    """Validates the atlas data, checking and setting column dtypes.

    Args:
        data (pd.DataFrame): Input DataFrame to validate.

    Raises:
        ValueError: If the columns are not the correct ones.

    Returns:
        The validated DataFrame
    """
    cols = (
        "Record",
        "Slice",
        "Component",
        "Region",
        "PSS",
        "ESS",
        "PS",
        "Included",
    )

    if tuple(data.columns) != tuple(cols):
        raise ValueError(
            f"Invalid column names found. They should be, exactly: {cols}."
        )

    slice_type = pd.CategoricalDtype(categories=("Base", "Mid", "Apex"), ordered=False)
    comp_type = pd.CategoricalDtype(
        categories=("Longitudinal", "Radial", "Circumferential"), ordered=False
    )
    region_type = pd.CategoricalDtype(
        categories=("Global", "", "AS", "A", "AL", "IL", "I", "IS"), ordered=False
    )

    col_types = (int, slice_type, comp_type, region_type, float, float, float, bool)

    for col, col_type in zip(cols, col_types):
        data[col] = data[col].astype(col_type)

    dlen = len(data)
    data = data.dropna()
    if dlen != len(data):
        msg = (
            f"Atlas data contains {dlen - len(data)} invalid values. They will be "
            f"removed if databased is saved."
        )
        if callback is None:
            print(msg)
        else:
            callback(msg)

    return data


def empty_data() -> pd.DataFrame:
    Record = pd.Series([])
    Slice = pd.Series([])
    Component = pd.Series([])
    Region = pd.Series([])
    PSS = pd.Series([])
    ESS = pd.Series([])
    PS = pd.Series([])
    Included = pd.Series([])

    return validate_data(
        pd.DataFrame(
            {
                "Record": Record,
                "Slice": Slice,
                "Component": Component,
                "Region": Region,
                "PSS": PSS,
                "ESS": ESS,
                "PS": PS,
                "Included": Included,
            }
        )
    )


def dummy_data() -> pd.DataFrame:
    M = 30
    N = 9 * 7 * M

    Record = pd.Series(np.random.random_integers(1, 20, N))
    Slice = pd.Series(np.random.choice(["Base", "Mid", "Apex"], size=N))
    Component = pd.Series(
        np.random.choice(["Longitudinal", "Radial", "Circumferential"], size=N)
    )
    Region = pd.Series(
        np.random.choice(["Global", "AS", "A", "AL", "IL", "I", "IS"], size=N)
    )
    PSS = pd.Series(np.random.random(N))
    ESS = pd.Series(np.random.random(N))
    PS = pd.Series(np.random.random(N))
    Included = pd.Series([True] * N)

    return validate_data(
        pd.DataFrame(
            {
                "Record": Record,
                "Slice": Slice,
                "Component": Component,
                "Region": Region,
                "PSS": PSS,
                "ESS": ESS,
                "PS": PS,
                "Included": Included,
            }
        )
    )
