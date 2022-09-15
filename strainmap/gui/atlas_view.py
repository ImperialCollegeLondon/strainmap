from __future__ import annotations

import tkinter as tk
import tkinter.filedialog
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Callable, Dict, NamedTuple, Optional, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)

import strainmap.coordinates

from .base_window_and_task import TaskViewBase, register_view  # noqa: F401

COLS: Tuple[str, ...] = (
    "Record",
    "Slice",
    "SAX",
    "Region",
    "Component",
    "PSS",
    "ESS",
    "PS",
    "pssGLS",
    "essGLS",
    "psGLS",
    "Included",
)

SLICES: Tuple[str, ...] = ("Base", "Mid", "Apex")
COMP: Tuple[str, ...] = ("Longitudinal", "Radial", "Circumferential")
REGIONS: Tuple[str, ...] = ("Global", "", "AS", "A", "AL", "IL", "I", "IS")


class SliceBoxes(NamedTuple):
    Base: ttk.Combobox
    Mid: ttk.Combobox
    Apex: ttk.Combobox


# FIXME Re-enable when needed
# @register_view
class AtlasTaskView(TaskViewBase):
    def __init__(self, root, controller):

        super().__init__(
            root,
            controller,
            button_text="Atlas",
            button_image="atlas.gif",
            button_row=99,
        )
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)

        self.notebook = ttk.Notebook(self, name="notebook")
        self.notebook.grid(row=1, column=0, sticky=tk.NSEW)
        self.notebook.columnconfigure(0, weight=1)
        self.notebook.rowconfigure(0, weight=1)

        self.path: Optional[Path] = None
        self.atlas_data = self.load_atlas_data(self.path)
        self.table = self.create_data_tab(self.atlas_data)

        if len(self.atlas_data) > 0:
            self.pss = self.create_plot("PSS", self.atlas_data)
            self.ess = self.create_plot("ESS", self.atlas_data)
            self.ps = self.create_plot("PS", self.atlas_data)
        else:
            self.pss = self.ess = self.ps = None

        self.overlay_data = tk.BooleanVar(value=False)
        self.dataset_box = self.create_datasets_selector()

    def tkraise(self, *args):
        """Brings the frame to the front."""
        super().tkraise()
        self.populate_dataset_box()

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
        canvas.get_tk_widget().grid(sticky=tk.NSEW)
        toolbar_frame = ttk.Frame(master=frame)
        toolbar_frame.grid(row=1, column=0, sticky=tk.NSEW)
        NavigationToolbar2Tk(canvas, toolbar_frame)

        self.notebook.add(frame, text=label)
        return plot

    def update_plots(self):
        """Updates the plots with the new data."""
        self.controller.progress("Updating plots...")

        if self.pss is None:
            self.pss = self.create_plot("PSS", self.atlas_data)
            self.ess = self.create_plot("ESS", self.atlas_data)
            self.ps = self.create_plot("PS", self.atlas_data)

        self.pss.update_plot(self.atlas_data)
        self.ess.update_plot(self.atlas_data)
        self.ps.update_plot(self.atlas_data)

    def overlay(self, data: pd.DataFrame):
        """Updates the plots with the overlaid data."""
        self.pss.overlay(data)
        self.ess.overlay(data)
        self.ps.overlay(data)

    def toggle_overlay(self, *args):
        """Toggles the overlay of the current data."""
        if not self.overlay_data.get():
            self.update_plots()
        else:
            data = self.get_new_data()
            self.overlay(data)

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
                "Add record": self.add_record,
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
        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=treeview.yview)
        treeview.configure(yscrollcommand=vsb.set)
        treeview.grid(column=0, row=0, sticky=tk.NSEW, padx=5, pady=5)
        vsb.grid(column=1, row=0, sticky=tk.NSEW)

        treeview["columns"] = tuple(data.columns)
        treeview["show"] = "headings"
        for v in treeview["columns"]:
            treeview.column(v, width=80, stretch=True)
            treeview.heading(v, text=v)

        for i, row in data.round(decimals=2).iterrows():
            treeview.insert("", tk.END, values=tuple(row))

        self.notebook.add(frame, text="Data")
        return treeview

    def update_table(self, data: pd.DataFrame):
        """Updates the table with the new data.

        Args:
            data (pd.DataFrame): Dataframe containing all the data.
        """
        self.controller.progress("Updating table...")
        self.table.delete(*self.table.get_children())
        for i, row in data.round(decimals=2).iterrows():
            self.table.insert("", tk.END, values=tuple(row))

    def create_datasets_selector(self) -> SliceBoxes:
        """Creates the dataset selector, linking datasets with normalised names."""
        frame = ttk.LabelFrame(self, text="Current patient:")
        frame.grid(row=0, column=0, sticky=tk.NSEW, padx=10, pady=10)

        boxes = SliceBoxes(
            ttk.Combobox(frame, state="readonly"),
            ttk.Combobox(frame, state="readonly"),
            ttk.Combobox(frame, state="readonly"),
        )
        for i, (label, box) in enumerate(zip(SLICES, boxes)):
            ttk.Label(frame, text=f"{label}:").grid(row=0, column=2 * i, sticky=tk.NSEW)
            box.grid(row=0, column=2 * i + 1, sticky=tk.NSEW, padx=10)

        ttk.Checkbutton(
            frame,
            text="Overlay?",
            variable=self.overlay_data,
            command=self.toggle_overlay,
        ).grid(row=0, column=6, sticky=tk.NSEW, padx=10)

        self.populate_dataset_box()
        return boxes

    def populate_dataset_box(self):
        """Pupulate dataset boxes with data from the current patient, if available."""
        if not self.data or not self.data.strain_markers:
            return

        available = tuple(self.data.strain_markers.keys())
        self.dataset_box.Base.config(values=available)
        self.dataset_box.Base.current(0)
        self.dataset_box.Mid.config(values=available)
        self.dataset_box.Mid.current(int(len(available) / 2))
        self.dataset_box.Apex.config(values=available)
        self.dataset_box.Apex.current(len(available) - 1)

    def remove_record(self, *args):
        """Remove a complete record from the database."""
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
        self.update_plots()
        self.controller.progress(f"Record {record} removed!")

    def include_exclude_record(self, *args):
        """Include and exclude a record from the plots without deleting it."""
        if not self.table.selection():
            return
        i = self.table.index(self.table.selection()[0])
        record = self.atlas_data.loc[i, "Record"]
        self.atlas_data.loc[
            self.atlas_data.Record == record, "Included"
        ] = ~self.atlas_data.loc[self.atlas_data.Record == record, "Included"]

        self.save_atlas(self.path)
        self.update_table(self.atlas_data)
        self.update_plots()
        self.controller.progress(f"Record {record} inclusion/exclusion updated!")

    def include_exclude_selected(self, *args):
        """Include and exclude selected row from the plots without deleting it."""
        if not self.table.selection():
            return
        i = self.table.index(self.table.selection()[0])
        self.atlas_data.loc[i, "Included"] = not self.atlas_data.loc[i, "Included"]

        self.save_atlas(self.path)
        self.update_table(self.atlas_data)
        self.update_plots()
        self.controller.progress("Selection inclusion/exclusion updated!")

    def load_atlas(self, *args, path: Optional[Path] = None):
        """Loads atlas database and updates the plots and table.

        Eventually, the atlas will be read from a default location, so the dialog will
        not be needed.
        """
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
        self.update_plots()
        self.controller.progress(f"Atlas loaded from '{path}'.")

    def save_as_atlas(self, *args):
        """Save the atlas with a new name."""
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
        self.controller.progress(f"Atlas saved at '{filename}!")

    def save_atlas(self, filename: Path):
        """Save the atlas using the currently selected filename."""
        try:
            self.controller.progress("Saving atlas...")
            self.atlas_data.to_csv(filename, index=False)
            self.path = Path(filename)
        except ValueError as err:
            self.controller.progress(f"Error found when saving: {str(err)}.")

    def load_atlas_data(self, path: Optional[Path] = None) -> pd.DataFrame:
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
            data = validate_data(pd.read_csv(path))
            self.path = path

        except ValueError as err:
            self.controller.progress(str(err))
            data = empty_data()
            self.path = None

        return data

    def add_record(self, *args) -> None:
        """Adds new data to the atlas from an StrainMap file."""
        if not self.path:
            self.controller.progress(
                "No atlas file. Open an existing atlas or "
                "'Save atlas as' to create an empty one."
            )
            return

        data = self.get_new_data()
        if data is None:
            return

        self.atlas_data = self.atlas_data.append(data)

        self.save_atlas(self.path)
        self.update_table(self.atlas_data)
        self.update_plots()
        self.overlay(data)
        self.controller.progress("New record added to the atlas!")

    def get_new_data(self) -> Optional[pd.DataFrame]:
        """Get new data from the current patient and add it to the database."""
        from ..models.readers import extract_strain_markers

        if not self.data or not self.data.strain_markers or len(self.data.gls) == 0:
            self.controller.progress("No patient data available. Load data to proceed.")
            return None

        step = (
            slice(2, None, 1) if self.data.orientation == "CW" else slice(None, 1, -1)
        )
        data = extract_strain_markers(
            h5file=self.data.filename,
            datasets={k.get(): v for k, v in zip(self.dataset_box, SLICES)},
            regions={
                "global - Estimated": REGIONS[:1],
                "angular x6 - Estimated": REGIONS[step],
            },
        )
        data["Record"] = (
            self.atlas_data.Record.max() + 1 if len(self.atlas_data) > 0 else 1
        )
        data["Included"] = True
        gls = pd.DataFrame(
            {
                "Record": data["Record"].min(),
                "pssGLS": [self.data.gls[0]],
                "essGLS": [self.data.gls[1]],
                "psGLS": [self.data.gls[2]],
            }
        )
        data = data.append(gls, ignore_index=True)
        return validate_data(data[list(COLS)])

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
            row=row, column=col, sticky=tk.NSEW, padx=5, pady=5
        )

    return frame


class GridPlot:
    def __init__(self, label: str, data: pd.Dataframe):
        self.label = label
        self.fig, self.ax = plt.subplots(3, 3, sharey="col", tight_layout=True)
        self.update_plot(data)

    def update_plot(self, data: pd.DataFrame):
        """Update plot with data."""
        pad = 5

        for i, s in enumerate(SLICES):
            for j, c in enumerate(COMP):
                self.ax[i][j].clear()
                d = data.loc[
                    (data["Slice"] == s) & (data["Component"] == c) & data["Included"]
                ]
                d.boxplot(
                    column=[self.label],
                    by=["Region"],
                    ax=self.ax[i][j],
                    grid=False,
                    showfliers=False,
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

    def overlay(self, data: pd.DataFrame):
        """Overlay data onto the existing plots."""
        for i, s in enumerate(SLICES):
            for j, c in enumerate(COMP):
                d = data.loc[
                    (data["Slice"] == s) & (data["Component"] == c) & data["Included"]
                ]
                self.ax[i][j].plot(
                    [REGIONS.index(r) + 1 for r in strainmap.coordinates.Region],
                    d[self.label],
                    "ro",
                    alpha=0.6,
                )

        self.fig.canvas.draw_idle()


def validate_data(data: pd.DataFrame) -> pd.DataFrame:
    """Validates the atlas data, checking and setting column dtypes.

    Args:
        data (pd.DataFrame): Input DataFrame to validate.

    Raises:
        ValueError: If the columns are not the correct ones.

    Returns:
        The validated DataFrame
    """
    if tuple(data.columns) != tuple(COLS):
        raise ValueError(
            f"Invalid column names found. They should be, exactly: {COLS}."
        )

    slice_type = pd.CategoricalDtype(categories=SLICES, ordered=False)
    comp_type = pd.CategoricalDtype(categories=COMP, ordered=False)
    region_type = pd.CategoricalDtype(categories=REGIONS, ordered=False)
    col_types = (
        int,
        slice_type,
        float,
        region_type,
        comp_type,
        float,
        float,
        float,
        float,
        float,
        float,
        bool,
    )

    for col, col_type in zip(COLS, col_types):
        data[col] = data[col].astype(col_type)

    return data


def empty_data() -> pd.DataFrame:
    """Create empty dataframe with the required columns."""
    Record = pd.Series([], dtype=int)
    Slice = pd.Series([], dtype=str)
    SAX = pd.Series([], dtype=int)
    Component = pd.Series([], dtype=str)
    Region = pd.Series([], dtype=str)
    PSS = pd.Series([], dtype=float)
    ESS = pd.Series([], dtype=float)
    PS = pd.Series([], dtype=float)
    pssGLS = pd.Series([], dtype=float)
    essGLS = pd.Series([], dtype=float)
    psGLS = pd.Series([], dtype=float)
    Included = pd.Series([], dtype=bool)

    return validate_data(
        pd.DataFrame(
            {
                "Record": Record,
                "Slice": Slice,
                "SAX": SAX,
                "Region": Region,
                "Component": Component,
                "PSS": PSS,
                "ESS": ESS,
                "PS": PS,
                "pssGLS": pssGLS,
                "essGLS": essGLS,
                "psGLS": psGLS,
                "Included": Included,
            }
        )
    )
