from __future__ import annotations

import tkinter as tk
import tkinter.filedialog
from tkinter import messagebox, ttk
import os
from functools import partial
from pathlib import Path

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

        visualise = ttk.Frame(master=self, name="visualise")
        visualise.grid(column=0, row=0, sticky=tk.NSEW, padx=10, pady=10)
        visualise.columnconfigure(0, weight=1)
        visualise.rowconfigure(0, weight=1)
        visualise.grid_propagate(flag=False)

        self.notebook = ttk.Notebook(visualise, name="notebook")
        self.notebook.grid(sticky=tk.NSEW)
        self.notebook.columnconfigure(0, weight=1)
        self.notebook.rowconfigure(0, weight=1)

        self.atlas_data = self.load_atlas_data()
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

    def create_data_tab(self, data: pd.DataFrame):
        frame = ttk.Frame(self.notebook)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        treeview = ttk.Treeview(frame, selectmode="browse")
        vsb = ttk.Scrollbar(frame, orient="vertical", command=treeview.yview())
        treeview.configure(yscrollcommand=vsb.set)
        treeview.grid(column=0, row=0, sticky=tk.NSEW, padx=5, pady=5)
        vsb.grid(column=1, row=0, sticky=tk.NSEW)

        treeview["columns"] = tuple(data.columns)
        treeview["show"] = "headings"
        for v in treeview["columns"]:
            treeview.column(v, stretch=True)
            treeview.heading(v, text=v)

        for i, row in data.iterrows():
            treeview.insert("", tk.END, values=tuple(row))

        self.notebook.add(frame, text="Data")
        return treeview

    @staticmethod
    def load_atlas_data():
        """Loads the atlas data from a csv file.

        TODO: Placeholder. Replace with the real thing ASAP.
        """
        M = 200
        N = 9 * M
        Record = pd.Series(np.random.random_integers(1, 20, N))
        Slice = pd.Series(np.random.choice(["Base", "Middle", "Apex"], size=N)).astype(
            "category"
        )
        Component = pd.Series(
            np.random.choice(["Longitudinal", "Radial", "Circumferential"], size=N)
        ).astype("category")
        cat_type = pd.CategoricalDtype(
            categories=("Global", "", "AS", "A", "AL", "IL", "I", "IS"), ordered=False
        )
        Region = pd.Series(
            np.random.choice(["Global", "AS", "A", "AL", "IL", "I", "IS"], size=N)
        ).astype(cat_type)
        PSS = pd.Series(np.random.random(N))
        ESS = pd.Series(np.random.random(N))
        PS = pd.Series(np.random.random(N))
        Included = pd.Series([True] * N).astype(bool)

        return pd.DataFrame(
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

    def update_widgets(self):
        pass

    def clear_widgets(self):
        pass


class GridPlot:

    slices = ("Base", "Middle", "Apex")
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
                d = data.loc[(data["Slice"] == s) & (data["Component"] == c)]
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
