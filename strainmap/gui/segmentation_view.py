import tkinter as tk
from tkinter import ttk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .base_window_and_task import Requisites, TaskViewBase, register_view
from .figure_actions import (
    BrightnessAndContrast,
    DrawContours,
    ScrollFrames,
    ZoomAndPan,
)
from .figure_actions_manager import FigureActionsManager


@register_view
class SegmentationTaskView(TaskViewBase):

    requisites = Requisites.DATALOADED

    def __init__(self, root):

        super().__init__(root, button_text="Segmentation", button_image="molecules.gif")
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)

        # Control-related attributes
        self.control = None
        self.data_to_segment = None
        self.datasets_var = tk.StringVar(value="")

        # Visualization-related attributes
        self.visualise = None
        self.fig = None

        self.create_controls()

    def create_controls(self):
        """ Creates the control frame widgets. """
        self.control = ttk.Frame(master=self, name="control")
        self.control.grid(sticky=tk.NSEW, padx=10, pady=10)

        self.visualise = ttk.Frame(master=self, name="visualise")
        self.visualise.grid(sticky=tk.NSEW, padx=10, pady=10)
        self.visualise.columnconfigure(0, weight=1)
        self.visualise.rowconfigure(0, weight=1)

        dataset_frame = ttk.Labelframe(
            self.control, name="datasetsFrame", text="Datasets:"
        )
        dataset_frame.grid(row=0, column=0, sticky=tk.NSEW, padx=5, pady=5)
        dataset_frame.columnconfigure(0, weight=1)

        datasets_box = ttk.Combobox(
            master=dataset_frame,
            name="datasetsBox",
            textvariable=self.datasets_var,
            values=[],
            state="readonly",
        )
        datasets_box.bind("<<ComboboxSelected>>", self.update_plots)
        datasets_box.grid(column=0, sticky=tk.NSEW, padx=5, pady=5)

        segment_frame = ttk.Labelframe(
            self.control, name="segmentFrame", text="Initial segmentation:"
        )
        segment_frame.grid(row=0, column=1, sticky=tk.NSEW, padx=5, pady=5)
        segment_frame.columnconfigure(0, weight=1)

        ttk.Label(master=segment_frame, name="readyEndo", text="-").grid(
            row=0, column=0, sticky=tk.NSEW
        )

        ttk.Button(
            master=segment_frame,
            name="initialEndo",
            text="Define endocardium",
            command=self.define_initial_segment,
        ).grid(row=0, column=1, sticky=tk.NSEW)

        ttk.Label(master=segment_frame, name="readyEpi", text="-").grid(
            row=1, column=0, sticky=tk.NSEW
        )

        ttk.Button(
            master=segment_frame,
            name="initialEpi",
            text="Define epicardium",
            command=self.define_initial_segment,
        ).grid(row=1, column=1, sticky=tk.NSEW)

        ttk.Button(
            master=segment_frame,
            name="clearSegments",
            text="Clear segments",
            command=self.clear_initial_segments,
        ).grid(row=2, column=0, columnspan=2, sticky=tk.NSEW)

        ttk.Button(
            master=segment_frame,
            name="runSegmentation",
            text="Run segmentation",
            command=self.run_automatic_segmentation,
            state="disabled",
        ).grid(row=0, column=2, rowspan=3, sticky=tk.NSEW)

        self.create_plots()

    def create_plots(self):
        """ Creates the animation plot area. """

        self.fig = Figure()

        canvas = FigureCanvasTkAgg(self.fig, master=self.visualise)
        canvas.draw()
        canvas.get_tk_widget().grid(sticky=tk.NSEW, padx=5, pady=5)

        FigureActionsManager(
            self.fig, ZoomAndPan, BrightnessAndContrast, ScrollFrames, DrawContours
        )

        return

    def update_plots(self, *args):
        """Updates an existing plot when a new dataset is chosen."""
        self.data_to_segment = self.get_data_to_segment()

        self.fig.axes.clear()
        self.fig.actions_manager.ScrollFrames.clear()

        ax_mag = self.fig.add_subplot(1, 2, 1)
        ax_vel = self.fig.add_subplot(1, 2, 2, sharex=ax_mag, sharey=ax_mag)

        ax_mag.clear()
        ax_vel.clear()

        for img in self.data_to_segment[0]:
            ax_mag.imshow(img, cmap=plt.get_cmap("binary_r"))

        for img in self.data_to_segment[1]:
            ax_vel.imshow(img, cmap=plt.get_cmap("binary_r"))

        ax_mag.get_xaxis().set_visible(False)
        ax_mag.get_yaxis().set_visible(False)
        ax_vel.get_xaxis().set_visible(False)
        ax_vel.get_yaxis().set_visible(False)

        self.fig.set_tight_layout(True)
        self.fig.canvas.draw()

    def get_data_to_segment(self):
        """Gets the data that will be segmented and removes the phantom, if needed"""
        dataset = self.datasets_var.get()

        magz = np.array(self.data.get_images(dataset, "MagZ"))
        magx = np.array(self.data.get_images(dataset, "MagX"))
        magy = np.array(self.data.get_images(dataset, "MagY"))
        mag = magx + magy + magz
        vel = np.array(self.data.get_images(dataset, "PhaseZ"))

        if len(self.data.bg_files) > 0:
            magz = np.array(self.data.get_bg_images(dataset, "MagZ"))
            magx = np.array(self.data.get_bg_images(dataset, "MagX"))
            magy = np.array(self.data.get_bg_images(dataset, "MagY"))
            mag = mag - (magx + magy + magz)
            vel = vel - np.array(self.data.get_bg_images(dataset, "PhaseZ"))

        return mag, vel

    def define_initial_segment(self):
        """Defines an initial segment in the plot."""
        pass

    def clear_initial_segments(self):
        """Clears the initial segments."""
        pass

    def run_automatic_segmentation(self):
        """Runs an automatic segmentation sequence."""
        pass

    def update_widgets(self):
        """ Updates widgets after an update in the data variable. """
        values = sorted(self.data.data_files.keys())
        self.nametowidget("control.datasetsFrame.datasetsBox")["values"] = values
        self.datasets_var.set(values[0])
        self.update_plots()

    def clear_widgets(self):
        """ Clear widgets after removing the data. """
        pass
