import tkinter as tk
from tkinter import ttk

from typing import Optional, Dict
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import Cursor
from matplotlib.figure import Figure

from .base_window_and_task import Requisites, TaskViewBase, register_view, trigger_event
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
        self.initial_segments = {"endocardium": None, "epicardium": None}
        self.final_segments = {"endocardium": None, "epicardium": None}
        self.endocardium_target_var = tk.StringVar(value="mag")
        self.epicardium_target_var = tk.StringVar(value="mag")
        self.datasets_var = tk.StringVar(value="")
        self.phantom_var = tk.StringVar(value="")

        # Visualization-related attributes
        self.visualise = None
        self.fig = None
        self.ax_mag = None
        self.ax_vel = None
        self.cursors: Dict[str, Optional[Cursor]] = {"mag": None, "vel": None}

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

        ttk.Label(master=dataset_frame, text="Data").grid(
            row=0, column=0, sticky=tk.NSEW
        )

        datasets_box = ttk.Combobox(
            master=dataset_frame,
            name="datasetsBox",
            textvariable=self.datasets_var,
            values=[],
            state="readonly",
        )
        datasets_box.bind("<<ComboboxSelected>>", self.update_plots)
        datasets_box.grid(row=0, column=1, sticky=tk.NSEW)

        ttk.Label(master=dataset_frame, text="Phantom").grid(
            row=1, column=0, sticky=tk.NSEW
        )

        phantom_box = ttk.Combobox(
            master=dataset_frame,
            name="phantomBox",
            textvariable=self.phantom_var,
            values=[],
            state="disabled",
        )
        phantom_box.bind("<<ComboboxSelected>>", self.update_plots)
        phantom_box.grid(row=1, column=1, sticky=tk.NSEW)

        segment_frame = ttk.Labelframe(
            self.control, name="segmentFrame", text="Initial segmentation:"
        )
        segment_frame.grid(row=0, column=1, sticky=tk.NSEW, padx=5, pady=5)
        segment_frame.columnconfigure(0, weight=1)

        ttk.Label(master=segment_frame, name="readyEndo", text="\u274C").grid(
            row=0, column=0, sticky=tk.NSEW
        )

        ttk.Button(
            master=segment_frame,
            name="initialEndo",
            text="Define endocardium",
            command=lambda: self.initial_contour("endocardium"),
        ).grid(row=0, column=1, sticky=tk.NSEW)

        for i, text in enumerate(["mag", "vel"]):
            ttk.Radiobutton(
                master=segment_frame,
                text=text,
                variable=self.endocardium_target_var,
                value=text,
            ).grid(row=0, column=2 + i, sticky=tk.NSEW)

        ttk.Label(master=segment_frame, name="readyEpi", text="\u274C").grid(
            row=1, column=0, sticky=tk.NSEW
        )

        ttk.Button(
            master=segment_frame,
            name="initialEpi",
            text="Define epicardium",
            command=lambda: self.initial_contour("epicardium"),
        ).grid(row=1, column=1, sticky=tk.NSEW)

        for i, text in enumerate(["mag", "vel"]):
            ttk.Radiobutton(
                master=segment_frame,
                text=text,
                variable=self.epicardium_target_var,
                value=text,
            ).grid(row=1, column=2 + i, sticky=tk.NSEW)

        ttk.Button(
            master=self.control,
            name="runSegmentation",
            text="Find segmentation",
            command=self.find_segmentation,
            state="disabled",
        ).grid(row=0, column=2, sticky=tk.NSEW)

        self.create_plots()

    def create_plots(self):
        """ Creates the animation plot area. """

        self.fig = Figure()
        self.ax_mag = self.fig.add_subplot(1, 2, 1)
        self.ax_vel = self.fig.add_subplot(
            1, 2, 2, sharex=self.ax_mag, sharey=self.ax_mag
        )

        self.ax_mag.get_xaxis().set_visible(False)
        self.ax_mag.get_yaxis().set_visible(False)
        self.ax_vel.get_xaxis().set_visible(False)
        self.ax_vel.get_yaxis().set_visible(False)

        self.fig.set_tight_layout(True)

        canvas = FigureCanvasTkAgg(self.fig, master=self.visualise)
        canvas.draw()
        canvas.get_tk_widget().grid(sticky=tk.NSEW)

        FigureActionsManager(
            self.fig, ZoomAndPan, BrightnessAndContrast, ScrollFrames, DrawContours
        )
        self.fig.actions_manager.DrawContours.num_contours = 0

        return

    def update_plots(self, *args):
        """Updates an existing plot when a new dataset is chosen."""
        self.fig.actions_manager.ScrollFrames.clear()

        xlim = ylim = None
        if len(self.ax_mag.images) > 0:
            xlim = self.ax_mag.get_xlim()
            ylim = self.ax_mag.get_ylim()

        self.ax_mag.clear()
        self.ax_vel.clear()

        self.update_images()
        self.plot_segments()

        if xlim is not None:
            self.ax_mag.set_xlim(*xlim)
            self.ax_mag.set_ylim(*ylim)

        self.fig.canvas.draw()

    def update_images(self):
        """Updates the images in the plot."""
        data_to_segment = self.get_data_to_segment()

        for img in data_to_segment[0]:
            self.ax_mag.imshow(img, cmap=plt.get_cmap("binary_r"))

        for img in data_to_segment[1]:
            self.ax_vel.imshow(img, cmap=plt.get_cmap("binary_r"))

    def plot_segments(self):
        """Updates the segments - if any - in the plot."""
        if len(self.data.segments) == 0:
            return

        num = len(self.data.segments["endocardium"])
        for i in range(num):
            endo = self.data.segments["endocardium"][i].xy
            epi = self.data.segments["epicardium"][i].xy
            self.ax_mag.plot(endo[:, 0], endo[:, 1])
            self.ax_vel.plot(endo[:, 0], endo[:, 1])
            self.ax_mag.plot(epi[:, 0], epi[:, 1])
            self.ax_vel.plot(epi[:, 0], epi[:, 1])

    def get_data_to_segment(self):
        """Gets the data that will be segmented and removes the phantom, if needed"""
        dataset = self.datasets_var.get()

        magz = np.array(self.data.get_images(dataset, "MagZ"))
        magx = np.array(self.data.get_images(dataset, "MagX"))
        magy = np.array(self.data.get_images(dataset, "MagY"))
        mag = magx + magy + magz
        vel = np.array(self.data.get_images(dataset, "PhaseZ"))

        if len(self.data.bg_files) > 0:
            dataset = self.phantom_var.get()
            magz = np.array(self.data.get_bg_images(dataset, "MagZ"))
            magx = np.array(self.data.get_bg_images(dataset, "MagX"))
            magy = np.array(self.data.get_bg_images(dataset, "MagY"))
            mag = mag - (magx + magy + magz)
            vel = vel - np.array(self.data.get_bg_images(dataset, "PhaseZ"))

        return mag, vel

    def initial_contour(self, side):
        """Starts, interrupts or clears the definition of an initial contour."""
        if self.cursors["mag"] is not None:
            self.leave_initial_edit_mode()
            self.fig.suptitle("")
            self.switch_button_text(side, f"Define {side}")
            self.switch_mark_state(side, "not_ready")
        elif self.initial_segments[side] is not None:
            self.clear_segments(side=side, initial_or_final="initial")
            self.switch_button_text(side, f"Define {side}")
            self.switch_mark_state(side, "not_ready")
        else:
            self.define_initial_contour(side)

    def switch_button_text(self, side, new_text):
        """Switch the text displayed in the initial segmentation buttons."""
        widget_name = {
            "endocardium": "control.segmentFrame.initialEndo",
            "epicardium": "control.segmentFrame.initialEpi",
        }[side]

        self.nametowidget(widget_name)["text"] = new_text

    def switch_mark_state(self, side, state):
        """Switch the text displayed in the initial segmentation buttons."""
        widget_name = {
            "endocardium": "control.segmentFrame.readyEndo",
            "epicardium": "control.segmentFrame.readyEpi",
        }[side]

        mark = {"ready": "\u2705", "not_ready": "\u274C"}[state]

        self.nametowidget(widget_name)["text"] = mark

    def enter_initial_edit_mode(self):
        """Enters the edit mode for defining the initial segments."""
        self.fig.actions_manager.DrawContours.num_contours = 1

        cursor_properties = dict(useblit=True, ls="--", color="red", linewidth=1)
        self.cursors["mag"] = Cursor(self.fig.axes[0], **cursor_properties)
        self.cursors["vel"] = Cursor(self.fig.axes[1], **cursor_properties)

    def leave_initial_edit_mode(self):
        """Leaves the edit mode for defining the initial segments."""
        self.cursors["mag"] = None
        self.cursors["vel"] = None

        for ax in self.fig.axes:
            self.fig.actions_manager.DrawContours.clear_drawing_(ax)

        self.fig.actions_manager.DrawContours.num_contours = 0
        self.fig.canvas.draw()

    def define_initial_contour(self, side):
        """Enables the definition of the initial segment for the endocardium."""
        self.enter_initial_edit_mode()
        self.switch_button_text(side, "Cancel")
        self.fig.suptitle(
            "Left click twice to define the center and the edge of the "
            f"initial segment for the {side.upper()}."
        )

        get_contour = partial(self.get_contour, side=side)
        self.fig.actions_manager.DrawContours.contours_updated = get_contour

    def get_contour(self, contour, *args, side):
        """Plots the just defined contour in both axes and leaves the edit mode."""
        if len(contour) == 1:
            self.initial_segments[side] = contour[0]
            self.plot_initial_segments()
            self.fig.suptitle("")
            self.switch_button_text(side, f"Clear {side}")
            self.switch_mark_state(side, "ready")
            self.segmentation_ready()

        self.leave_initial_edit_mode()

    def segmentation_ready(self):
        """Checks if all is ready to start a segmentation enabling the widgets if so."""
        if (
            self.initial_segments["endocardium"] is not None
            and self.initial_segments["epicardium"] is not None
        ):
            self.nametowidget("control.runSegmentation")["state"] = "enable"

    def clear_segments(self, side="both", initial_or_final="both"):
        """Clears the initial segments."""
        self.ax_mag.lines.clear()
        self.ax_vel.lines.clear()

        if initial_or_final in ["initial", "both"]:
            if side == "endocardium":
                self.initial_segments["endocardium"] = None
            elif side == "epicardium":
                self.initial_segments["epicardium"] = None
            else:
                self.initial_segments = {"endocardium": None, "epicardium": None}

            self.nametowidget("control.runSegmentation")["state"] = "disabled"

        if initial_or_final in ["final", "both"]:
            if side == "endocardium":
                self.final_segments["endocardium"] = None
            elif side == "epicardium":
                self.final_segments["epicardium"] = None
            else:
                self.final_segments = {"endocardium": None, "epicardium": None}

        self.plot_initial_segments()
        self.plot_segments()
        self.fig.canvas.draw()

    def plot_initial_segments(self):
        """PLots the initial segments."""
        style = dict(color="yellow", ls="--", linewidth=1)
        for side in self.initial_segments:
            if self.initial_segments[side] is not None:
                self.ax_mag.plot(*self.initial_segments[side], **style)
                self.ax_vel.plot(*self.initial_segments[side], **style)

    @trigger_event
    def find_segmentation(self):
        """Runs an automatic segmentation sequence."""
        self.clear_segments(side="both", initial_or_final="final")
        return dict(
            data=self.data,
            dataset=self.datasets_var.get(),
            phantom_dataset=self.phantom_var.get(),
            endo_target=self.endocardium_target_var.get(),
            epi_target=self.epicardium_target_var.get(),
            endo_initial=self.initial_segments["endocardium"],
            epi_initial=self.initial_segments["epicardium"],
        )

    def update_widgets(self):
        """ Updates widgets after an update in the data variable. """
        values = sorted(self.data.data_files.keys())
        self.nametowidget("control.datasetsFrame.datasetsBox")["values"] = values
        self.datasets_var.set(values[0])

        if len(self.data.bg_files.keys()) > 0:
            bg_values = sorted(self.data.bg_files.keys())
            self.nametowidget("control.datasetsFrame.phantomBox")["values"] = bg_values
            self.nametowidget("control.datasetsFrame.phantomBox")["state"] = "enable"
            self.phantom_var.set(bg_values[0])
        else:
            self.nametowidget("control.datasetsFrame.phantomBox")["values"] = []
            self.nametowidget("control.datasetsFrame.phantomBox")["state"] = "disabled"
            self.phantom_var.set("")

        self.update_plots()

    def clear_widgets(self):
        """ Clear widgets after removing the data. """
        pass
