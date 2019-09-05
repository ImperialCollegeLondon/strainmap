import tkinter as tk
from tkinter import ttk
from copy import copy
from collections import defaultdict

from typing import Optional, Dict
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import Cursor
from matplotlib.figure import Figure

from .base_window_and_task import Requisites, TaskViewBase, register_view, trigger_event
from .figure_actions import (
    BrightnessAndContrast,
    DrawContours,
    ScrollFrames,
    ZoomAndPan,
    DragContours,
)
from .figure_actions_manager import FigureActionsManager
from ..models.contour_mask import contour_diff


@register_view
class SegmentationTaskView(TaskViewBase):

    requisites = Requisites.DATALOADED

    def __init__(self, root):

        super().__init__(root, button_text="Segmentation", button_image="molecules.gif")
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)

        # Control-related attributes
        self.current_frame = 0
        self.images: Dict[str, Optional[np.ndarray]] = {"mag": None, "vel": None}
        self.initial_segments: Dict[str, Optional[np.ndarray]] = {
            "endocardium": None,
            "epicardium": None,
        }
        self.final_segments: Dict[str, Optional[np.ndarray]] = {
            "endocardium": None,
            "epicardium": None,
        }
        self.endocardium_target_var = tk.StringVar(value="mag")
        self.epicardium_target_var = tk.StringVar(value="mag")
        self.datasets_var = tk.StringVar(value="")
        self.endo_redy_var = tk.StringVar(value="\u274C")
        self.epi_redy_var = tk.StringVar(value="\u274C")
        self.drag_width_scale = tk.IntVar(value=10)
        self.drag_width_label = tk.StringVar(value="10%")
        self.undo_stack = defaultdict(list)
        self.working_frame_var = tk.IntVar(value=0)

        # Visualization-related attributes
        self.fig = None
        self.ax_mag = None
        self.ax_vel = None
        self.cursors: Dict[str, Optional[Cursor]] = {"mag": None, "vel": None}
        self.datasets_box = None
        self.clear_btn = None
        self.ini_endo_btn = None
        self.ini_epi_btn = None
        self.segment_btn = None
        self.undo_last_btn = None
        self.undo_all_btn = None
        self.next_btn = None
        self.pbar = None

        self.create_controls()

    def create_controls(self):
        """ Creates all the widgets of the view. """
        # Top frames
        control = ttk.Frame(master=self)
        visualise_frame = ttk.Frame(master=self)
        visualise_frame.columnconfigure(0, weight=1)
        visualise_frame.rowconfigure(1, weight=1)

        # Dataset frame
        dataset_frame = ttk.Labelframe(control, text="Datasets:")
        dataset_frame.columnconfigure(0, weight=1)
        dataset_frame.rowconfigure(0, weight=1)
        dataset_frame.rowconfigure(1, weight=1)

        self.datasets_box = ttk.Combobox(
            master=dataset_frame,
            textvariable=self.datasets_var,
            values=[],
            state="readonly",
        )
        self.datasets_box.bind("<<ComboboxSelected>>", self.dataset_changed)

        self.clear_btn = ttk.Button(
            master=dataset_frame,
            text="Clear existing segmentation",
            state="disabled",
            width=20,
            command=self.clear_segment_variables,
        )

        # Automatic segmentation frame
        segment_frame = ttk.Labelframe(control, text="Initial segmentation:")
        segment_frame.columnconfigure(0, weight=1)
        segment_frame.rowconfigure(0, weight=1)
        segment_frame.rowconfigure(1, weight=1)

        endo_redy_lbl = ttk.Label(master=segment_frame, textvariable=self.endo_redy_var)
        epi_redy_lbl = ttk.Label(master=segment_frame, textvariable=self.epi_redy_var)

        self.ini_endo_btn = ttk.Button(
            master=segment_frame,
            text="Define endocardium",
            command=lambda: self.initial_contour("endocardium"),
        )

        self.ini_epi_btn = ttk.Button(
            master=segment_frame,
            text="Define epicardium",
            width=15,
            command=lambda: self.initial_contour("epicardium"),
        )

        for i, text in enumerate(["mag", "vel"]):
            ttk.Radiobutton(
                master=segment_frame,
                text=text,
                variable=self.endocardium_target_var,
                value=text,
            ).grid(row=0, column=2 + i, sticky=tk.NSEW)
            ttk.Radiobutton(
                master=segment_frame,
                text=text,
                variable=self.epicardium_target_var,
                value=text,
            ).grid(row=1, column=2 + i, sticky=tk.NSEW)

        # Manual segmentation frame
        manual_frame = ttk.Labelframe(control, text="Manual segmentation:")
        manual_frame.columnconfigure(0, weight=1)

        drag_lbl = ttk.Label(manual_frame, text="Drag width: ")
        width_lbl = ttk.Label(manual_frame, textvariable=self.drag_width_label, width=5)

        scale = ttk.Scale(
            manual_frame,
            orient=tk.HORIZONTAL,
            from_=5,
            to=35,
            variable=self.drag_width_scale,
            command=self.update_drag_width,
        )

        self.undo_last_btn = ttk.Button(
            master=manual_frame,
            text="Undo last change",
            state="disabled",
            command=lambda: self.undo(-1),
        )

        self.undo_all_btn = ttk.Button(
            master=manual_frame,
            text="Undo all in frame",
            state="disabled",
            command=lambda: self.undo(0),
        )

        # Progress frame
        progress_frame = ttk.Frame(visualise_frame)
        progress_frame.columnconfigure(3, weight=1)

        progress_lbl = ttk.Label(progress_frame, text="Working frame: ")
        frame_btn = ttk.Button(
            progress_frame,
            textvariable=self.working_frame_var,
            width=5,
            command=self.go_to_frame,
        )

        self.next_btn = ttk.Button(
            progress_frame,
            text="Start",
            width=5,
            state="disabled",
            command=self.find_segmentation,
        )

        self.pbar = ttk.Progressbar(
            master=progress_frame,
            orient=tk.HORIZONTAL,
            mode="determinate",
            variable=self.working_frame_var,
            maximum=49.0,
        )

        control.grid(sticky=tk.NSEW, padx=10, pady=10)
        visualise_frame.grid(sticky=tk.NSEW, padx=10)
        dataset_frame.grid(row=0, column=0, sticky=tk.NSEW, padx=5, pady=5)
        self.datasets_box.grid(row=0, column=0, sticky=tk.NSEW)
        self.clear_btn.grid(row=1, column=0, sticky=tk.NSEW)
        segment_frame.grid(row=0, column=1, sticky=tk.NSEW, padx=5, pady=5)
        endo_redy_lbl.grid(row=0, column=0, sticky=tk.NSEW)
        epi_redy_lbl.grid(row=1, column=0, sticky=tk.NSEW)
        self.ini_endo_btn.grid(row=0, column=1, sticky=tk.NSEW)
        self.ini_epi_btn.grid(row=1, column=1, sticky=tk.NSEW)
        manual_frame.grid(row=0, column=3, sticky=tk.NSEW, padx=5, pady=5)
        drag_lbl.grid(row=0, sticky=tk.NSEW, padx=5, pady=5)
        width_lbl.grid(row=0, column=1, sticky=tk.E, padx=5, pady=5)
        scale.grid(row=1, column=0, columnspan=2, sticky=tk.NSEW, padx=5, pady=5)
        self.undo_last_btn.grid(row=0, column=2, sticky=tk.NSEW)
        self.undo_all_btn.grid(row=1, column=2, sticky=tk.NSEW)
        progress_frame.grid(
            row=0, column=0, columnspan=4, sticky=tk.NSEW, padx=5, pady=5
        )
        progress_lbl.grid(row=0, column=0, sticky=tk.NSEW, padx=5, pady=5)
        frame_btn.grid(row=0, column=1, sticky=tk.NSEW, padx=5, pady=5)
        self.next_btn.grid(row=0, column=2, sticky=tk.NSEW, padx=5, pady=5)
        self.pbar.grid(row=0, column=3, sticky=tk.NSEW)

        self.create_plots(visualise_frame)
        self.update_drag_width()

    def create_plots(self, visualise_frame):
        """ Creates the animation plot area. """

        self.fig = Figure()
        self.ax_mag = self.fig.add_subplot(1, 2, 1)
        self.ax_vel = self.fig.add_subplot(
            1, 2, 2, sharex=self.ax_mag, sharey=self.ax_mag
        )
        self.ax_mag.set_position((0.03, 0.05, 0.45, 0.85))
        self.ax_vel.set_position((0.52, 0.05, 0.45, 0.85))

        self.ax_mag.get_xaxis().set_visible(False)
        self.ax_mag.get_yaxis().set_visible(False)
        self.ax_vel.get_xaxis().set_visible(False)
        self.ax_vel.get_yaxis().set_visible(False)

        canvas = FigureCanvasTkAgg(self.fig, master=visualise_frame)
        canvas.draw()
        canvas.get_tk_widget().grid(sticky=tk.NSEW)

        self.fig.actions_manager = FigureActionsManager(
            self.fig,
            ZoomAndPan,
            BrightnessAndContrast,
            ScrollFrames,
            DrawContours,
            DragContours,
        )
        self.fig.actions_manager.DrawContours.num_contours = 0
        self.fig.actions_manager.ScrollFrames.link_axes(self.ax_mag, self.ax_vel)
        self.fig.actions_manager.DragContours.set_contour_updated(self.contour_edited)

        self.ax_mag.set_title("Magnitude", loc="right")
        self.ax_vel.set_title("Velocity", loc="right")

        self.fig.actions_manager.ScrollFrames.set_scroller(
            lambda frame, axes="mag": self.scroll(frame, axes), self.ax_mag
        )
        self.fig.actions_manager.ScrollFrames.set_scroller(
            lambda frame, axes="vel": self.scroll(frame, axes), self.ax_vel
        )

    def dataset_changed(self, *args):
        """Updates the GUI when a new dataset is chosen."""
        dataset = self.datasets_var.get()
        self.update_state(dataset)
        self.fig.actions_manager.ScrollFrames.clear()

        xlim = ylim = None
        if len(self.ax_mag.images) > 0:
            xlim = self.ax_mag.get_xlim()
            ylim = self.ax_mag.get_ylim()

        self.ax_mag.images.clear()
        self.ax_vel.images.clear()
        self.ax_mag.lines.clear()
        self.ax_vel.lines.clear()

        self.plot_images(dataset)
        self.plot_segments(dataset)

        if xlim is not None:
            self.ax_mag.set_xlim(*xlim)
            self.ax_mag.set_ylim(*ylim)

        self.fig.canvas.draw()

    def plot_images(self, dataset):
        """Updates the images in the plot."""
        self.images = self.get_data_to_segment(dataset)
        self.pbar.config(maximum=self.images["mag"].shape[0] - 1)

        self.ax_mag.imshow(
            self.images["mag"][self.current_frame], cmap=plt.get_cmap("binary_r")
        )
        self.ax_vel.imshow(
            self.images["vel"][self.current_frame], cmap=plt.get_cmap("binary_r")
        )

    def plot_segments(self, dataset):
        """Updates the segments in the plot, if any."""
        if len(self.data.segments[dataset]) == 0:
            return

        colors = ("tab:blue", "tab:orange")
        for i, side in enumerate(["endocardium", "epicardium"]):
            self.final_segments[side] = self.data.segments[dataset][side]
            self.ax_mag.plot(
                *self.final_segments[side][self.current_frame],
                dashes=(5, 5),
                picker=6,
                label=side,
                color=colors[i],
            )
            self.ax_vel.plot(
                *self.final_segments[side][self.current_frame],
                dashes=(5, 5),
                picker=6,
                label=side,
                color=colors[i],
            )

        self.plot_centroid()

    def plot_centroid(self):
        """Plots the centroid of the mask defined by the current segments."""
        options = dict(marker="+", color="r", markersize=10, label="centroid")
        centroid = self.current_centroid
        self.ax_mag.plot(*centroid, **options)
        self.ax_vel.plot(*centroid, **options)

    @property
    def current_centroid(self):
        """Return an array with the position of the centroid at a given time."""
        mask = contour_diff(
            self.final_segments["epicardium"][self.current_frame].T,
            self.final_segments["endocardium"][self.current_frame].T,
            shape=self.images["mag"][self.current_frame].shape,
        )
        return np.array(ndimage.measurements.center_of_mass(mask))

    def update_state(self, dataset):
        """Updates the state of buttons and vars when something happens in the GUI."""
        self.undo_stack = defaultdict(list)
        self.update_undo_state()
        if len(self.data.segments[dataset]) == 0:
            self.clear_segment_variables(button_pressed=False)
            self.update_initial_segmentation_state(True)
            self.clear_btn.state(["disabled"])
        else:
            self.update_initial_segmentation_state(False)
            self.clear_btn.state(["!disabled"])

    def go_to_frame(self, *args):
        """Manually sets the current frame and re-draws the GUI."""
        frame = self.working_frame_var.get()
        self.fig.actions_manager.ScrollFrames.go_to_frame(frame, self.fig.axes[0])
        self.fig.canvas.draw_idle()

    def next_frame(self):
        """Confirm segmentation of current frame, moves to the next and segment it."""
        self.update_segmentation()

        frame = self.working_frame_var.get() + 1
        self.working_frame_var.set(frame)
        self.current_frame = frame
        self.find_segmentation()

        if frame + 1 == self.images["mag"].shape[0]:
            self.next_btn.config(text="Finish", command=self.finish_segmentation)

        self.go_to_frame()

    def scroll(self, frame, image=None):
        """Provides the next images and lines to plot when scrolling."""
        self.current_frame = frame % self.images["mag"].shape[0]

        img = endo = epi = centroid = None

        if image in ["mag", "vel"]:
            img = self.images[image][self.current_frame]
        if self.final_segments["endocardium"] is not None:
            endo = self.final_segments["endocardium"][self.current_frame]
            epi = self.final_segments["epicardium"][self.current_frame]
            if np.isnan(endo[0, 0]).any():
                centroid = np.full(2, np.nan)
            else:
                centroid = self.current_centroid

        self.update_undo_state()

        self.fig.actions_manager.DragContours.disabled = (
            self.current_frame < self.working_frame_var.get()
            or self.next_btn["text"] == "Done!"
        )

        return self.current_frame, img, (endo, epi, centroid)

    def contour_edited(self, label, axes, data):
        """After a contour is modified, this function is executed."""

        self.undo_stack[self.current_frame].append(
            dict(label=label, data=copy(self.final_segments[label][self.current_frame]))
        )
        self.update_undo_state()

        self.final_segments[label][self.current_frame] = data

        for ax in set(self.fig.axes):
            for l in ax.lines:
                if l.get_label() == label:
                    l.set_data(data)
                elif l.get_label() == "centroid":
                    l.set_data(self.current_centroid)

    def undo(self, index):
        """Undo the last manual modification in the current frame."""
        last = self.undo_stack[self.current_frame].pop(index)
        self.final_segments[last["label"]][self.current_frame] = last["data"]

        for ax in self.fig.axes:
            for l in ax.lines:
                if l.get_label() == last["label"]:
                    l.set_data(last["data"])

        self.fig.canvas.draw()

        if index == 0:
            self.undo_stack[self.current_frame] = []

        self.update_undo_state()

    def update_undo_state(self):
        """Updates the state of the undo buttons."""
        if len(self.undo_stack[self.current_frame]) == 0:
            self.undo_stack.pop(self.current_frame)
            self.undo_last_btn.state(["disabled"])
            self.undo_all_btn.state(["disabled"])
        else:
            self.undo_last_btn.state(["!disabled"])
            self.undo_all_btn.state(["!disabled"])

    def update_initial_segmentation_state(self, enable=True):
        """Change the state of the segmentation buttons."""
        state = "!disabled" if enable else "disabled"
        self.ini_endo_btn.state([state])
        self.ini_epi_btn.state([state])

    def update_drag_width(self, *args):
        """Updates the dragging width."""
        width = self.drag_width_scale.get()
        self.drag_width_label.set(f"{width}%")
        self.fig.actions_manager.DragContours.contour_fraction = width / 100

    def get_data_to_segment(self, dataset):
        """Gets the data that will be segmented."""

        magz = self.data.get_images(dataset, "MagZ")
        magx = self.data.get_images(dataset, "MagX")
        magy = self.data.get_images(dataset, "MagY")
        mag = magx + magy + magz
        vel = self.data.get_images(dataset, "PhaseZ")

        return {"mag": mag, "vel": vel}

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
        if side == "endocardium":
            self.ini_endo_btn.config(text=new_text)
        else:
            self.ini_epi_btn.config(text=new_text)

    def switch_mark_state(self, side, state):
        """Switch the text displayed in the initial segmentation buttons."""
        mark = {"ready": "\u2705", "not_ready": "\u274C"}[state]
        if side == "endocardium":
            self.endo_redy_var.set(mark)
        else:
            self.epi_redy_var.set(mark)

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
        """Enables the definition of the initial segment for the side."""
        self.working_frame_var.set(0)
        self.go_to_frame()
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
            self.next_btn.state(["!disabled"])

    def clear_segment_variables(self, button_pressed=True):
        """Clears all segmentation when a dataset with no segmentation is loaded."""
        self.initial_segments = {"endocardium": None, "epicardium": None}
        self.final_segments = {"endocardium": None, "epicardium": None}

        for side in ["endocardium", "epicardium"]:
            self.switch_button_text(side, f"Define {side}")
            self.switch_mark_state(side, "not_ready")

        self.undo_stack = defaultdict(list)
        self.working_frame_var.set(0)
        self.current_frame = 0
        self.next_btn.state(["disabled"])
        self.next_btn.config(text="Start", command=self.find_segmentation)
        self.fig.actions_manager.DragContours.disabled = False
        if button_pressed:
            self.update_segmentation()

    def clear_segments(self, side="both", initial_or_final="both"):
        """Clears the segments."""
        self.ax_mag.lines.clear()
        self.ax_vel.lines.clear()

        if initial_or_final in ["initial", "both"]:
            if side == "endocardium":
                self.initial_segments["endocardium"] = None
            elif side == "epicardium":
                self.initial_segments["epicardium"] = None
            else:
                self.initial_segments = {"endocardium": None, "epicardium": None}

            self.plot_initial_segments()

        if initial_or_final in ["final", "both"]:
            if side == "endocardium":
                self.final_segments["endocardium"] = None
            elif side == "epicardium":
                self.final_segments["epicardium"] = None
            else:
                self.final_segments = {"endocardium": None, "epicardium": None}

            self.plot_segments(self.datasets_var.get())

        self.fig.canvas.draw()

    def plot_initial_segments(self):
        """Plots the initial segments."""
        style = dict(color="yellow", ls="--", linewidth=1)
        for side in self.initial_segments:
            if self.initial_segments[side] is not None:
                self.ax_mag.plot(*self.initial_segments[side], **style)
                self.ax_vel.plot(*self.initial_segments[side], **style)

    def finish_segmentation(self):
        """Finish the segmentation, updating values and state of buttons."""
        self.update_segmentation(unlock=True)
        self.next_btn.config(text="Done!")
        self.next_btn.state(["disabled"])
        self.fig.actions_manager.DragContours.disabled = True

    @trigger_event
    def find_segmentation(self):
        """Runs an automatic segmentation sequence."""
        frame = self.working_frame_var.get()

        if frame == 0:
            initial = self.initial_segments
            self.next_btn.config(text="Next \u25B6", command=self.next_frame)
        else:
            initial = {
                "endocardium": self.final_segments["endocardium"][frame - 1],
                "epicardium": self.final_segments["epicardium"][frame - 1],
            }

        images = {
            "endocardium": self.images[self.endocardium_target_var.get()][frame],
            "epicardium": self.images[self.epicardium_target_var.get()][frame],
        }

        return dict(
            data=self.data,
            dataset_name=self.datasets_var.get(),
            frame=frame,
            images=images,
            initials=initial,
            unlock=False,
        )

    @trigger_event
    def update_segmentation(self, unlock=False):
        """Confirm the new segments after a manual segmentation process."""
        return dict(
            data=self.data,
            dataset_name=self.datasets_var.get(),
            segments=copy(self.final_segments),
            frame=self.working_frame_var.get(),
            unlock=unlock,
        )

    def stop_animation(self):
        """Stops an animation, if there is one running."""
        self.fig.actions_manager.ScrollFrames.stop_animation()

    def update_widgets(self):
        """ Updates widgets after an update in the data variable. """
        values = list(self.data.data_files.keys())
        current = self.datasets_var.get()
        self.datasets_box.config(values=values)
        if current in values:
            self.datasets_var.set(current)
        else:
            self.datasets_var.set(values[0])

        self.dataset_changed()

    def clear_widgets(self):
        """ Clear widgets after removing the data. """
        pass
