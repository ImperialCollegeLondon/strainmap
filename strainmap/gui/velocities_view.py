import tkinter as tk
from tkinter import ttk
import tkinter.filedialog
import re

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from .base_window_and_task import Requisites, TaskViewBase, register_view
from .figure_actions_manager import FigureActionsManager
from .figure_actions import Markers, SimpleScroller


def get_sa_location(dataset):
    pattern = r"[sS][aA]([0-9])"
    m = re.search(pattern, dataset)
    return int(m.group(1)) if hasattr(m, "group") else 99


@register_view
class VelocitiesTaskView(TaskViewBase):

    requisites = Requisites.SEGMENTED
    axes_lbl = ("_long", "_rad", "_circ")
    marker_idx = {"PS": 0, "PD": 1, "PAS": 2, "PC1": 0, "PC2": 1, "PC3": 2, "ES": 3}

    def __init__(self, root, controller):

        super().__init__(
            root,
            controller,
            button_text="Velocities",
            button_image="speed.gif",
            button_row=2,
        )
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)

        self.visualise_frame = None
        self.datasets_box = None
        self.datasets_var = tk.StringVar(value="")
        self.velocities_frame = None
        self.velocities_var = tk.StringVar(value="")
        self.bg_box = None
        self.bg_var = tk.StringVar(value="Estimated")
        self.plot = None
        self.regional_fig = None
        self.param_tables = []
        self.current_region = 0
        self.images = None
        self.update_vel_btn = None
        self.reverse_vel_var = (
            tk.BooleanVar(value=False),
            tk.BooleanVar(value=False),
            tk.BooleanVar(value=False),
        )
        self.reverse_status = (False, False, False)
        self.clockwise_var = tk.BooleanVar(value=False)

        # Figure-related variables
        self.fig = None
        self.axes = None
        self.maps = None
        self.vel_lines = None
        self.bg_images = None
        self.vel_masks = None
        self.cbar = None
        self.limits = None
        self.marker_artists = None
        self.vel_lim = dict()
        self.fixed_markers = []

        self.create_controls()

    def create_controls(self):
        """ Creates all the widgets of the view. """
        # Top frames
        control = ttk.Frame(master=self)
        control.columnconfigure(49, weight=1)
        self.visualise_frame = ttk.Frame(master=self)
        self.visualise_frame.columnconfigure(0, weight=1)
        self.visualise_frame.rowconfigure(0, weight=1)
        info = ttk.Frame(master=self)
        info.columnconfigure(0, weight=1)
        info.columnconfigure(1, weight=1)
        info.columnconfigure(2, weight=1)

        # Dataset frame
        dataset_frame = ttk.Labelframe(control, text="Datasets:", borderwidth=0)
        dataset_frame.columnconfigure(0, weight=1)

        self.datasets_box = ttk.Combobox(
            master=dataset_frame,
            textvariable=self.datasets_var,
            values=[],
            state="readonly",
        )
        self.datasets_box.bind("<<ComboboxSelected>>", self.dataset_changed)

        # Background frame
        bg_frame = ttk.Labelframe(control, text="Background:")
        bg_frame.columnconfigure(0, weight=1)
        bg_frame.rowconfigure(0, weight=1)
        self.bg_box = ttk.Combobox(
            master=bg_frame,
            textvariable=self.bg_var,
            values=["Estimated"],
            state="readonly",
        )
        self.bg_box.bind("<<ComboboxSelected>>", self.bg_changed)

        # Velocities frame
        self.velocities_frame = ttk.Labelframe(control, text="Velocities:")

        # Information frame
        marker_lbl = (("PS", "PD", "PAS"), ("PS", "PD", "PAS"), ("PC1", "PC2", "PC3"))
        for labels in marker_lbl:
            self.param_tables.append(ttk.Treeview(info, height=8))
            self.param_tables[-1].tag_configure("current", background="#f8d568")
            self.param_tables[-1].tag_configure("others", background="#FFFFFF")
            self.param_tables[-1]["columns"] = labels
            self.param_tables[-1].heading("#0", text="Region")
            self.param_tables[-1].column("#0", width=110, stretch=tk.YES)

            for l in labels:
                self.param_tables[-1].heading(l, text=l)
                self.param_tables[-1].column(l, width=80, stretch=tk.YES, anchor=tk.E)

        # Sign reversal frame
        reversal_frame = ttk.Labelframe(control, text="Reverse sign:")
        x = ttk.Checkbutton(
            reversal_frame,
            text="X",
            variable=self.reverse_vel_var[0],
            command=self.reversal_checked,
        )
        y = ttk.Checkbutton(
            reversal_frame,
            text="Y",
            variable=self.reverse_vel_var[1],
            command=self.reversal_checked,
        )
        z = ttk.Checkbutton(
            reversal_frame,
            text="Z",
            variable=self.reverse_vel_var[2],
            command=self.reversal_checked,
        )
        self.update_vel_btn = ttk.Button(
            control,
            text="Update",
            command=self.recalculate_velocities,
            state="disabled",
        )
        export_btn = ttk.Button(control, text="To Excel", command=self.export)
        export_super_btn = ttk.Button(
            control, text="Export superpixels", command=self.export_superpixel
        )

        # Grid all the widgets
        control.grid(sticky=tk.NSEW, pady=5)
        self.visualise_frame.grid(sticky=tk.NSEW, padx=5, pady=5)
        info.grid(sticky=tk.NSEW, pady=5)
        dataset_frame.grid(row=0, column=0, sticky=tk.NSEW, padx=5)
        self.datasets_box.grid(row=0, column=0, sticky=tk.NSEW)
        # bg_frame.grid(row=1, column=0, sticky=tk.NSEW, padx=5)
        # self.bg_box.grid(row=0, column=0, sticky=tk.NSEW)
        self.velocities_frame.grid(row=0, column=3, sticky=tk.NSEW, padx=5)
        reversal_frame.grid(row=0, column=1, sticky=tk.NSEW, padx=5)
        x.grid(row=0, column=0, sticky=tk.NSEW, padx=5)
        y.grid(row=0, column=1, sticky=tk.NSEW, padx=5)
        z.grid(row=0, column=2, sticky=tk.NSEW, padx=5)
        self.update_vel_btn.grid(row=0, column=2, sticky=tk.NSEW, padx=5)
        export_btn.grid(row=0, column=98, sticky=tk.NSEW, padx=5)
        export_super_btn.grid(row=0, column=99, sticky=tk.NSEW, padx=5)

        for i, table in enumerate(self.param_tables):
            table.grid(row=0, column=i, sticky=tk.NSEW, padx=5)

    def dataset_changed(self, *args):
        """Updates the view when the selected dataset is changed."""
        current = self.datasets_var.get()
        self.images = self.data.data_files.mag(current)
        if self.data.velocities.get(current):
            self.update_velocities_list(current)
        else:
            self.populate_bg_box(current)
            self.calculate_velocities(current)

        self.replot()

    def bg_changed(self, *args):
        """When the background is changed, new velocities need to be calculated."""
        bg = self.bg_var.get()
        dataset = self.datasets_var.get()
        existing_vels = self.data.velocities[dataset].keys()
        if not any([bg in vel_label for vel_label in existing_vels]):
            self.calculate_velocities(dataset)
            self.replot()

    def recalculate_velocities(self):
        """Recalculate velocities after a sign reversal."""
        self.update_vel_btn.state(["disabled"])
        dataset = self.datasets_var.get()
        existing_vels = self.data.velocities[dataset].keys()
        existing_bg = {vel_label.split(" - ")[-1] for vel_label in existing_vels}
        for bg in existing_bg:
            self.calculate_velocities(dataset, bg=bg)
            self.replot()

    def reversal_checked(self):
        """Enables/disables de update velocities button if amy sign reversal changes."""
        if tuple(var.get() for var in self.reverse_vel_var) != self.data.sign_reversal:
            self.update_vel_btn.state(["!disabled"])
        else:
            self.update_vel_btn.state(["disabled"])

    def find_velocity_limits(self, vel_label):
        """Finds suitable maximum and minimum for the velocity plots."""
        vel = self.data.velocities[self.datasets_var.get()][vel_label]
        for i, label in enumerate(self.axes_lbl):
            m = (vel[:, i, :].max() - vel[:, i, :].min()) * 0.10
            self.vel_lim[label] = (vel[:, i, :].min() - m, vel[:, i, :].max() + m)

    def replot(self):
        """Updates the plot to show the chosen velocity."""
        dataset = self.datasets_var.get()
        vel_label = self.velocities_var.get()
        bg = vel_label.split(" - ")[-1]
        if not any([bg in k for k in self.data.masks[dataset]]):
            self.calculate_velocities(dataset, bg, init_markers=False)

        self.find_velocity_limits(vel_label)
        if self.data.velocities[dataset][vel_label].shape[0] == 24:
            self.color_plots(dataset, vel_label)
        elif self.fig is None or self.current_region == -1:
            self.current_region = 0
            self.markers_figure(
                self.velocities,
                self.velocity_maps,
                self.images,
                self.markers[self.current_region],
            )
            self.populate_tables()
        else:
            self.current_region = -1
            self.scroll()
            self.draw()
            self.populate_tables()

        self.bg_var.set(self.velocities_var.get().split(" - ")[-1])

    def color_plots(self, dataset, vel_label):
        """Creates the color plots for the case of 24 angular regions."""
        gmark = self.data.markers[dataset][f"global - {vel_label.split(' - ')[-1]}"][0]
        markers_idx = gmark[:, :3, 0].flatten()
        self.fig = colour_figure(
            self.data.velocities[dataset][vel_label],
            self.region_labels(6),
            markers_idx,
            self.visualise_frame,
        )
        self.fig.canvas.draw_idle()
        self.current_region = -1
        markers = self.data.markers[dataset][vel_label.replace("24", "6")]
        self.populate_tables(markers)

    def marker_moved(self, table, marker):
        """Updates plot and table after a marker has been moved."""
        if marker == "ES":
            self.update_table_es()
            for f in self.fixed_markers:
                f.set_xdata([self.es_marker[0]] * 2)
        else:
            self.update_table_one_marker(table, marker)
            self.update_one_map(
                self.velocity_maps,
                self.images,
                self.markers[self.current_region],
                table,
                marker,
            )

    def scroll(self, step=1, *args):
        """Changes the region being plotted when scrolling with the mouse."""
        current_region = (self.current_region + step) % self.regions

        if self.current_region != current_region:
            self.fig.actions_manager.SimpleScroller.disabled = True
            self.current_region = current_region
            self.update_velocities(self.velocities, draw=False)
            self.update_maps(
                self.velocity_maps,
                self.images,
                self.markers[self.current_region],
                draw=False,
            )
            self.update_markers(self.markers[self.current_region], draw=False)
            self.populate_tables()
            self.fig.actions_manager.SimpleScroller.disabled = False
            for vel_label, ax in self.axes.items():
                ax.set_ylim(*self.vel_lim[vel_label])

        return self.current_region, None, None

    @property
    def regions(self) -> int:
        """Number of regions for the selected velocity."""
        return len(
            self.data.velocities[self.datasets_var.get()][self.velocities_var.get()]
        )

    @property
    def velocities(self) -> np.ndarray:
        """Velocities of the current region."""
        return self.data.velocities[self.datasets_var.get()][self.velocities_var.get()][
            self.current_region
        ]

    @property
    def markers(self) -> np.ndarray:
        """Markers of the current region."""
        return self.data.markers[self.datasets_var.get()][self.velocities_var.get()]

    @property
    def masks(self) -> np.ndarray:
        """Masks for the current region"""
        return (
            self.data.masks[self.datasets_var.get()][self.velocities_var.get()]
            != self.current_region + 1
        )

    @property
    def velocity_maps(self):
        """Calculate velocity maps out of the masks and cylindrical velocities."""
        cyl_label = f"cylindrical -{self.velocities_var.get().split('-')[-1]}"
        cylindrical = self.data.masks[self.datasets_var.get()][cyl_label]
        bmask = np.broadcast_to(self.masks, cylindrical.shape)
        return np.ma.masked_where(bmask, cylindrical)

    @staticmethod
    def region_labels(regions):
        """Provides the region labels, if any."""
        if regions == 6:
            return "AS", "A", "AL", "IL", "I", "IS"
        else:
            return list(range(1, regions + 1))

    def populate_tables(self, markers=None):
        """Populates the information tables with the marker parameters."""
        markers = markers if markers is not None else self.markers
        for t in self.param_tables:
            old_list = t.get_children()
            if len(old_list) > 0:
                t.delete(*old_list)

        labels = self.region_labels(len(markers))
        for i, t in enumerate(self.param_tables):
            vel = t.insert("", tk.END, text="Velocity (cm/s)", open=True)
            time = t.insert("", tk.END, text="Norm. Time (ms)", open=True)
            for j, marker in enumerate(markers):
                tag = "current" if j == self.current_region else "others"
                val = np.around(marker[i, :3, 1], decimals=2).tolist()
                t.insert(vel, tk.END, text=labels[j], values=val, tags=(tag,))
                val = np.around(marker[i, :3, 2], decimals=2).tolist()
                t.insert(time, tk.END, text=labels[j], values=val, tags=(tag,))

    def update_table_one_marker(self, table, marker):
        """Updates peak velocity and time table entry for a single marker."""
        table = self.axes_lbl.index(table)
        idx = self.marker_idx[marker]
        t = self.param_tables[table]
        velitem = t.get_children(t.get_children()[0])[self.current_region]
        timeitem = t.get_children(t.get_children()[1])[self.current_region]

        t.set(
            velitem,
            column=marker,
            value=round(self.markers[self.current_region, table, idx, 1], 2),
        )
        t.set(
            timeitem,
            column=marker,
            value=round(self.markers[self.current_region, table, idx, 2], 2),
        )

    def update_table_es(self):
        """Updates a row of the markers table after ES has moved."""
        for i, t in enumerate(self.param_tables):
            for r in range(len(self.markers)):
                timeitem = t.get_children(t.get_children()[1])[r]

                t.item(
                    timeitem,
                    values=np.around(self.markers[r, i, :3, 2], decimals=2).tolist(),
                )

    def update_velocities_list(self, dataset):
        """Updates the list of radio buttons with the currently available velocities."""
        velocities = self.data.velocities[dataset]

        for v in self.velocities_frame.winfo_children():
            v.grid_remove()

        vel_list = [v for v in velocities if "global" in v or "6" in v or "24" in v]
        for i, v in enumerate(vel_list):
            text = v.split(" - ")[0]
            ttk.Radiobutton(
                self.velocities_frame,
                text=text,
                value=v,
                variable=self.velocities_var,
                command=self.replot,
            ).grid(row=0, column=i, sticky=tk.NSEW)

        if self.velocities_var.get() not in velocities and len(velocities) > 0:
            self.velocities_var.set(vel_list[0])

        self.bg_var.set(self.velocities_var.get().split(" - ")[-1])

    def calculate_velocities(self, dataset, bg=None, init_markers=True):
        """Calculate pre-defined velocities for the chosen dataset."""
        self.controller.calculate_velocities(
            dataset_name=dataset,
            global_velocity=True,
            angular_regions=[6, 24],
            radial_regions=[3],
            bg=self.bg_var.get() if bg is None else bg,
            sign_reversal=tuple(var.get() for var in self.reverse_vel_var),
            init_markers=init_markers,
        )
        self.update_velocities_list(self.datasets_var.get())

    def export(self, *args):
        """Exports the current velocity data to an XLSX file."""
        meta = self.data.metadata()
        name, date = [meta[key] for key in ["Patient Name", "Date of Scan"]]
        init = f"{name}_{date}_{self.datasets_var.get()}_velocity.xlsx"

        filename = tk.filedialog.asksaveasfilename(
            initialfile=init,
            defaultextension="xlsx",
            filetypes=[("Excel files", "*.xlsx")],
        )
        if filename != "":
            self.controller.export_velocity(
                filename=filename,
                dataset=self.datasets_var.get(),
                vel_label=self.velocities_var.get(),
            )

    def export_superpixel(self, *args):
        """ Exports the current superpixel velocity data to an XLSX file.

        TODO: Remove in final version
        """
        from ..models.writers import export_superpixel

        meta = self.data.metadata()
        name, date = [meta[key] for key in ["Patient Name", "Date of Scan"]]
        init = f"{name}_{date}_{self.datasets_var.get()}_velocity_super.xlsx"

        filename = tk.filedialog.asksaveasfilename(
            initialfile=init,
            defaultextension="xlsx",
            filetypes=[("Excel files", "*.xlsx")],
        )
        if filename != "":
            export_superpixel(
                data=self.data, dataset=self.datasets_var.get(), filename=filename
            )

    def populate_dataset_box(self):
        """Populate the dataset box with available segmentations."""
        values = sorted(self.data.segments.keys(), key=get_sa_location)
        current = self.datasets_var.get()
        self.datasets_box.config(values=values)
        if current not in values:
            self.datasets_var.set(values[0])

    def populate_bg_box(self, dataset):
        """Populates the background box and try to match the bg choice by name."""
        values = ["Estimated", "None"] + (
            self.data.bg_files.datasets if self.data.bg_files is not None else []
        )
        self.bg_box.config(values=values)
        if dataset in values:
            self.bg_var.set(dataset)
        else:
            self.bg_var.set(values[0])

    def update_sign_reversal(self):
        """Updates the sign reversal information with data.sign_reversal info."""
        for i, var in enumerate(self.data.sign_reversal):
            self.reverse_vel_var[i].set(bool(var))

    def update_widgets(self):
        """ Updates widgets after an update in the data var. """
        self.populate_dataset_box()
        self.populate_bg_box(self.datasets_var.get())
        self.update_sign_reversal()
        self.calculate_all()
        self.dataset_changed()

    def calculate_all(self):
        """ Calculates all velocities not already calculated. """
        to_regenerate = [
            k
            for k in self.data.velocities.keys()
            if len(self.data.velocities.get(k, {}).values()) > 0
            and list(self.data.velocities.get(k, {}).values())[0] is None
        ]
        if len(to_regenerate) > 0:
            self.controller.regenerate_velocities(
                datasets=to_regenerate, callback=self.master.progress
            )

        existing = [
            k
            for k in self.data.velocities.keys()
            if len(self.data.velocities.get(k, {}).values()) > 0
            and list(self.data.velocities.get(k, {}).values())[0] is not None
        ]

        needed = list(self.data.segments.keys())
        for i, d in enumerate(needed):
            if d not in existing:
                self.master.progress(
                    f"Calculating velocities for dataset {d} - {i+1}/{len(needed)}",
                    i / len(needed),
                )
                self.calculate_velocities(d)
        self.master.progress(f"Done!", 1)

    def clear_widgets(self):
        """ Clear widgets after removing the data. """
        pass

    @property
    def es_marker(self):
        """ ES marker position. """
        return self.data.markers[self.datasets_var.get()]["global - Estimated"][
            0, 1, 3, :2
        ]

    def markers_figure(
        self,
        velocities: np.ndarray,
        vel_masks: np.ndarray,
        images: np.ndarray,
        markers: np.ndarray,
    ):
        self.fig = Figure(constrained_layout=True)
        canvas = FigureCanvasTkAgg(self.fig, master=self.visualise_frame)
        canvas.get_tk_widget().grid(row=0, column=0, sticky=tk.NSEW)
        toolbar_frame = ttk.Frame(master=self.visualise_frame)
        toolbar_frame.grid(row=1, column=0, sticky=tk.NSEW)
        NavigationToolbar2Tk(canvas, toolbar_frame)

        self.fig.actions_manager = FigureActionsManager(
            self.fig, Markers, SimpleScroller
        )
        self.fig.actions_manager.Markers.set_marker_moved(  # type: ignore
            self.update_marker
        )
        self.fig.actions_manager.SimpleScroller.set_scroller(  # type: ignore
            self.scroll
        )

        gs = self.fig.add_gridspec(2, 9, height_ratios=[5, 2])
        self.axes = self.add_velocity_subplots(gs)
        self.maps = self.add_maps_subplots(gs)
        self.vel_lines = self.add_velocity_lines(velocities)
        self.bg_images, self.vel_masks, self.cbar = self.images_and_velocity_masks(
            images, vel_masks, markers
        )
        self.marker_artists = self.add_markers(markers)

        self.draw()

    def draw(self):
        """Convenience method for re-drawing the figure."""
        self.fig.canvas.draw_idle()

    def add_velocity_subplots(self, gs):
        """Adds the velocity subplots."""
        ax_long = self.fig.add_subplot(gs[0, :3])
        ax_rad = self.fig.add_subplot(gs[0, 3:6])
        ax_circ = self.fig.add_subplot(gs[0, 6:])

        ax_long.axhline(color="k", lw=1)
        ax_rad.axhline(color="k", lw=1)
        ax_circ.axhline(color="k", lw=1)

        ax_long.set_title("Longitudinal")
        ax_long.set_ylabel("Velocity (cm/s)")
        ax_long.set_xlabel("Frame")
        ax_rad.set_title("Radial")
        ax_rad.set_xlabel("Frame")
        ax_circ.set_title("Circumferential")
        ax_circ.set_xlabel("Frame")

        ax_long.set_xlim(0, self.data.data_files.frames)
        ax_rad.set_xlim(0, self.data.data_files.frames)
        ax_circ.set_xlim(0, self.data.data_files.frames)

        return {"_long": ax_long, "_rad": ax_rad, "_circ": ax_circ}

    def add_velocity_lines(self, vels):
        """Add lines to the velocity plots.

        vels - 2D array with the velocities with shape [components (3), frames]
        """
        x = np.arange(vels.shape[-1])
        output = dict()
        for i, label in enumerate(self.axes_lbl):
            output[label] = self.axes[label].plot(x, vels[i], "k", label=label)[0]
            self.axes[label].set_ylim(*self.vel_lim[label])
            self.axes[label].autoscale(False)
        return output

    def add_maps_subplots(self, gs):
        """Adds the maps subplots."""
        maps = []
        colours = ["red", "green", "blue"] * 2 + ["orange", "darkblue", "purple"]
        for i, color in enumerate(colours):
            maps.append(self.fig.add_subplot(gs[1, i]))
            maps[-1].get_xaxis().set_visible(False)
            maps[-1].get_yaxis().set_visible(False)
            for side in ["left", "right", "bottom", "top"]:
                maps[-1].spines[side].set_color(color)
                maps[-1].spines[side].set_linewidth(3)

        return maps

    def images_and_velocity_masks(self, mag, vel_masks, markers):
        """Add bg and masks to the map subplots."""
        bg = {l: [] for l in self.axes_lbl}
        masks = {l: [] for l in self.axes_lbl}

        if "global" in self.velocities_var.get():
            self.limits = self.find_limits(vel_masks[0, 0])

        rmin, rmax, cmin, cmax = self.limits
        vmin, vmax = vel_masks.min(), vel_masks.max()
        for i in range(9):
            axes = self.axes_lbl[i // 3]
            frame = int(markers[i // 3, i % 3, 0])
            bg[axes].append(
                self.maps[i].imshow(
                    mag[frame, rmin : rmax + 1, cmin : cmax + 1],
                    cmap=plt.get_cmap("binary_r"),
                )
            )
            masks[axes].append(
                self.maps[i].imshow(
                    vel_masks[i // 3, frame, rmin : rmax + 1, cmin : cmax + 1],
                    cmap=plt.get_cmap("seismic"),
                    vmin=vmin,
                    vmax=vmax,
                )
            )

        cbar = self.fig.colorbar(masks["_long"][0], ax=self.maps[0], pad=-1.4)

        return bg, masks, cbar

    @staticmethod
    def find_limits(mask, margin=30):
        """Find the appropiate limits of a masked array in order to plot it nicely."""
        rows, cols = mask.nonzero()

        cmin = max(cols.min() - margin, 0)
        rmin = max(rows.min() - margin, 0)
        cmax = min(cols.max() + margin, mask.shape[0])
        rmax = min(rows.max() + margin, mask.shape[1])

        return rmin, rmax, cmin, cmax

    def add_markers(self, markers):
        """Adds markers to the plots.).

        - markers - Contains all the marker information for all regions and components.
            Their shape is [regions, component (3), marker_id (3 or 4), marker_data (3)]
        """
        add_marker = self.fig.actions_manager.Markers.add_marker

        vel_lbl = ["_long"] * 3 + ["_rad"] * 3 + ["_circ"] * 3
        colors = ["red", "green", "blue"] * 2 + ["orange", "darkblue", "purple"]
        marker_lbl = ["PS", "PD", "PAS"] * 2 + ["PC1", "PC2", "PC3"]

        markers_artists = []
        for i, label in enumerate(vel_lbl):
            markers_artists.append(
                add_marker(
                    self.vel_lines[label],
                    xy=markers[i // 3, i % 3, :2],
                    label=marker_lbl[i],
                    color=colors[i],
                    marker=str(i % 3 + 1),
                    markeredgewidth=2,
                )
            )

        for label in ("_long", "_rad", "_circ"):
            self.fixed_markers.append(
                self.vel_lines[label].axes.axvline(
                    self.es_marker[0], color="grey", linewidth=2, linestyle="--"
                )
            )

        markers_artists.append(
            add_marker(
                self.vel_lines["_rad"],
                xy=self.es_marker,
                vline=True,
                label="ES",
                color="black",
                linewidth=2,
            )
        )

        self.axes["_long"].legend(frameon=False, markerscale=0.5)
        self.axes["_rad"].legend(frameon=False, markerscale=0.5)
        self.axes["_circ"].legend(frameon=False, markerscale=0.5)

        return markers_artists

    def update_marker_position(self, marker_label, data_label, new_x):
        """Convenience method to update the marker position."""
        self.actions_manager.Markers.update_marker_position(
            marker_label, data_label, new_x
        )

    def update_line(self, vel_label, data, draw=False):
        """Updates the data of the chosen line."""
        self.vel_lines[vel_label].set_data(data)
        if draw:
            self.draw()

    def update_bg(self, vel_label, idx, data, draw=False):
        """Updates the data of the chosen bg."""
        self.update_data(self.bg_images[vel_label][idx], vel_label, data, draw)

    def update_mask(self, vel_label, idx, data, draw=False):
        """Updates the data of the chosen bg."""
        self.update_data(self.vel_masks[vel_label][idx], vel_label, data, draw)

    def update_data(self, subplot, vel_label, data, draw=False):
        """Common data updating method."""
        subplot.set_data(data)
        self.axes[vel_label].relim()
        self.axes[vel_label].autoscale()

        if draw:
            self.draw()

    def update_maps(
        self, vel_masks: np.ndarray, images: np.ndarray, markers: np.ndarray, draw=True
    ):
        """Updates the maps (masks and background data)."""
        rmin, rmax, cmin, cmax = self.limits
        for i in range(9):
            axes = self.axes_lbl[i // 3]
            frame = int(markers[i // 3, i % 3, 0])
            self.update_mask(
                axes, i % 3, vel_masks[i // 3, frame, rmin : rmax + 1, cmin : cmax + 1]
            )
            self.update_bg(axes, i % 3, images[frame, rmin : rmax + 1, cmin : cmax + 1])

        if draw:
            self.draw()

    def update_one_map(
        self,
        vel_masks: np.ndarray,
        images: np.ndarray,
        markers: np.ndarray,
        axes: str,
        marker_lbl: str,
    ):
        """Updates the maps correspoinding to a single marker."""
        rmin, rmax, cmin, cmax = self.limits
        component = self.axes_lbl.index(axes)
        idx = self.marker_idx[marker_lbl]
        frame = int(markers[component, idx, 0])
        self.update_mask(
            axes, idx, vel_masks[component, frame, rmin : rmax + 1, cmin : cmax + 1]
        )
        self.update_bg(axes, idx, images[frame, rmin : rmax + 1, cmin : cmax + 1])
        self.axes[axes].set_ylim(*self.vel_lim[axes])
        self.draw()

    def update_velocities(self, vels, draw=True):
        """Updates all velocities."""
        x = np.arange(vels.shape[-1])
        for i, label in enumerate(self.axes_lbl):
            self.update_line(label, (x, vels[i]))
        if draw:
            self.draw()

    def update_markers(self, markers, draw=True):
        """Updates the position of all markers in a figure."""
        update_position = self.fig.actions_manager.Markers.update_marker_position

        for i, artist in enumerate(self.marker_artists[:-1]):
            update_position(artist, int(markers[i // 3, i % 3, 0]))

        update_position(self.marker_artists[-1], self.es_marker[0])

        for f in self.fixed_markers:
            f.set_xdata([self.es_marker[0]] * 2)

        if draw:
            self.draw()

    def update_marker(self, marker, data, x, y, position):
        """When a marker moves, mask data should be updated."""
        self.controller.update_marker(
            dataset=self.datasets_var.get(),
            vel_label=self.velocities_var.get(),
            region=self.current_region,
            component=self.axes_lbl.index(data.get_label()),
            marker_idx=self.marker_idx[marker.get_label()],
            position=position,
        )
        self.marker_moved(data.get_label(), marker.get_label())


def colour_figure(
    velocities: np.ndarray, labels: tuple, markers_idx: np.ndarray, master: ttk.Frame
) -> Figure:
    """Creates the color plots for the regional velocities."""
    fig = Figure(constrained_layout=True)
    canvas = FigureCanvasTkAgg(fig, master=master)
    canvas.get_tk_widget().grid(row=0, column=0, sticky=tk.NSEW)
    ax = fig.subplots(ncols=3, nrows=1)

    space = velocities.shape[0] / len(labels)
    lines_pos = np.arange(space, velocities.shape[0], space) - 0.5
    labels_pos = np.arange(space // 2, velocities.shape[0], space) - 0.5
    marker_lbl = ["PS", "PD", "PAS"] * 2 + ["PC1", "PC2", "PC3"]

    for i, title in enumerate(("Longitudinal", "Radial", "Circumferential")):
        ax[i].imshow(
            velocities[:, i],
            cmap=plt.get_cmap("jet"),
            aspect="auto",
            interpolation="bilinear",
        )
        ax[i].set_title(title)

        ax[i].set_yticks(lines_pos, minor=True)
        ax[i].set_yticks(labels_pos, minor=False)
        ax[i].set_yticklabels(labels[::-1], minor=False)
        ax[i].yaxis.grid(True, which="minor", color="k", linestyle="-")
        ax[i].set_ylim((-0.5, velocities.shape[0] - 0.5))

        ax[i].set_xticks(markers_idx[3 * i : 3 * i + 3], minor=False)
        ax[i].set_xticklabels(marker_lbl[3 * i : 3 * i + 3], minor=False)

        fig.colorbar(ax[i].images[0], ax=ax[i], orientation="horizontal")

    return fig
