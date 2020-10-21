import tkinter as tk
from tkinter import ttk
import tkinter.filedialog

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from .base_window_and_task import Requisites, TaskViewBase, register_view
from .figure_actions_manager import FigureActionsManager
from .figure_actions import Markers, SimpleScroller
from ..tools import get_sa_location


@register_view
class StrainTaskView(TaskViewBase):

    requisites = Requisites.VELOCITIES
    axes_lbl = ("_long", "_rad", "_circ")
    marker_idx = {"PS": 0, "P": 2}

    def __init__(self, root, controller):

        super().__init__(
            root,
            controller,
            button_text="Strain",
            button_image="strain.gif",
            button_row=3,
        )
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)

        self.visualise_frame = None
        self.datasets_box = None
        self.datasets_var = tk.StringVar(value="")
        self.output_frame = None
        self.gls_frame = None
        self.strain_var = tk.StringVar(value="")
        self.plot = None
        self.regional_fig = None
        self.param_tables = []
        self.current_region = 0
        self.images = None
        self.exclude = (tk.BooleanVar(value=False), tk.BooleanVar(value=False))
        self.effective_disp = tk.BooleanVar(value=True)
        self.resample = tk.BooleanVar(value=True)
        self.gls = (tk.StringVar(), tk.StringVar(), tk.StringVar())
        self.timeshift_var = tk.DoubleVar(value=0.0)
        self.export_btn = None
        self.export_twist_btn = None

        # Figure-related variables
        self.fig = None
        self.axes = None
        self.maps = None
        self.bg_images = None
        self.strain_lines = None
        self.strain_masks = None
        self.cbar = None
        self.limits = None
        self.marker_artists = None
        self.strain_lim = dict()
        self.gls_lbl = []
        self.fixed_markers = []

        self.create_controls()

    def tkraise(self, *args):
        """Brings the frame to the front."""
        super().tkraise()
        self.populate_tables()

    def create_controls(self):
        """ Creates all the widgets of the view. """
        # Top frames
        control = ttk.Frame(master=self)
        control.columnconfigure(4, weight=1)
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

        # Strain frame
        strain_frame = ttk.Frame(control)

        ex_first = ttk.Checkbutton(
            master=strain_frame, text="Exclude first", variable=self.exclude[0]
        )
        ex_last = ttk.Checkbutton(
            master=strain_frame, text="Exclude last", variable=self.exclude[1]
        )
        effective = ttk.Checkbutton(
            master=strain_frame, text="Eff. displacement.", variable=self.effective_disp
        )
        resample = ttk.Checkbutton(
            master=strain_frame, text="Resample RR", variable=self.resample
        )
        recalc_btn = ttk.Button(
            master=strain_frame, text="Recalculate", command=self.recalculate
        )
        timeshift_lbl = ttk.Label(strain_frame, text="Time shift (s):", width=10)
        timeshift = ttk.Entry(strain_frame, textvariable=self.timeshift_var, width=10)

        # Strain frame
        self.output_frame = ttk.Frame(master=control)
        self.output_frame.rowconfigure(0, weight=1)
        self.output_frame.rowconfigure(1, weight=1)
        ttk.Label(self.output_frame, text="Strain:").grid(
            row=0, column=0, sticky=tk.NSEW
        )
        for i, v in enumerate(self.gls):
            self.output_frame.columnconfigure(i, weight=1)
            self.gls_lbl.append(
                ttk.Label(master=self.output_frame, textvariable=v, width=10)
            )
            self.gls_lbl[-1].grid(row=1, column=i, sticky=tk.NSEW)

        # Information frame
        marker_lbl = (("PS", "ES", "P"),) * 3
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

        self.export_btn = ttk.Button(
            control, text="To Excel", command=self.export, state="disabled"
        )
        self.export_twist_btn = ttk.Button(
            control,
            text="Export rotation",
            command=self.export_rotation,
            state="disabled",
        )

        # Grid all the widgets
        control.grid(sticky=tk.NSEW, pady=5)
        self.visualise_frame.grid(sticky=tk.NSEW, padx=5, pady=5)
        info.grid(sticky=tk.NSEW, pady=5)
        dataset_frame.grid(row=0, column=0, sticky=tk.NSEW, padx=5)
        self.datasets_box.grid(row=0, column=0, sticky=tk.NSEW)
        strain_frame.grid(row=0, column=1, sticky=tk.NSEW, padx=5)
        ex_first.grid(row=0, column=0, sticky=tk.NSEW)
        ex_last.grid(row=1, column=0, sticky=tk.NSEW)
        effective.grid(row=0, column=1, sticky=tk.NSEW, padx=5)
        resample.grid(row=1, column=1, sticky=tk.NSEW, padx=5)
        timeshift_lbl.grid(row=0, column=2, sticky=tk.NSEW)
        timeshift.grid(row=1, column=2, sticky=tk.NSEW)
        recalc_btn.grid(row=0, column=3, rowspan=2, sticky=tk.NSEW, padx=5)
        self.output_frame.grid(row=0, column=4, sticky=tk.NSEW)
        self.export_btn.grid(row=0, column=98, sticky=tk.NSEW, padx=5)
        self.export_twist_btn.grid(row=0, column=99, sticky=tk.NSEW, padx=5)
        for i, table in enumerate(self.param_tables):
            table.grid(row=0, column=i, sticky=tk.NSEW, padx=5)

    def dataset_changed(self, *args):
        """Updates the view when the selected dataset is changed."""
        self.controller.progress("Changing selected cine...")
        current = self.datasets_var.get()
        self.images = self.data.data_files.mag(current)
        self.timeshift_var.set(self.data.timeshift)
        if self.data.strain.get(current):
            self.update_strain_list(current)
        else:
            self.calculate_strain()
        self.replot()
        self.controller.progress("Done!")

    def find_strain_limits(self, strain_label):
        """Finds suitable maximum and minimum for the strain plots."""
        strain = self.data.strain[self.datasets_var.get()][strain_label]
        for i, label in enumerate(self.axes_lbl):
            m = (strain[:, i, :].max() - strain[:, i, :].min()) * 0.10
            self.strain_lim[label] = (
                strain[:, i, :].min() - m,
                strain[:, i, :].max() + m,
            )

    def display_plots(self, show=True):
        """Show/hide the plots"""
        if show:
            self.visualise_frame.grid(row=1, column=0, sticky=tk.NSEW, padx=5, pady=5)
            self.export_btn.state(["!disabled"])
            self.export_twist_btn.state(["!disabled"])
            self.controller.window.update()
        else:
            self.visualise_frame.grid_forget()
            self.export_btn.state(["disabled"])
            self.export_twist_btn.state(["disabled"])
            self.controller.window.update()

    def replot(self):
        """Updates the plot to show the chosen strain."""
        strain_label = self.strain_var.get()
        self.find_strain_limits(strain_label)
        if self.fig is None or self.current_region == -1:
            self.current_region = 0
            self.markers_figure(
                self.strain,
                self.strain_maps,
                self.images,
                self.markers[self.current_region],
            )
            self.populate_tables()
        else:
            self.current_region = -1
            self.scroll()
            self.draw()
            self.populate_tables()

        self.display_plots(True)

    def marker_moved(self, table, marker):
        """Updates plot and table after a marker has been moved."""
        if marker == "ES":
            self.update_table_es()
        else:
            self.update_table_one_marker(table, marker)
            self.update_one_map(
                self.strain_maps,
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
            self.update_strains(self.strain, draw=False)
            self.update_maps(
                self.strain_maps,
                self.images,
                self.markers[self.current_region],
                draw=False,
            )
            self.update_markers(self.markers[self.current_region], draw=False)
            self.populate_tables()
            self.fig.actions_manager.SimpleScroller.disabled = False
            for strain_label, ax in self.axes.items():
                ax.set_ylim(*self.strain_lim[strain_label])

        return self.current_region, None, None

    @property
    def regions(self) -> int:
        """Number of regions for the selected strain."""
        return len(self.data.strain[self.datasets_var.get()][self.strain_var.get()])

    @property
    def strain(self) -> np.ndarray:
        """Strains of the current region."""
        return self.data.strain[self.datasets_var.get()][self.strain_var.get()][
            self.current_region
        ]

    @property
    def markers(self) -> np.ndarray:
        """Markers of the current region."""
        return self.data.strain_markers[self.datasets_var.get()][self.strain_var.get()]

    @property
    def masks(self) -> np.ndarray:
        """Masks for the current region"""
        return (
            self.data.masks[self.datasets_var.get()][self.strain_var.get()]
            != self.current_region + 1
        )

    @property
    def strain_maps(self):
        """Calculate strain maps out of the masks and cylindrical strain."""
        cyl_label = f"cylindrical -{self.strain_var.get().split('-')[-1]}"
        cylindrical = self.data.strain[self.datasets_var.get()][cyl_label]
        bmask = np.broadcast_to(self.masks, cylindrical.shape)
        return np.ma.masked_where(bmask, cylindrical)

    def region_labels(self, regions):
        """Provides the region labels, if any."""
        if regions == 6:
            labels = "AS", "A", "AL", "IL", "I", "IS"
            if self.data.orientation == "CCW":
                labels = labels[::-1]
            return labels
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
            strain = t.insert("", tk.END, text="Strain (%)", open=True)
            time = t.insert("", tk.END, text="Time (ms)", open=True)
            for j, marker in enumerate(markers):
                tag = "current" if j == self.current_region else "others"
                val = np.around(marker[i, :, 1], decimals=2).tolist()
                t.insert(strain, tk.END, text=labels[j], values=val, tags=(tag,))
                val = np.around(marker[i, :, 2] * 1000, decimals=0).tolist()
                t.insert(time, tk.END, text=labels[j], values=val, tags=(tag,))

    def update_table_one_marker(self, table, marker):
        """Updates peak strain and time table entry for a single marker."""
        table = self.axes_lbl.index(table)
        idx = self.marker_idx[marker]
        t = self.param_tables[table]
        strainitem = t.get_children(t.get_children()[0])[self.current_region]
        timeitem = t.get_children(t.get_children()[1])[self.current_region]

        t.set(
            strainitem,
            column=marker,
            value=round(self.markers[self.current_region, table, idx, 1], 2),
        )
        t.set(
            timeitem,
            column=marker,
            value=round(self.markers[self.current_region, table, idx, 2], 2),
        )

    def update_strain_list(self, dataset):
        """Updates the list of radio buttons with the currently available strains."""
        strain = self.data.strain[dataset]

        # for v in self.output_frame.winfo_children():
        #     v.grid_remove()

        vel_list = [v for v in strain if "global" in v or "6" in v]
        for i, v in enumerate(vel_list):
            text = v.split(" - ")[0]
            ttk.Radiobutton(
                self.output_frame,
                text=text,
                value=v,
                variable=self.strain_var,
                command=self.replot,
            ).grid(row=0, column=i + 1, sticky=tk.NSEW)

        if self.strain_var.get() not in strain and len(strain) > 0:
            self.strain_var.set(vel_list[0])

    def calculate_strain(self, recalculate=False):
        """Calculate strain for the chosen dataset."""
        datasets = sorted(self.data.velocities.keys(), key=get_sa_location)
        if self.exclude[0].get():
            datasets.pop(0)
        if self.exclude[1].get():
            datasets.pop(-1)

        self.display_plots(False)
        self.controller.calculate_strain(
            datasets=datasets,
            effective_displacement=self.effective_disp.get(),
            resample=self.resample.get(),
            recalculate=recalculate,
            timeshift=self.timeshift_var.get(),
        )
        lbl = ("PSS", "PES", "PS")
        for i, v in enumerate(self.gls):
            v.set(value=f"{lbl[i]}: {round(self.data.gls[i] * 100, 1)}%")
        self.populate_dataset_box(datasets)
        self.update_strain_list(self.datasets_var.get())

    def recalculate(self, *args):
        """Re-calculate strain after changing any of the conditions."""
        self.calculate_strain(recalculate=True)
        self.replot()

    def export(self, *args):
        """Exports the current strain data to an XLSX file."""
        meta = self.data.metadata()
        name, date = [meta[key] for key in ["Patient Name", "Date of Scan"]]
        init = f"{name}_{date}_{self.datasets_var.get()}_strain.xlsx"

        filename = tk.filedialog.asksaveasfilename(
            initialfile=init,
            defaultextension="xlsx",
            filetypes=[("Excel files", "*.xlsx")],
        )
        if filename != "":
            self.controller.export_strain(
                filename=filename,
                dataset=self.datasets_var.get(),
                vel_label=self.strain_var.get(),
            )

    def export_rotation(self, *args):
        """Exports the rotation data to an XLSX file."""
        meta = self.data.metadata()
        name, date = [meta[key] for key in ["Patient Name", "Date of Scan"]]
        init = f"{name}_{date}_rotation.xlsx"

        filename = tk.filedialog.asksaveasfilename(
            initialfile=init,
            defaultextension="xlsx",
            filetypes=[("Excel files", "*.xlsx")],
        )
        if filename != "":
            self.controller.export_rotation(filename=filename)

    def populate_dataset_box(self, datasets=None):
        """Populate the dataset box with the datasets that have velocities."""
        vdatasets = sorted(self.data.velocities.keys(), key=get_sa_location)
        values = vdatasets if datasets is None else datasets
        current = self.datasets_var.get()
        self.datasets_box.config(values=values)
        if current not in values:
            self.datasets_var.set(values[0])

    def update_widgets(self):
        """ Updates widgets after an update in the data var. """
        self.populate_dataset_box()
        self.dataset_changed()

    def clear_widgets(self):
        """ Clear widgets after removing the data. """
        pass

    def markers_figure(
        self,
        strain: np.ndarray,
        masks: np.ndarray,
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
        self.axes = self.add_strain_subplots(gs)
        self.maps = self.add_maps_subplots(gs)
        self.strain_lines = self.add_strain_lines(strain)
        self.bg_images, self.strain_masks, self.cbar = self.images_and_strain_masks(
            images, masks, markers
        )
        self.marker_artists = self.add_markers(markers)

        self.draw()

    def draw(self):
        """Convenience method for re-drawing the figure."""
        self.fig.canvas.draw_idle()

    def add_strain_subplots(self, gs):
        """Adds the strain subplots."""
        ax_long = self.fig.add_subplot(gs[0, :3])
        ax_rad = self.fig.add_subplot(gs[0, 3:6])
        ax_circ = self.fig.add_subplot(gs[0, 6:])

        ax_long.axhline(color="k", lw=1)
        ax_rad.axhline(color="k", lw=1)
        ax_circ.axhline(color="k", lw=1)

        ax_long.set_title("Longitudinal")
        ax_long.set_ylabel("Strain (%)")
        ax_long.set_xlabel("Frame")
        ax_rad.set_title("Radial")
        ax_rad.set_xlabel("Frame")
        ax_circ.set_title("Circumferential")
        ax_circ.set_xlabel("Frame")

        ax_long.set_xlim(0, self.data.data_files.frames)
        ax_rad.set_xlim(0, self.data.data_files.frames)
        ax_circ.set_xlim(0, self.data.data_files.frames)

        return {"_long": ax_long, "_rad": ax_rad, "_circ": ax_circ}

    def add_strain_lines(self, strain):
        """Add lines to the strain plots.

        strain - 2D array with the strains with shape [components (3), frames]
        """
        x = np.arange(strain.shape[-1])
        output = dict()
        for i, label in enumerate(self.axes_lbl):
            output[label] = self.axes[label].plot(x, strain[i], "k", label=label)[0]
            self.axes[label].set_ylim(*self.strain_lim[label])
            self.axes[label].autoscale(False)
        return output

    def add_maps_subplots(self, gs):
        """Adds the maps subplots."""
        maps = []
        colours = ["red", "grey", "blue"] * 3
        for i, color in enumerate(colours):
            maps.append(self.fig.add_subplot(gs[1, i]))
            maps[-1].get_xaxis().set_visible(False)
            maps[-1].get_yaxis().set_visible(False)
            for side in ["left", "right", "bottom", "top"]:
                maps[-1].spines[side].set_color(color)
                maps[-1].spines[side].set_linewidth(3)

        return maps

    def images_and_strain_masks(self, mag, strain_masks, markers):
        """Add bg and masks to the map subplots."""
        bg = {l: [] for l in self.axes_lbl}
        masks = {l: [] for l in self.axes_lbl}

        if "global" in self.strain_var.get():
            self.limits = self.find_limits(strain_masks[1, 0])

        rmin, rmax, cmin, cmax = self.limits
        vmin, vmax = strain_masks.min(), strain_masks.max()
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
                    strain_masks[i // 3, frame, rmin : rmax + 1, cmin : cmax + 1],
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

        comp_lbl = ["_long", "_rad", "_circ"]
        colors = ["red", "grey", "blue"]
        marker_lbl = ("PS", "ES", "P")

        markers_artists = []
        for i, clbl in enumerate(comp_lbl):
            for j, mlbl in enumerate(marker_lbl):
                if mlbl == "ES":
                    self.fixed_markers.append(
                        self.strain_lines[clbl].axes.axvline(
                            markers[i, j, 0],
                            label=mlbl,
                            color="grey",
                            linewidth=2,
                            linestyle="--",
                        )
                    )
                    continue
                markers_artists.append(
                    add_marker(
                        self.strain_lines[clbl],
                        xy=markers[i, j, :2],
                        label=mlbl,
                        color=colors[j],
                        marker=str(j + 1),
                        markeredgewidth=1.5,
                        markersize=15,
                    )
                )

        self.axes["_long"].legend(frameon=False, markerscale=0.7)
        self.axes["_rad"].legend(frameon=False, markerscale=0.7)
        self.axes["_circ"].legend(frameon=False, markerscale=0.7)

        return markers_artists

    def update_marker_position(self, marker_label, data_label, new_x):
        """Convenience method to update the marker position."""
        self.actions_manager.Markers.update_marker_position(
            marker_label, data_label, new_x
        )

    def update_line(self, label, data, draw=False):
        """Updates the data of the chosen line."""
        self.strain_lines[label].set_data(data)
        if draw:
            self.draw()

    def update_bg(self, label, idx, data, draw=False):
        """Updates the data of the chosen bg."""
        self.update_data(self.bg_images[label][idx], label, data, draw)

    def update_mask(self, label, idx, data, draw=False):
        """Updates the data of the chosen bg."""
        self.update_data(self.strain_masks[label][idx], label, data, draw)

    def update_data(self, subplot, label, data, draw=False):
        """Common data updating method."""
        subplot.set_data(data)
        self.axes[label].relim()
        self.axes[label].autoscale()

        if draw:
            self.draw()

    def update_maps(
        self,
        strain_masks: np.ndarray,
        images: np.ndarray,
        markers: np.ndarray,
        draw=True,
    ):
        """Updates the maps (masks and background data)."""
        rmin, rmax, cmin, cmax = self.limits
        for i in range(9):
            axes = self.axes_lbl[i // 3]
            frame = int(markers[i // 3, i % 3, 0])
            self.update_mask(
                axes,
                i % 3,
                strain_masks[i // 3, frame, rmin : rmax + 1, cmin : cmax + 1],
            )
            self.update_bg(axes, i % 3, images[frame, rmin : rmax + 1, cmin : cmax + 1])

        if draw:
            self.draw()

    def update_one_map(
        self,
        strain_masks: np.ndarray,
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
            axes, idx, strain_masks[component, frame, rmin : rmax + 1, cmin : cmax + 1]
        )
        self.update_bg(axes, idx, images[frame, rmin : rmax + 1, cmin : cmax + 1])
        self.axes[axes].set_ylim(*self.strain_lim[axes])
        self.draw()

    def update_strains(self, strain, draw=True):
        """Updates all strains."""
        x = np.arange(strain.shape[-1])
        for i, label in enumerate(self.axes_lbl):
            self.update_line(label, (x, strain[i]))
        if draw:
            self.draw()

    def update_markers(self, markers, draw=True):
        """Updates the position of all markers in a figure."""
        update_position = self.fig.actions_manager.Markers.update_marker_position

        for i in range(len(self.axes_lbl)):
            update_position(self.marker_artists[2 * i], int(markers[i, 0, 0]))
            update_position(self.marker_artists[2 * i + 1], int(markers[i, 2, 0]))
            self.fixed_markers[i].set_xdata([markers[i, 1, 0], markers[i, 1, 0]])

        if draw:
            self.draw()

    def update_marker(self, marker, data, x, y, position):
        """When a marker moves, mask data should be updated."""
        self.controller.update_strain_marker(
            dataset=self.datasets_var.get(),
            label=self.strain_var.get(),
            region=self.current_region,
            component=self.axes_lbl.index(data.get_label()),
            marker_idx=self.marker_idx[marker.get_label()],
            position=position,
        )
        self.marker_moved(data.get_label(), marker.get_label())
