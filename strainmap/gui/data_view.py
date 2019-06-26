import tkinter as tk
import tkinter.filedialog
from tkinter import messagebox, ttk

import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from .base_window_and_task import TaskViewBase, register_view, trigger_event


@register_view
class DataTaskView(TaskViewBase):
    def __init__(self, root):

        super().__init__(root, button_text="Data", button_image="save.gif")

        # Control-related attributes
        self.data_folder = tk.StringVar()
        self.output_file = tk.StringVar()
        self.dataselector = None
        self.create_controls()

        # Visualization-related attributes
        self.notebook = None
        self.treeview = None
        self.time_step = None
        self.fig = None
        self.ax = None
        self.series_types_var = tk.StringVar()
        self.variables_var = tk.StringVar(value="MagZ")
        self.anim = False

    def create_controls(self) -> None:
        """ Creates the controls to load and save data."""

        self.control.columnconfigure(0, weight=1)

        ttk.Button(
            master=self.control,
            name="chooseDataFolder",
            text="New analysis from data folder",
            command=self.load_data,
        ).grid(sticky=tk.NSEW)

        ttk.Button(
            master=self.control,
            name="openStrainMapFile",
            text="Resume analysis from StrainMap file",
            command=self.open_existing_file,
        ).grid(sticky=tk.NSEW)

        ttk.Button(
            master=self.control,
            name="chooseOutputFile",
            text="Save analysis as...",
            command=self.select_output_file,
            state="disabled",
        ).grid(sticky=tk.NSEW)

        # Current data folder widgets -----------
        info = ttk.Labelframe(self.control, text="Current data folder:")
        info.grid(row=60, sticky=(tk.EW, tk.N), pady=5)
        info.columnconfigure(0, weight=1)

        ttk.Entry(
            master=info,
            textvariable=self.data_folder,
            state="disabled",
            justify="center",
        ).grid(sticky=tk.NSEW)

        # Current output file widgets -----------
        info = ttk.Labelframe(self.control, text="Current output file:")
        info.grid(sticky=(tk.EW, tk.N))
        info.columnconfigure(0, weight=1)

        ttk.Entry(
            master=info,
            textvariable=self.output_file,
            state="disabled",
            justify="center",
        ).grid(sticky=tk.NSEW)

        # Clear data widget -------------
        ttk.Button(
            master=self.control,
            name="clearAllData",
            text="Clear all data",
            command=self.clear_data,
        ).grid(sticky=tk.NSEW)

    def create_data_selector(self):
        """ Creates the selector for the data. """
        self.dataselector = ttk.LabelFrame(
            self.control, name="dataSelector", text="Data selector:"
        )
        self.dataselector.grid(row=40, sticky=(tk.EW, tk.S), pady=5)
        self.dataselector.columnconfigure(0, weight=1)

        slice = ttk.Labelframe(self.dataselector, text="Slice:")
        slice.grid(column=0, row=0, sticky=(tk.EW, tk.N), padx=5, pady=5)
        slice.columnconfigure(0, weight=1)

        values = sorted(self.data.data_files.keys())
        self.series_types_var.set(values[0])
        for v in values:
            ttk.Radiobutton(
                master=slice,
                variable=self.series_types_var,
                text=v,
                value=v,
                command=self.update_visualization,
            ).grid(column=0, sticky=tk.NSEW, padx=5, pady=5)

        maps = ttk.Labelframe(self.dataselector, text="Map:")
        maps.grid(column=0, row=1, sticky=(tk.EW, tk.N), padx=5, pady=5)
        maps.columnconfigure(0, weight=1)

        for v in ["MagZ", "PhaseZ", "MagX", "PhaseX", "MagY", "PhaseY"]:
            ttk.Radiobutton(
                master=maps,
                variable=self.variables_var,
                text=v,
                value=v,
                command=self.update_visualization,
            ).grid(sticky=tk.NSEW, padx=5, pady=5)

        timestep = ttk.Labelframe(self.dataselector, name="timeStep", text="Time step:")
        timestep.grid(column=0, row=2, sticky=(tk.EW, tk.N), padx=5, pady=5)
        timestep.columnconfigure(0, weight=1)
        timestep.columnconfigure(1, weight=1)

        values = [*range(len(self.data.data_files[values[0]]["MagZ"]))]
        self.time_step = ttk.Spinbox(
            master=timestep,
            name="stepFilenum",
            command=lambda: self.update_visualization(step_changed=True),
            values=values,
            state="readonly",
        )
        self.time_step.grid(column=0, row=0, sticky=tk.NSEW)
        self.time_step.set(values[0])

        ttk.Button(
            master=timestep,
            name="playAnimation",
            command=lambda: self.update_visualization(play_stop=True),
            text="> / ||",
        ).grid(column=1, row=0, sticky=tk.NSEW)

    def create_data_viewer(self):
        """ Creates the viewer for the data, including the animation and the DICOM. """
        self.notebook = ttk.Notebook(self.visualise, name="notebook")
        self.notebook.grid(sticky=tk.NSEW)
        self.notebook.columnconfigure(0, weight=1)
        self.notebook.rowconfigure(0, weight=1)

        self.notebook.add(self.create_animation_viewer(), text="Animation")
        self.notebook.add(self.create_dicom_viewer(), text="DICOM Data")

    def create_animation_viewer(self):
        """ Creates the animation plot area. """

        self.fig = Figure()
        self.ax = self.fig.add_subplot()

        animation_frame = ttk.Frame(self.notebook)
        animation_frame.columnconfigure(0, weight=1)
        animation_frame.rowconfigure(0, weight=1)

        canvas = FigureCanvasTkAgg(self.fig, master=animation_frame)
        canvas.draw()
        canvas.get_tk_widget().grid(sticky=tk.NSEW, padx=5, pady=5)

        return animation_frame

    def create_dicom_viewer(self):

        dicom_frame = ttk.Frame(self.notebook)
        dicom_frame.columnconfigure(0, weight=1)
        dicom_frame.rowconfigure(0, weight=1)

        self.treeview = ttk.Treeview(dicom_frame, selectmode="browse")
        vsb = ttk.Scrollbar(
            dicom_frame, orient="vertical", command=self.treeview.yview()
        )
        self.treeview.configure(yscrollcommand=vsb.set)
        self.treeview.grid(column=0, row=0, sticky=tk.NSEW, padx=5, pady=5)
        vsb.grid(column=1, row=0, sticky=tk.NSEW)

        self.treeview["columns"] = ("1", "2")
        self.treeview["show"] = "headings"
        self.treeview.column("1", width=300, stretch=False)
        self.treeview.heading("1", text="Tags")
        self.treeview.heading("2", text="Values")

        return dicom_frame

    @trigger_event
    def load_data(self):

        path = tk.filedialog.askdirectory(title="Select DATA directory")

        output = {}
        if path != "":
            self.data_folder.set(path)
            output = {"data_files": path}

        return output

    @trigger_event
    def clear_data(self):
        """ Clears all data from memory."""
        clear = messagebox.askokcancel(
            "Warning!",
            "This will erase all data from memory\nDo you want to continue?",
            icon="warning",
        )
        return {"clear": clear}

    def open_existing_file(self):
        """ Opens an existing StrainMap file."""
        messagebox.showinfo(message="This functionality is not implemented, yet.")
        self.output_file.set("")

    def select_output_file(self):
        """ Selects an output file in which to store the analysis."""
        messagebox.showinfo(message="This functionality is not implemented, yet.")
        self.output_file.set("")

    def update_visualization(self, step_changed=False, play_stop=False):
        """ Updates the visualization whenever the data selected changes. """
        series = self.series_types_var.get()
        variable = self.variables_var.get()
        data = self.data.get_images(series, variable)
        self.time_step["values"] = [*range(len(data))]
        idx = int(self.time_step.get())

        if self.notebook.tab(self.notebook.select(), "text") == "Animation":
            if step_changed:
                self.plot_step(data, idx)
            else:
                self.play_animation(data, idx, play_stop)
        elif self.notebook.tab(self.notebook.select(), "text") == "DICOM Data":
            self.update_tree(series, variable, idx)

    def plot_step(self, data, idx):
        """ Static plot of a given image. Stops the running simulation, if any. """
        if self.anim and self.anim.event_source:
            self.ax.clear()
            self.anim._stop()
        else:
            self.ax.clear()

        self.ax.imshow(data[idx])
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)
        plt.tight_layout()

        self.fig.canvas.draw()

    def play_animation(self, data, idx, play_stop=False):
        """ Plays the animation. """

        if self.anim and self.anim.event_source:
            self.ax.clear()
            self.anim._stop()
            if play_stop:
                return

        im = self.ax.imshow(data[idx], animated=True)
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)
        plt.tight_layout()

        def updatefig(*args):
            i = args[0] % len(data)
            im.set_array(data[i])
            return (im,)

        anim_running = True

        def on_click(event):
            nonlocal anim_running
            if self.anim and self.anim.event_source:
                if anim_running:
                    self.anim.event_source.stop()
                else:
                    self.anim.event_source.start()
                anim_running = not anim_running

        self.fig.canvas.mpl_connect("button_press_event", on_click)
        self.anim = animation.FuncAnimation(self.fig, updatefig, interval=20, blit=True)

    def update_tree(self, series, variable, idx):
        """ Updates the treeview with data from the selected options. """

        self.treeview.delete(*self.treeview.get_children())

        data = self.data.read_dicom_file_tags(series, variable, idx)
        for d in data:
            self.treeview.insert("", tk.END, values=(d, data.get(d)))

    def update_widgets(self):
        """ Updates widgets after an update in the data variable. """
        self.nametowidget("control.chooseOutputFile")["state"] = "enable"
        self.create_data_selector()
        self.create_data_viewer()
        self.update_visualization()

    def clear_widgets(self):
        """ Clear widgets after removing the data. """
        self.data_folder.set("")
        self.output_file.set("")
        self.nametowidget("control.chooseOutputFile")["state"] = "disabled"

        if self.notebook:
            self.dataselector.grid_remove()
            self.notebook.destroy()
            self.notebook = None
            self.dataselector = None
