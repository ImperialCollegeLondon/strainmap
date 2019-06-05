import tkinter as tk
from tkinter import ttk

import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from ..base_classes import TabBase, register_tab


@register_tab
class Animation(TabBase):

    pre_requisites = {"data_loaded"}

    def __init__(self, root):

        super().__init__(root, tab_text="Animation")

        self.fig = Figure()
        self.ax = self.fig.add_subplot()

        self.series_types_var = tk.StringVar(
            value=sorted(self.data.data_files.keys())[0]
        )
        self.variables_var = tk.StringVar(value="MagZ")
        self.run_animation_button = None
        self.anim = False

    def create_tab_contents(self):
        """ Creates the contents of the tab. """
        for i in range(3):
            self.tab.columnconfigure(i, weight=1)
        self.tab.rowconfigure(1, weight=1)

        series = ttk.Labelframe(self.tab, text="Available series:")
        series.grid(column=0, row=0, sticky=(tk.EW, tk.N), padx=5, pady=5)
        series.columnconfigure(0, weight=1)

        ttk.Combobox(
            master=series,
            textvariable=self.series_types_var,
            values=sorted(self.data.data_files.keys()),
            state="enable",
        ).grid(sticky=tk.NSEW, padx=5, pady=5)

        variable = ttk.Labelframe(self.tab, text="Select variable:")
        variable.grid(column=1, row=0, sticky=(tk.EW, tk.N), padx=5, pady=5)
        variable.columnconfigure(0, weight=1)

        ttk.Combobox(
            master=variable,
            textvariable=self.variables_var,
            values=["MagZ", "PhaseZ", "MagX", "PhaseX", "MagY", "PhaseY"],
            state="enable",
        ).grid(sticky=tk.NSEW, padx=5, pady=5)

        self.run_animation_button = ttk.Button(
            master=self.tab,
            text="Launch animation",
            padding=5,
            command=self.launch_animation,
            width=30,
        )
        self.run_animation_button.grid(column=2, row=0, sticky=tk.NSEW, padx=5, pady=5)

        canvas = FigureCanvasTkAgg(self.fig, master=self.tab)
        canvas.draw()
        canvas.get_tk_widget().grid(
            column=0, row=1, columnspan=3, sticky=tk.NSEW, padx=5, pady=5
        )

    def launch_animation(self):
        """ Launch the animation. """

        if self.anim:
            self.ax.clear()
            self.anim._stop()

        series = self.series_types_var.get()
        variable = self.variables_var.get()
        data = self.data.get_images(series, variable)

        i = 0
        im = self.ax.imshow(data[i], animated=True)

        def updatefig(*args):
            nonlocal i
            i += 1
            i = i % len(data)
            im.set_array(data[i])
            return (im,)

        anim_running = True

        def on_click(event):
            nonlocal anim_running
            if anim_running:
                self.anim.event_source.stop()
                anim_running = False
            else:
                self.anim.event_source.start()
                anim_running = True

        self.fig.canvas.mpl_connect("button_press_event", on_click)

        self.anim = animation.FuncAnimation(self.fig, updatefig, interval=20, blit=True)
