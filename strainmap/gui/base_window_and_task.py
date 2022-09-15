""" Views are the GUI part of StrainMap functionality.

Views can be used to load/save data, perform a segmentation, postprocessing... This
module defines StrainMAp main window, a TaskViewBase class that takes care of the front
end related aspect of any StrainMap action and the registers needed to register other
views.
"""
import tkinter as tk
import weakref
from abc import ABC, abstractmethod
from enum import Flag, auto
from pathlib import Path
from tkinter import ttk
from typing import List, Optional, Text, Type

from PIL import Image, ImageTk

from .. import __VERSION__

ICONS_DIRECTORY = Path(__file__).parent / "icons"


class Requisites(Flag):
    NONE = auto()
    DATALOADED = auto()
    SEGMENTED = auto()
    VELOCITIES = auto()

    @staticmethod
    def check(achieved, required):
        return required & achieved == required


class TaskViewBase(ABC, ttk.Frame):
    """Base class for all the views.

    It is derived from ttk.Frame and has two child areas in which add other widgets:

        - control: left frame 300 px wide for control related widgets.
        - visualise: right frame, expandable, for visualising and interacting with data.

    It also provides a button - acting as tab handle - featuring an image and a text.

    In the control frame, Widgets located in rows smaller than 50 will be packed
    towards the top of the frame, while those located in those larger will be packed
    towards the bottom.

    The "actions" parameter is a dictionary with the actions that the view is allowed
    to perform that modify the data. These are routed via the StrainMap object.
    """

    requisites = Requisites.NONE

    def __init__(
        self,
        root: tk.Tk,
        controller: weakref.ReferenceType,
        button_text: Optional[Text] = None,
        button_image: Optional[Text] = None,
        button_row: Optional[int] = None,
    ):
        super().__init__(root)
        self.__controller = controller
        self.is_stale = False
        self.button_row = button_row

        if button_image is not None:
            self.image = Image.open(ICONS_DIRECTORY / button_image)
        else:
            self.image = Image.new("RGB", (40, 40))

        self.image = ImageTk.PhotoImage(self.image)

        self.button = ttk.Button(
            master=root.button_frame,
            text=button_text,
            image=self.image,
            compound="top",
            command=self.tkraise,
        )

    def tkraise(self, *args):
        """Brings the frame to the front."""
        super().tkraise()
        self.master.bind("<Control_L><v>", lambda x: None)
        self.controller.progress("")
        if self.is_stale:
            self.update_widgets()
            self.is_stale = False

    @property
    def data(self):
        return self.controller.data

    @property
    def controller(self):
        return self.__controller()

    @abstractmethod
    def update_widgets(self):
        """Updates widgets after an update in the data var."""
        pass

    @abstractmethod
    def clear_widgets(self):
        """Clear widgets after removing the data."""
        pass


REGISTERED_VIEWS: List[Type[TaskViewBase]] = []
""" Registry of available views. """


def register_view(view: Type[TaskViewBase]) -> Type[TaskViewBase]:
    """Registers a view to make it available to StrainMap."""

    if not issubclass(view, TaskViewBase):
        raise RuntimeError("A view must inherit from TaskViewBase")

    REGISTERED_VIEWS.append(view)

    return view


def fixed_map(option, style):
    """Fix for setting text colour for Tkinter 8.6.9
    From: https://core.tcl.tk/tk/info/509cafafae

    Returns the style map for 'option' with any styles starting with ('!disabled',
    '!selected', ...) filtered out. style.map() returns an empty list for missing
    options, so this should be future-safe.

    Solution found here: https://stackoverflow.com/a/57009674/3778792
    """
    return [
        elm
        for elm in style.map("Treeview", query_opt=option)
        if elm[0] != "!disabled !selected"
    ]


class MainWindow(tk.Tk):
    """StrainMap main window."""

    def __init__(self):
        super().__init__()
        style = ttk.Style()
        style.theme_use("clam")
        style.map(
            "Treeview",
            foreground=fixed_map("foreground", style),
            background=fixed_map("background", style),
        )
        style.configure("TLabelframe", borderwidth=0)
        style.configure("TProgressbar", foreground="#f8d568", background="#f8d568")

        self.title("StrainMap")
        self.minsize(1280, 720)
        self.geometry(f"{0}x{1}+0+0".format(1280, 720))
        self.protocol("WM_DELETE_WINDOW", self.__quit)
        self.closed = False

        self.rowconfigure(1, weight=1)
        self.columnconfigure(1, weight=1)

        self.button_frame = ttk.Frame(self, width=90)
        self.button_frame.grid(column=0, row=1, sticky=tk.NS)
        self.button_frame.columnconfigure(0, weight=1)
        self.button_frame.rowconfigure(50, weight=1)
        self.button_frame.grid_propagate(flag=False)

        self.bar_var = tk.DoubleVar(value=0)
        self.msg_var = tk.StringVar(value="")
        self.msg_frame = ttk.Frame(self)
        self.msg = ttk.Label(
            self.msg_frame, textvariable=self.msg_var, relief=tk.SUNKEN
        )
        version = ttk.Label(self.msg_frame, text=f"v{__VERSION__}")
        self.bar = ttk.Progressbar(
            self.msg_frame,
            orient="horizontal",
            mode="determinate",
            variable=self.bar_var,
            maximum=1,
            length=300,
        )
        self.msg_frame.grid(column=0, row=99, columnspan=2, sticky=tk.EW)
        self.bar.grid(column=0, row=0, sticky=tk.EW, padx=5, pady=5)
        self.msg.grid(column=1, row=0, sticky=tk.EW, padx=5, pady=5)
        self.msg_frame.columnconfigure(1, weight=1)
        version.grid(column=2, row=0, sticky=tk.EW, padx=5, pady=5)

    @property
    def views(self) -> List[TaskViewBase]:
        return [v for v in self.winfo_children() if isinstance(v, TaskViewBase)]

    @property
    def view_classes(self):
        return [type(v) for v in self.winfo_children() if isinstance(v, TaskViewBase)]

    def add(self, view: Type[TaskViewBase], controller):
        """Creates a view if not already created and adds it to the main window."""
        if view not in self.view_classes:
            v = view(root=self, controller=controller)
            if v.button_row is not None:
                v.button.grid(
                    column=0, row=v.button_row, sticky=(tk.EW, tk.N), padx=5, pady=5
                )
            else:
                v.button.grid(column=0, sticky=(tk.EW, tk.N), padx=5, pady=5)
            v.grid(column=1, row=1, sticky=tk.NSEW)
            v.lower()

    def remove(self, view: Type[TaskViewBase]):
        """Removes an existing view from the main window."""
        if view in self.views:
            view.button.destroy()
            view.destroy()  # type: ignore

    def mainloop(self, *args):
        """We initiate the main loop.

        This is a hack found here: http://github.com/matplotlib/matplotlib/issues/9637
        to avoid a crashing that happens when combining certain versions of tcl and
        certain versions of Python. It reveals when scrolling in a Matplotlib plot.
        It seems to be a problem only under MacOS.

        TODO: Investigate if there is a more elegant solution to this.
        """

        while not self.closed:
            try:
                self.update_idletasks()
                self.update()
            except UnicodeDecodeError:
                print("Caught Scroll Error")

    def __quit(self):
        """Safe quit the program."""
        self._stop_animations()
        self.closed = True
        self.quit()

    def _stop_animations(self):
        """Stop any animation that is running, if any."""
        for view in self.views:
            if hasattr(view, "stop_animation"):
                view.stop_animation()

    def progress(self, msg: str, value: float = 0):
        """Prints a message in the lower ribbon and moves the progress bar.

        To actually make any changes, it forces an update of the GUI."""
        self.msg_var.set(value=msg)
        self.bar_var.set(value=min(1.0, value))
        self.update_idletasks()
        self.update()
