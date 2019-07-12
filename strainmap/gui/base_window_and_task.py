""" Views are the GUI part of StrainMap functionality.

Views can be used to load/save data, perform a segmentation, postprocessing... This
module defines StrainMAp main window, a TaskViewBase class that takes care of the front
end related aspect of any StrainMap action and the registers needed to register other
views.
"""
import tkinter as tk
import sys
from abc import ABC, abstractmethod
from enum import Flag, auto
from pathlib import Path
from tkinter import ttk
from typing import Optional, Text, Type, List, Callable
from functools import wraps

from PIL import Image, ImageTk

ICONS_DIRECTORY = Path(__file__).parent / "icons"


class Requisites(Flag):
    NONE = auto()
    DATALOADED = auto()
    SEGMENTED = auto()

    @staticmethod
    def check(achieved, required):
        return required & achieved == required


class TaskViewBase(ABC, ttk.Frame):
    """ Base class for all the views.

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
        button_text: Optional[Text] = None,
        button_image: Optional[Text] = None,
    ):
        super().__init__(root)
        self.__data = None
        self.rowconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)

        if button_image is not None:
            self.image = Image.open(ICONS_DIRECTORY / button_image)
        else:
            self.image = Image.new("RGB", (80, 80))

        self.image = ImageTk.PhotoImage(self.image)

        self.button = ttk.Button(
            master=root.button_frame,
            text=button_text,
            image=self.image,
            padding=5,
            compound="top",
            command=self.tkraise,
        )

        self.control = ttk.Frame(master=self, width=300, name="control")
        self.control.grid(column=0, row=0, sticky=tk.NSEW, padx=10, pady=10)
        self.control.rowconfigure(50, weight=1)
        self.control.grid_propagate(flag=False)

        self.visualise = ttk.Frame(master=self, name="visualise")
        self.visualise.grid(column=1, row=0, sticky=tk.NSEW, padx=10, pady=10)
        self.visualise.columnconfigure(0, weight=1)
        self.visualise.rowconfigure(0, weight=1)
        self.visualise.grid_propagate(flag=False)

    @property
    def data(self):
        return self.__data

    @data.setter
    def data(self, data):
        self.__data = data
        if data is None:
            self.clear_widgets()
        else:
            self.update_widgets()

    @abstractmethod
    def update_widgets(self):
        """ Updates widgets after an update in the data variable. """
        pass

    @abstractmethod
    def clear_widgets(self):
        """ Clear widgets after removing the data. """
        pass


REGISTERED_VIEWS: List[Type[TaskViewBase]] = []
""" Registry of available views. """


def register_view(view: Type[TaskViewBase]) -> Type[TaskViewBase]:
    """Registers a view to make it available to StrainMap. """

    if not issubclass(view, TaskViewBase):
        raise RuntimeError("A view must inherit from TaskViewBase")

    REGISTERED_VIEWS.append(view)

    return view


class MainWindow(tk.Tk):
    """ StrainMap main window."""

    def __init__(self):
        super().__init__()
        self.title("StrainMap")
        self.minsize(1280, 720)
        self.protocol("WM_DELETE_WINDOW", self.__quit)
        self.closed = False

        self.rowconfigure(1, weight=1)
        self.columnconfigure(1, weight=1)

        self.button_frame = ttk.Frame(self, width=150)
        self.button_frame.grid(column=0, row=1, sticky=tk.NS)
        self.button_frame.columnconfigure(0, weight=1)
        self.button_frame.grid_propagate(flag=False)

    @property
    def views(self) -> List[TaskViewBase]:
        return [v for v in self.winfo_children() if isinstance(v, TaskViewBase)]

    @property
    def view_classes(self):
        return [type(v) for v in self.winfo_children() if isinstance(v, TaskViewBase)]

    def add(self, view: Type[TaskViewBase]):
        """ Creates a view if not already created and adds it to the main window."""
        if view not in self.view_classes:
            v = view(root=self)
            v.button.grid(column=0, sticky=(tk.EW, tk.N), padx=10, pady=10)
            v.grid(column=1, row=1, sticky=tk.NSEW)
            v.lower()

    def remove(self, view: Type[TaskViewBase]):
        """ Removes an existing view from the main window."""
        if view in self.views:
            view.button.destroy()
            view.destroy()  # type: ignore

    def mainloop(self, *args):
        """ We initiate the main loop.

        This is a hack found here: http://github.com/matplotlib/matplotlib/issues/9637
        to avoid a crashing that happens when combining certain versions of tcl and
        certain versions of Python. It reveals when scrolling in a Matplotlib plot.
        It seems to be a problem only under MacOS.

        TODO: Investigate if there is a more elegant solution to this.
        """
        if sys.platform == "darwin":
            while not self.closed:
                try:
                    self.update_idletasks()
                    self.update()
                except UnicodeDecodeError:
                    print("Caught Scroll Error")
        else:
            self.mainloop(*args)

    def __quit(self):
        """ Safe quit the program."""
        self.closed = True
        self.quit()


REGISTERED_BINDINGS: dict = {}
""" Registered event bindings."""

REGISTERED_TRIGGERS: list = []
""" Registered event triggers."""

EVENTS: dict = {}
""" Dictionary with the events linked to the control."""


def trigger_event(fun: Optional[Callable] = None, name: Optional[Text] = None):
    """Registers a view method that will trigger an event. """

    if fun is None:
        return lambda x: trigger_event(x, name=name)

    name = name if name else fun.__name__

    @wraps(fun)
    def wrapper(*args, **kwargs):
        params = fun(*args, **kwargs)
        params = params if params else {}
        EVENTS[name](**params)
        return

    REGISTERED_TRIGGERS.append(name)

    return wrapper


def bind_event(fun: Callable, name=None):
    """Registers a control method that will be bound to an event."""

    name = name if name else fun.__name__

    REGISTERED_BINDINGS[name] = fun

    return fun
