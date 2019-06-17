""" Views are the GUI part of StrainMap functionality.

Views can be used to load/save data, perform a segmentation, postprocessing... This
module defines StrainMAp main window, a ViewBase class that takes care of the front end
related aspect of any StrainMap action and the registers needed to register other views.
"""
import tkinter as tk
from abc import ABC, abstractmethod
from collections import namedtuple
from enum import Flag, auto
from pathlib import Path
from tkinter import ttk
from typing import Mapping, Optional, Text, Type, Tuple, Any, List

from PIL import Image, ImageTk

ICONS_DIRECTORY = Path(__file__).parent / "icons"

REGISTERED_VIEWS = []
""" Registry of available views. """


class Requisites(Flag):
    NONE = auto()
    DATALOADED = auto()
    SEGMENTED = auto()

    @staticmethod
    def check(achieved, required):
        return required & achieved == required


class ViewBase(ABC, ttk.Frame):
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
    actions: Tuple[Any, ...] = ()

    def __init__(
        self,
        root: tk.Tk,
        actions: Mapping = None,
        button_text: Optional[Text] = None,
        button_image: Optional[Text] = None,
    ):
        super().__init__(root)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)

        self.image = None
        if button_image is not None:
            self.image = ImageTk.PhotoImage(Image.open(ICONS_DIRECTORY / button_image))

        self.button = ttk.Button(
            master=root.button_frame,
            text=button_text,
            image=self.image,
            padding=5,
            compound="top",
            command=self.tkraise,
        )

        self.control = ttk.Frame(master=self, width=300)
        self.control.grid(column=0, row=0, sticky=tk.NSEW, padx=5, pady=5)
        self.control.rowconfigure(50, weight=1)
        self.control.grid_propagate(flag=False)

        self.visualise = ttk.Frame(master=self)
        self.visualise.grid(column=1, row=0, sticky=tk.NSEW, padx=5, pady=5)
        self.visualise.columnconfigure(0, weight=1)
        self.visualise.rowconfigure(0, weight=1)
        self.visualise.grid_propagate(flag=False)

        if isinstance(actions, Mapping):
            self.actions = namedtuple("Actions", actions.keys())(**actions)

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


def register_view(view: Type[ViewBase]) -> Type[ViewBase]:
    """Registers a view to make it available to StrainMap. """

    if not issubclass(view, ViewBase):
        raise RuntimeError("A view must inherit from ViewBase")

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
    def views(self) -> List[ViewBase]:
        return [v for v in self.winfo_children() if isinstance(v, ViewBase)]

    @property
    def view_classes(self):
        return [type(v) for v in self.winfo_children() if isinstance(v, ViewBase)]

    def add(self, view: Type[ViewBase], actions: Mapping):
        """ Creates a view if not already created and adds it to the main window."""
        if view not in self.view_classes:
            v = view(root=self, actions=actions)
            v.button.grid(column=0, sticky=(tk.EW, tk.N), padx=5, pady=5)
            v.grid(column=1, row=1, sticky=tk.NSEW, padx=5, pady=5)
            v.lower()

    def remove(self, view: Type[ViewBase]):
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

        while not self.closed:
            try:
                self.update_idletasks()
                self.update()
            except UnicodeDecodeError:
                print("Caught Scroll Error")

    def __quit(self):
        """ Safe quit the program."""
        self.closed = True
        self.quit()
