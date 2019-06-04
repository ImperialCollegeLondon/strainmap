""" Tasks are the way of expanding StrainMap functionality.

Everything is a tasks: load/save data, perform a segmentation, postprocessing... This
module defines a TasKBase class that takes care of the front end related aspect of a
task. This involve:

- create a button for the lateral pane.
- create some controls to choose the options of the tasks.
- create a new notebook tab in which to show the outputs of the task.

The logic of the tasks (e.g. perform the segmentation of an image) has to be implemented
somewhere else.
"""
import tkinter as tk
from abc import ABC, abstractmethod
from tkinter import ttk
from typing import Callable, Optional, Text, Type

from PIL import Image, ImageTk

from .data_structures import StrainMapData
from .defaults import ICONS_DIRECTORY

REGISTERED_TASKS = set()
""" Registry of available tasks. """


class TaskBase(ABC):
    """ Base class for all the tasks. """

    pre_requisites = set()

    def __init__(
        self,
        root: tk.Tk,
        button_text: Optional[Text] = None,
        button_image: Optional[Text] = None,
    ):
        self.data = root.data

        self.unlock_achievement = lambda x: root.unlock_achievement(x)
        self.lock_achievement = lambda x: root.lock_achievement(x)

        self.image = None
        if button_image is not None:
            self.image = ImageTk.PhotoImage(Image.open(ICONS_DIRECTORY / button_image))

        self.button = ttk.Button(
            master=root.button_frame,
            text=button_text,
            image=self.image,
            padding=5,
            compound="top",
            command=self._create_and_rise,
        )
        self.button.grid(column=0, sticky=(tk.EW, tk.N), padx=5, pady=5)

        self.control = ttk.Frame(master=root.control_frame)
        self.control.grid_propagate(flag=False)

        self.controls_created = False

    @abstractmethod
    def create_controls(self):
        """ Creates the control frame widgets. """
        pass

    def _create_and_rise(self):
        """ Creates and rises the control frame to the top stack.

        Widgets located in rows smaller than 50 will be packed towards the top of the
        frame, while those located in those larger will be packed towards the bottom.
        """
        if not self.controls_created:
            self.control.grid(column=0, row=0, sticky=tk.NSEW, padx=5, pady=5)
            self.control.rowconfigure(50, weight=1)
            self.create_controls()
        self.control.tkraise()

    def unload(self):
        """ Unloads the task by 'ungridding' its components. """
        self.button.grid_forget()
        self.control.grid_forget()


def register_task(task_class: Type[TaskBase]) -> TaskBase:
    """Registers a task to make it available to SetrainMap. """

    if not issubclass(task_class, TaskBase):
        raise RuntimeError("A task must inherit from TaskBase")

    REGISTERED_TASKS.add(task_class)

    return task_class
