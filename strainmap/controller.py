""" Entry point of StrainMap, creating the main window and variables used along the
whole code. """
import weakref
from typing import Callable

from pubsub import pub

from .exceptions import NoDICOMDataException
from .gui import *  # noqa: F403,F401
from .gui.base_window_and_task import REGISTERED_VIEWS, Requisites
from .models import segmentation
from .models.strainmap_data_model import StrainMapData
from .models.velocities import calculate_velocities, update_markers
from .models.writers import rotation_to_xlsx, velocity_to_xlsx


class StrainMap(object):
    """StrainMap main window."""

    registered_views = REGISTERED_VIEWS

    def __init__(self):
        from .gui.base_window_and_task import MainWindow

        self.window = MainWindow()
        self.achieved = Requisites.NONE
        self.data = None
        self.review_mode = False
        self.add_menu_bar()
        self.unlock()

    @property
    def progress(self) -> Callable:
        """Convenience property for accessing the progress bar and message ribbon"""
        return self.window.progress

    def run(self):
        """Runs StrainMap by calling the top window mainloop."""
        self.window.mainloop()

    def unlock(self, requisite=Requisites.NONE):
        """Adds requisites and loads views. If loaded, they are marked to update."""
        self.achieved = self.achieved | requisite

        for view in self.registered_views:
            if Requisites.check(self.achieved, view.requisites):
                self.window.add(view, weakref.ref(self))

        for view in self.window.views:
            if (
                Requisites.check(requisite, view.requisites)
                and requisite != Requisites.NONE
            ):
                view.is_stale = True

    def lock(self, requisite):
        """Removes requisites and updates loaded views."""
        self.achieved = self.achieved ^ requisite

        for view in self.window.views:
            if not Requisites.check(self.achieved, view.requisites):
                self.window.remove(view)

    def lock_toggle(self, condition, requisite):
        """Conditional lock/unlock of a requisite."""
        if condition:
            self.unlock(requisite)
        else:
            self.lock(requisite)

    def update_views(self):
        """Updates the data attribute of the views and the widgets depending on it."""
        for view in self.window.views:
            view.update_widgets()

    def load_data_from_folder(self, data_files):
        """Creates a StrainMapData object."""
        try:
            self.data = StrainMapData.from_folder(data_files)
            self.lock_toggle(self.data.data_files, Requisites.DATALOADED)
            self.lock(Requisites.SEGMENTED)
        except NoDICOMDataException:
            self.clear_data()

    def load_data_from_file(self, strainmap_file):
        """Creates a StrainMapData object."""
        self.data.from_file(strainmap_file)
        self.lock_toggle(self.data.data_files, Requisites.DATALOADED)
        self.lock_toggle(self.data.segments.shape != (), Requisites.SEGMENTED)
        self.lock_toggle(self.data.cylindrical.shape != (), Requisites.VELOCITIES)
        return self.data is not None

    def clear_data(self):
        """Clears the StrainMapData object from the widgets."""
        self.data = None
        self.lock(Requisites.DATALOADED)
        self.lock(Requisites.SEGMENTED)
        self.lock(Requisites.VELOCITIES)

    def add_file(self, strainmap_file):
        return self.data.add_file(strainmap_file)

    def new_segmentation(self, unlock=True, **kwargs):
        """Runs an automated segmentation routine."""
        segmentation.new_segmentation(data=self.data, **kwargs)
        there_are_segments = self.data.segments.shape != ()
        if there_are_segments and unlock:
            self.unlock(Requisites.SEGMENTED)
        elif not there_are_segments:
            self.lock(Requisites.SEGMENTED)

    def update_segmentation(self, unlock=True, **kwargs):
        """Runs an automated segmentation routine."""
        segmentation.update_segmentation(data=self.data, **kwargs)
        there_are_segments = self.data.segments.shape != ()
        if there_are_segments and unlock:
            self.unlock(Requisites.SEGMENTED)
        elif not there_are_segments:
            self.lock(Requisites.SEGMENTED)

    def update_and_find_next(self, **kwargs):
        """Runs an automated segmentation routine."""
        segmentation.update_and_find_next(data=self.data, **kwargs)

    def remove_segmentation(self, cine):
        """Clears an existing segmentation."""
        segmentation.remove_segmentation(data=self.data, cine=cine)
        if self.data.segments.shape == ():
            self.lock(Requisites.SEGMENTED)
            self.lock(Requisites.VELOCITIES)

    def calculate_velocities(self, **kwargs):
        """Calculates the velocities based on a given segmentation."""
        calculate_velocities(data=self.data, **kwargs)
        self.lock_toggle(self.data.velocities.shape != (), Requisites.VELOCITIES)

    def update_marker(self, **kwargs):
        """Updates the markers information after moving one of them."""
        update_markers(**kwargs)
        self.data.save("markers")

    def export_velocity(self, **kwargs):
        """Exports velocity data to a XLSX file."""
        velocity_to_xlsx(data=self.data, **kwargs)

    def export_rotation(self, **kwargs):
        """Exports rotation to a XLSX file."""
        rotation_to_xlsx(data=self.data, **kwargs)

    def add_menu_bar(self):
        """Add a menu bar to the main window."""
        import tkinter as tk

        try:
            menubar = tk.Menu(self.window)
        except AttributeError:
            from logging import getLogger

            getLogger().warning(
                "Probably just dealing with a mock window in a test. Ignoring."
            )
            return

        file = tk.Menu(menubar)
        file.add_command(
            label="Export data for AI training", command=self.export_for_training
        )
        file.add_command(
            label="Select AI", command=lambda: pub.sendMessage("segmentation.select_ai")
        )
        menubar.add_cascade(label="AI management", menu=file)

        self.window.config(menu=menubar)

    def export_for_training(self):
        """Export data and masks for re-training the AI."""
        import tkinter as tk
        from pathlib import Path
        from tkinter import messagebox

        from .models.writers import export_for_training

        if self.data is None or self.data.segments.shape == ():
            messagebox.showerror(
                title="Missing segmentation!",
                message="There are no labelled (i.e. segmented) data available to "
                "export.",
            )
            return

        path = tk.filedialog.askdirectory(
            title="Select directory to export the data into",
            initialdir=Path("~").expanduser(),
        )
        if path == "":
            return

        export_for_training(Path(path), self.data)
