""" Entry point of StrainMap, creating the main window and variables used along the
whole code. """
import weakref
from typing import Callable

from .gui import *  # noqa: F403,F401
from .gui.base_window_and_task import REGISTERED_VIEWS, Requisites
from .models.strainmap_data_model import StrainMapData
from .models import segmentation
from .models.velocities import calculate_velocities, update_marker, regenerate
from .models.writers import velocity_to_xlsx, strain_to_xlsx, rotation_to_xlsx
from .models.strain import (
    calculate_strain,
    update_strain_es_marker,
    update_marker as update_strain_marker,
)


class StrainMap(object):
    """ StrainMap main window."""

    registered_views = REGISTERED_VIEWS

    def __init__(self):
        from .gui.base_window_and_task import MainWindow

        self.window = MainWindow()
        self.achieved = Requisites.NONE
        self.data = None
        self.review_mode = False
        self.unlock()

    @property
    def progress(self) -> Callable:
        """Convenience property for accessing the progress bar and message ribbon"""
        return self.window.progress

    def run(self):
        """ Runs StrainMap by calling the top window mainloop. """
        self.window.mainloop()

    def unlock(self, requisite=Requisites.NONE):
        """ Adds requisites and loads views. If loaded, they are marked to update."""
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
        """ Removes requisites and updates loaded views."""
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
        """ Updates the data attribute of the views and the widgets depending on it. """
        for view in self.window.views:
            view.update_widgets()

    def load_data_from_folder(self, data_files):
        """Creates a StrainMapData object."""
        self.data = StrainMapData.from_folder(data_files)
        self.lock_toggle(self.data.data_files, Requisites.DATALOADED)
        self.lock(Requisites.SEGMENTED)
        return self.data is not None

    def load_data_from_file(self, strainmap_file):
        """Creates a StrainMapData object."""
        self.data = StrainMapData.from_file(strainmap_file)
        there_are_segments = any(len(i) != 0 for i in self.data.segments.values())
        self.lock_toggle(self.data.data_files, Requisites.DATALOADED)
        self.lock_toggle(there_are_segments, Requisites.SEGMENTED)
        return self.data is not None

    def clear_data(self):
        """Clears the StrainMapData object from the widgets."""
        self.data = None
        self.lock(Requisites.DATALOADED)
        self.lock(Requisites.SEGMENTED)
        self.lock(Requisites.VELOCITIES)

    def add_paths(self, data_files=None):
        self.data.add_paths(data_files)
        there_are_segments = any(len(i) != 0 for i in self.data.segments.values())
        self.lock_toggle(self.data.data_files, Requisites.DATALOADED)
        self.lock_toggle(there_are_segments, Requisites.SEGMENTED)

    def add_h5_file(self, strainmap_file):
        return self.data.add_h5_file(strainmap_file)

    def find_segmentation(self, unlock=True, **kwargs):
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

    def remove_segmentation(self, dataset_name):
        """Clears an existing segmentation."""
        segmentation.remove_segmentation(data=self.data, cine=dataset_name)
        there_are_segments = any(len(i) != 0 for i in self.data.segments.values())
        if not there_are_segments:
            self.lock(Requisites.SEGMENTED)
            self.lock(Requisites.VELOCITIES)

    def calculate_velocities(self, **kwargs):
        """Calculates the velocities based on a given segmentation."""
        calculate_velocities(data=self.data, **kwargs)
        there_are_velocities = (
            sum(len(i) != 0 for i in self.data.velocities.values()) > 1
        )
        self.lock_toggle(there_are_velocities, Requisites.VELOCITIES)

    def regenerate_velocities(self, **kwargs):
        """Calculates the velocities based on a given segmentation."""
        regenerate(data=self.data, **kwargs)
        there_are_velocities = (
            sum(len(i) != 0 for i in self.data.velocities.values()) > 1
        )
        self.lock_toggle(there_are_velocities, Requisites.VELOCITIES)

    def update_marker(self, **kwargs):
        """Updates the markers information after moving one of them."""
        update_marker(data=self.data, **kwargs)
        if kwargs.get("marker_idx", 0) == 3:
            update_strain_es_marker(data=self.data, **kwargs)

    def export_velocity(self, **kwargs):
        """Exports velocity data to a XLSX file."""
        velocity_to_xlsx(data=self.data, **kwargs)

    def calculate_strain(self, **kwargs):
        """Calculates the strain based on the available velocities."""
        return calculate_strain(data=self.data, callback=self.window.progress, **kwargs)

    def update_strain_marker(self, **kwargs):
        """Updates the strain markers information after moving one of them."""
        update_strain_marker(data=self.data, **kwargs)

    def export_strain(self, **kwargs):
        """Exports strain data to a XLSX file."""
        strain_to_xlsx(data=self.data, **kwargs)

    def export_rotation(self, **kwargs):
        """Exports rotation to a XLSX file."""
        rotation_to_xlsx(data=self.data, **kwargs)
