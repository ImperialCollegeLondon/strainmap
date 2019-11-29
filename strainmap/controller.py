""" Entry point of StrainMap, creating the main window and variables used along the
whole code. """
import weakref

from .gui import *  # noqa: F403,F401
from .gui.base_window_and_task import REGISTERED_VIEWS, Requisites
from .models.strainmap_data_model import StrainMapData
from .models import quick_segmentation
from .models.velocities import calculate_velocities, update_marker
from .models.writers import velocity_to_xlsx


class StrainMap(object):
    """ StrainMap main window."""

    registered_views = REGISTERED_VIEWS

    def __init__(self):
        from .gui.base_window_and_task import MainWindow

        self.window = MainWindow()
        self.achieved = Requisites.NONE
        self.data = None
        self.unlock()

    def run(self):
        """ Runs StrainMap by calling the top window mainloop. """
        self.window.mainloop()

    def unlock(self, requisite=Requisites.NONE):
        """ Adds requisites and loads views. If loaded, they are marked to update."""
        self.achieved = self.achieved | requisite

        for view in self.registered_views:
            if Requisites.check(self.achieved, view.requisites):
                self.window.add(view, weakref.ref(self))

        for view in list(self.window.views):
            if (
                Requisites.check(self.achieved, view.requisites)
                and view.requisites != Requisites.NONE
            ):
                view.to_update = True

    def lock(self, requisite):
        """ Removes requisites and updates loaded views."""
        self.achieved = self.achieved ^ requisite

        for view in list(self.window.views):
            if not Requisites.check(self.achieved, view.requisites):
                self.window.remove(view)

    def lock_unlock(self, condition, requisite):
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
        self.lock_unlock(self.data.data_files, Requisites.DATALOADED)
        return self.data is not None

    def load_data_from_file(self, strainmap_file):
        """Creates a StrainMapData object."""
        self.data = StrainMapData.from_file(strainmap_file)
        there_are_segments = any(len(i) != 0 for i in self.data.segments.values())
        self.lock_unlock(self.data.data_files, Requisites.DATALOADED)
        self.lock_unlock(there_are_segments, Requisites.SEGMENTED)
        return self.data is not None

    def clear_data(self):
        """Clears the StrainMapData object from the widgets."""
        self.data = None
        self.lock(Requisites.DATALOADED)
        self.lock(Requisites.SEGMENTED)

    def add_paths(self, data_files=None, bg_files=None):
        self.data.add_paths(data_files, bg_files)
        there_are_segments = any(len(i) != 0 for i in self.data.segments.values())
        self.lock_unlock(self.data.data_files, Requisites.DATALOADED)
        self.lock_unlock(there_are_segments, Requisites.SEGMENTED)

    def add_h5_file(self, strainmap_file):
        return self.data.add_h5_file(strainmap_file)

    def find_segmentation(self, unlock=True, **kwargs):
        """Runs an automated segmentation routine."""
        quick_segmentation.find_segmentation(data=self.data, **kwargs)
        there_are_segments = any(len(i) != 0 for i in self.data.segments.values())
        if there_are_segments and unlock:
            self.unlock(Requisites.SEGMENTED)
        elif not there_are_segments:
            self.lock(Requisites.SEGMENTED)

    def update_segmentation(self, unlock=True, **kwargs):
        """Runs an automated segmentation routine."""
        quick_segmentation.update_segmentation(data=self.data, **kwargs)
        there_are_segments = any(len(i) != 0 for i in self.data.segments.values())
        if there_are_segments and unlock:
            self.unlock(Requisites.SEGMENTED)
        elif not there_are_segments:
            self.lock(Requisites.SEGMENTED)

    def update_and_find_next(self, **kwargs):
        """Runs an automated segmentation routine."""
        quick_segmentation.update_and_find_next(data=self.data, **kwargs)

    def clear_segmentation(self, dataset_name):
        """Clears an existing segmentation."""
        quick_segmentation.clear_segmentation(data=self.data, dataset_name=dataset_name)
        there_are_segments = any(len(i) != 0 for i in self.data.segments.values())
        if not there_are_segments:
            self.lock(Requisites.SEGMENTED)

    def calculate_velocities(self, **kwargs):
        """Calculates the velocities based on a given segmentation."""
        calculate_velocities(data=self.data, **kwargs)

    def update_marker(self, **kwargs):
        """Updates the markers information after moving one of them."""
        update_marker(data=self.data, **kwargs)

    def export_velocity(self, **kwargs):
        """Exports velocity data to a XLSX file."""
        velocity_to_xlsx(data=self.data, **kwargs)
