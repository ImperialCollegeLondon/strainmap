""" Entry point of StrainMap, creating the main window and variables used along the
whole code. """
from .gui.base_classes import REGISTERED_VIEWS, MainWindow, Requisites
from .actions import REGISTERED_ACTIONS


class StrainMap(object):
    """ StrainMap main window."""

    def __init__(self):

        self.window = MainWindow()
        self.achieved = Requisites.NONE
        self.data = None
        self.unlock()

    def run(self):
        """ Runs StrainMap by calling the top window mainloop. """
        self.window.mainloop()

    def unlock(self, requisite=Requisites.NONE):
        """ Adds requisites and loads views. """
        self.achieved = self.achieved | requisite

        for view in REGISTERED_VIEWS:
            if Requisites.check(self.achieved, view.requisites):
                self.window.add(view, self.select_actions(view))
                self.window.views[-1].data = self.data

    def lock(self, requisite):
        """ Removes requisites and updates loaded views."""
        self.achieved = self.achieved ^ requisite

        for view in list(self.window.views):
            if not Requisites.check(self.achieved, view.requisites):
                self.window.remove(view)

    def select_actions(self, view):
        """ Selects the actions relevant for the view. """
        actions = {}
        for a in view.actions:
            actions[a] = lambda act=a, **kwargs: REGISTERED_ACTIONS[act](self, **kwargs)
        return actions

    def update_views(self, data):
        """ Updates the data attribute of the views and the widgets depending on it. """
        self.data = data
        for view in self.window.views:
            view.data = data
