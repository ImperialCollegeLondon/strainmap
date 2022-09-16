import logging

import decouple

from strainmap import controller

logging.basicConfig(level=decouple.config("STRAINMAP_LOGGING_LEVEL", "WARNING"))


def strainmap():
    controller.StrainMap().run()
