import logging

import decouple

from strainmap import controller

logging.basicConfig(level=decouple.config("STRAINMAP_LOGGING_LEVEL", "WARNING"))
if __name__ == "__main__":
    controller.StrainMap().run()
