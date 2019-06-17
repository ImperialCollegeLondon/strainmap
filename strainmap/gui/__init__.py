import glob
from pathlib import Path

views = glob.glob(str(Path(__file__).parent / "*_view.py"))

# Ensures that all views are registered
__all__ = [Path(f).name.strip(".py") for f in views]

from . import *  # noqa: F403,F401
