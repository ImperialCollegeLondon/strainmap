import glob
from pathlib import Path

__all__ = [
    Path(f).name.strip(".py") for f in glob.glob(str(Path(__file__).parent / "*.py"))
]
__all__.remove("__init__")
