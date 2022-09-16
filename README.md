[![Test and build](https://github.com/ImperialCollegeLondon/strainmap/actions/workflows/ci.yml/badge.svg)](https://github.com/ImperialCollegeLondon/strainmap/actions/workflows/ci.yml)
[![GitHub Pages](https://github.com/ImperialCollegeLondon/strainmap/actions/workflows/docs.yml/badge.svg)](https://imperialcollegelondon.github.io/strainmap/)

# StrainMap

Copyright (c) 2022, Imperial College London
All rights reserved.

This installation instructions assume the following pre-requisites:

- Python 3.7
- Git

If these are already installed and the path correctly configured, the following should download the last version of StrainMap, create and activate a virtual environment, install all StrainMap dependencies and, finally, install StrainMap itself in development mode. 

```bash
git clone https://github.com/ImperialCollegeLondon/strainmap.git
cd strainmap
python3 -mvenv venv
. venv/bin/activate
pip install -U setuptools wheel pip
pip install -e .
```

To use the StrainMap GUI, simply run within the virtual environment:

```bash
python -m strainmap
```

Whenever the virtual environment is in use, it should be possible to import StrainMap as with any other module with `import strainmap`.
