[![Test and build](https://github.com/ImperialCollegeLondon/strainmap/actions/workflows/ci.yml/badge.svg)](https://github.com/ImperialCollegeLondon/strainmap/actions/workflows/ci.yml)
[![GitHub
Pages](https://github.com/ImperialCollegeLondon/strainmap/actions/workflows/docs.yml/badge.svg)](https://imperialcollegelondon.github.io/strainmap/)
[![PyPI version shields.io](https://img.shields.io/pypi/v/strainmap.svg)](https://pypi.python.org/pypi/strainmap/)
[![PyPI status](https://img.shields.io/pypi/status/strainmap.svg)](https://pypi.python.org/pypi/strainmap/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/strainmap.svg)](https://pypi.python.org/pypi/strainmap/)
[![PyPI license](https://img.shields.io/pypi/l/strainmap.svg)](https://pypi.python.org/pypi/strainmap/)

# StrainMap

Copyright (c) 2022, Imperial College London
All rights reserved.

StrainMap provides a user-friendly and efficient way to analyse MRI data acquired with a
novel, high temporal and spatial resolution velocity-encoded MRI technique suitable for
regional strain analysis in a short breath-hold. These images include magnitude and
phase components.

The segmentation stage lets the user select the inner and outer
walls of the heart. This needs to be done for all images taken over a heartbeat and for
as many slices (cross-sections of the heart) as available. The process can be manual –
very long – or assisted by several machine learning technologies such as snakes
segmentation or a deep neural network. The segmented heart, together with the phase
information can be used in the next stage to extract information of the instantaneous,
spatially-resolved velocity of the myocardium during the heartbeat in the form of
velocity curves ad heatmaps. All this information can be exported for further analysis
elsewhere.

## Installation

### Recommended way 

The recommended way for end users to access and use the tool is via `pipx`:

1. Install and configure [`pipx`](https://pypa.github.io/pipx/) following the
   instructions appropriate for your operative system. Make sure it works well before
   moving on.
2. Install StrainMap with `pipx install strainmap`.It might take a
   while to complete, but afterwards updates should be pretty fast.
3. To run StrainMap just open a terminal and execute `strainmap`. You might want to
   create a shortcut for this command in the desktop, for convenience.

Whenever there is a new version of StrainMap, just run `pipx upgrade strainmap` and
it will be downloaded and installed with no fuss.

### Use a StrainMap executable

Alternatively, you can download from the [release
page](https://github.com/ImperialCollegeLondon/strainmap/releases) the self-contained
executable corresponding to the version you are interested in. Bear in mind that these
executables contain StrainMap and *all its dependencies*, meaning that each download can
be, potentially, very large.

## For developers

This installation instructions assume the following pre-requisites:

- Python >=3.8
- [Poetry](https://python-poetry.org/) >= 1.11
- Git

If these are already installed and the path correctly configured, the following should download the last version of StrainMap, create and activate a virtual environment, install all StrainMap dependencies and, finally, install StrainMap itself in development mode. 

```bash
git clone https://github.com/ImperialCollegeLondon/strainmap.git
cd strainmap
poetry install
```

To use StrainMap simply run:

```bash
poetry run python -m strainmap
```
