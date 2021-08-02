from setuptools import find_packages, setup

dev_require = [
    "pytest<6.0.0",
    "pytest-cov",
    "pytest-flake8",
    "pytest-mypy",
    "pytest-mock",
    "flake8<3.8.0",
    "pytest-runner",
    "mypy<0.790",
    "black",
]
pyinstaller = ["pyinstaller==4.3"]

setup(
    name="StrainMap",
    version="1.0.0",
    url="https://github.com/imperialcollegelondon/strainmap",
    author="Research Computing Service, Imperial College London",
    author_email="rcs-support@imperial.ac.uk",
    setup_requires=["pytest-runner"],
    install_requires=[
        "setuptools>=49.1.1",
        "pillow==8.2.0",
        "pydicom==2.1.2",
        "scikit-image==0.18.1",
        "matplotlib==3.3.4",
        "numpy==1.19.5",
        "scipy==1.6.3",
        "nibabel==3.2.1",
        "openpyxl==3.0.7",
        "h5py==3.1.0",
        "natsort==7.1.1",
        "pandas==1.2.4",
        "tensorflow==2.5.0",
        "tensorlayer==2.2.3",
        "opencv-python==4.5.2.52",
        "xarray==0.18.2",
        "python-decouple==3.4",
        "netCDF4==1.5.6",
    ],
    package_data={"strainmap.gui": ["icons/*.gif", "icons/CREDITS.md"]},
    packages=find_packages("."),
    tests_require=dev_require,
    extras_require={"dev": dev_require + pyinstaller},
)
