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
pyinstaller = ["pyinstaller"]

setup(
    name="StrainMap",
    version="0.11.2",
    url="https://github.com/imperialcollegelondon/strainmap",
    author="Research Computing Service, Imperial College London",
    author_email="rcs-support@imperial.ac.uk",
    setup_requires=["pytest-runner"],
    install_requires=[
        "setuptools>=49.1.1",
        "pillow",
        "pydicom",
        "scikit-image",
        "matplotlib<=3.2.2",
        "numpy",
        "scipy",
        "nibabel",
        "openpyxl",
        "h5py",
        "natsort",
        "pandas<=1.2.4",
        "tensorflow",
        "tensorlayer",
        "toml",
        "opencv-python",
        "xarray",
        "python-decouple",
        "netcdf4",
    ],
    package_data={"strainmap.gui": ["icons/*.gif", "icons/CREDITS.md"]},
    packages=find_packages("."),
    tests_require=dev_require,
    extras_require={"dev": dev_require + pyinstaller},
)
