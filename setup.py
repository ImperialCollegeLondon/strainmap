from setuptools import find_packages, setup

tests_require = [
    "pytest<6.0.0",
    "pytest-cov",
    "pytest-flake8",
    "pytest-mypy",
    "pytest-mock",
    "flake8<3.8.0",
    "pytest-runner",
    "mypy<0.790",
]
pyinstaller = ["pyinstaller"]

setup(
    name="StrainMap",
    version="0.11.1",
    url="https://github.com/imperialcollegelondon/strainmap",
    author="Research Computing Service, Imperial College London",
    author_email="rcs-support@imperial.ac.uk",
    setup_requires=["pytest-runner"],
    install_requires=[
        "setuptools>=49.1.1",
        "pillow",
        "pydicom",
        "scikit-image",
        "matplotlib",
        "numpy",
        "scipy",
        "nibabel",
        "openpyxl",
        "h5py",
        "natsort",
        "pandas",
        "tensorflow",
        "tensorlayer",
        "toml",
        "opencv-python",
        "xarray",
    ],
    package_data={"strainmap.gui": ["icons/*.gif", "icons/CREDITS.md"]},
    packages=find_packages("."),
    tests_require=tests_require,
    extras_require={"dev": tests_require + pyinstaller},
)
