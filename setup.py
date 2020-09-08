from setuptools import setup, find_packages

tests_require = [
    "pytest",
    "pytest-cov",
    "pytest-flake8",
    "pytest-mypy",
    "pytest-mock",
    "flake8<3.8.0",
]
pyinstaller = ["pyinstaller"]

setup(
    name="StrainMap",
    version="0.10.7",
    url="https://github.com/imperialcollegelondon/strainmap",
    author="Research Computing Service, Imperial College London",
    author_email="rcs-support@imperial.ac.uk",
    setup_requires=["pytest-runner"],
    install_requires=[
        "setuptools>=49.1.1",
        "pillow",
        "pydicom",
        "scikit-image<=0.15",
        "matplotlib<=3.2.2",
        "numpy",
        "scipy<=1.3.1",
        "nibabel",
        "openpyxl",
        "h5py",
        "natsort",
        "sparse",
        "pandas"
    ],
    package_data={"strainmap.gui": ["icons/*.gif", "icons/CREDITS.md"]},
    packages=find_packages("."),
    tests_require=tests_require,
    extras_require={"dev": tests_require + pyinstaller},
)
