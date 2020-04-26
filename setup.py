from setuptools import setup, find_packages

tests_require = ["pytest", "pytest-cov", "pytest-flake8", "pytest-mypy", "pytest-mock"]
pyinstaller = ["pyinstaller"]

setup(
    name="StrainMap",
    version="0.2.0",
    url="https://github.com/imperialcollegelondon/strainmap",
    author="Research Computing Service, Imperial College London",
    author_email="rcs-support@imperial.ac.uk",
    setup_requires=["pytest-runner"],
    install_requires=[
        "pillow",
        "pydicom",
        "scikit-image",
        "matplotlib",
        "numpy",
        "scipy<=1.3.1",
        "nibabel",
        "openpyxl",
        "h5py",
        "natsort",
        "sparse",
    ],
    package_data={"strainmap.gui": ["icons/*.gif", "icons/CREDITS.md"]},
    packages=find_packages("."),
    tests_require=tests_require,
    extras_require={"dev": tests_require + pyinstaller},
)
