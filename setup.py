from setuptools import setup

tests_require = ["pytest", "pytest-cov", "pytest-flake8", "pytest-mypy", "pytest-mock"]

setup(
    name="StrainMap",
    version="0.1",
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
        "scipy",
        "nibabel",
        "openpyxl",
        "h5py",
    ],
    tests_require=tests_require,
    extras_require={"dev": tests_require},
)
