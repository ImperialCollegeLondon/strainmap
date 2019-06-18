from setuptools import setup

setup(
    name="StrainMap",
    version="0.1",
    url="https://github.com/imperialcollegelondon/strainmap",
    author="Research Computing Service, Imperial College London",
    author_email="rcs-support@imperial.ac.uk",
    setup_requires=["pytest-runner"],
    install_requires=["pillow", "pydicom", "matplotlib"],
    tests_require=[
        "pytest",
        "pytest-cov",
        "pytest-flake8",
        "pytest-mypy",
        "pytest-mock",
    ],
)
