from setuptools import setup

setup(
    name="StrainMap",
    version="0.1",
    url="https://github.com/imperialcollegelondon/strainmap",
    author="Research Computing Service, Imperial College London",
    author_email="rcs-support@imperial.ac.uk",
    setup_requires=["pytest-runner"],
    tests_require=["pytest-cov", "pytest-flake8", "pytest-mypy"],
)
