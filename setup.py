
import subprocess
import re
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name = "boostrsa",
    version = "0.0.1dev6",
    author = "seojin",
    author_email = "pures1@hanyang.ac.kr",
    description = "This is toolbox for boosting calculation speed using GPU",
    long_description = long_description,
    long_description_content_type="text/markdown",
    url = "https://github.com/SeojinYoon/boostrsa.git",
    packages = find_packages(where = "src"),
    package_dir = {"": "src"},
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires = [
        "numpy",
        "pandas",
        "tqdm",
    ]
)
