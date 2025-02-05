####===============####
# Filename: setup.py
# Author: kojogyaase
# Project: CVAR - UMaine 
# Created 2024-05-24
# @Copyright (c)  oatomobile  All rights reserved.
# @last-modified 2024-05-26
####===============####

import os
from importlib import util as import_util
from setuptools import find_packages
from setuptools import setup
here = os.path.abspath(os.path.dirname(__file__))

# # Get the long description from the README.mf file
# with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
#   long_description = f.read()

# # Get the version from metadata.
# spec = import_util.spec_from_file_location(
#     "_metadata",
#     "/_metadata.py",
# )
# _metadata = import_util.module_from_spec(spec)
# spec.loader.exec_module(_metadata)
# version = _metadata.__version__

setup(
    name="carla-rl",
    # version=version,
    description=
    "This a tool for RL develpment and Testin in CARLA",
    # long_description=long_description,
    long_description_content_type="text/markdown",
    # url="https://github.com/oatml/oatomobile",
    author="Computer Vision and Robtics Lab @ UMaine",
    author_email="elvis.gyaase@maine.edu",
    license="Apache License, Version 2.0",
    packages=find_packages(include=['vision_rl', 'vision_rl.*']),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
