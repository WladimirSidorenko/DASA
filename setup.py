#!/usr/bin/env python
# -*- mode: python; coding: utf-8; -*-

##################################################################
# Libraries
from setuptools import setup
from glob import glob
from os import path
import codecs


##################################################################
# Variables and Constants
PWD = path.abspath(path.dirname(__file__))
ENCODING = "utf-8"

with codecs.open(path.join(PWD, "README.rst"), encoding="utf-8") as ifile:
    long_description = ifile.read()

INSTALL_REQUIRES = []
with codecs.open(path.join(PWD, "requirements.txt"),
                 encoding=ENCODING) as ifile:
    for iline in ifile:
        iline = iline.strip()
        if iline:
            INSTALL_REQUIRES.append(iline)

TEST_REQUIRES = []
with codecs.open(path.join(PWD, "test-requirements.txt"),
                 encoding=ENCODING) as ifile:
    for iline in ifile:
        iline = iline.strip()
        if iline:
            TEST_REQUIRES.append(iline)

##################################################################
# setup()
setup(
    name="dasa",
    version="0.1.0a0",
    description=("Discourse-aware sentiment analysis methods."),
    long_description=long_description,
    author="Wladimir Sidorenko (Uladzimir Sidarenka)",
    author_email="wlsidorenko@gmail.com",
    license="MIT",
    url="https://github.com/WladimirSidorenko/DASA",
    include_package_data=True,
    packages=["dasa"],
    package_data={},
    install_requires=INSTALL_REQUIRES,
    dependency_links=[
    ],
    setup_requires=["pytest-runner"],
    tests_require=TEST_REQUIRES,
    provides=["dasa (0.1.0a0)"],
    scripts=glob(path.join("scripts", "dasa*")),
    classifiers=["Development Status :: 3 - Alpha",
                 "Environment :: Console",
                 "Intended Audience :: Science/Research",
                 "License :: OSI Approved :: MIT License",
                 "Natural Language :: German",
                 "Operating System :: Unix",
                 "Operating System :: MacOS",
                 "Programming Language :: Python :: 2",
                 "Programming Language :: Python :: 2.6",
                 "Programming Language :: Python :: 2.7",
                 "Programming Language :: Python :: 3",
                 "Topic :: Text Processing :: Linguistic"],
    keywords="sentiment-analysis discourse NLP linguistics")
