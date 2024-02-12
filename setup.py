#! /usr/bin/env python3
# Copyright 2024 the President and Fellows of Harvard College
# Licensed under the MIT License

from setuptools import setup


setup_args = dict(
    name="daschlab",
    version="0.1",
    description="DASCH data analysis",
    long_description="A package for DASCH data analysis",
    long_description_content_type="text/markdown",
    author="DASCH Team",
    url="https://github.com/pkgw/daschlab",
    packages=[
        "daschlab",
    ],
    license="MIT",
    include_package_data=True,
    install_requires=[
        "pywwt>=0.23",
    ],
)

if __name__ == "__main__":
    setup(**setup_args)
