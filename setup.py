#! /usr/bin/env python3
# Copyright 2024 the President and Fellows of Harvard College
# Licensed under the MIT License

from setuptools import setup


def get_long_desc():
    in_preamble = True
    lines = []

    with open("README.md", "rt", encoding="utf8") as f:
        for line in f:
            if in_preamble:
                if line.startswith("<!--pypi-begin-->"):
                    in_preamble = False
            else:
                if line.startswith("<!--pypi-end-->"):
                    break
                else:
                    lines.append(line)

    lines.append(
        """

For more information, including installation instructions, please visit [the
project homepage].

[the project homepage]: https://daschlab.readthedocs.io/
"""
    )
    return "".join(lines)


setup_args = dict(
    name="daschlab",  # cranko project-name
    version="1.0.0",  # cranko project-version
    description="DASCH data analysis",
    long_description=get_long_desc(),
    long_description_content_type="text/markdown",
    author="Peter Williams",
    url="https://github.com/pkgw/daschlab",
    packages=[
        "daschlab",
    ],
    license="MIT",
    include_package_data=True,
    install_requires=[
        "astropy>=6",
        "bokeh>=3.1",
        "dataclasses-json>=0.6",
        "ipykernel>=6",
        "numpy>=1.20",
        "pillow>=10",
        "pycairo>=1.20",
        "pytz>=2024",
        "pywwt>=0.25",
        "requests>=2",
        "scipy>=1.10",
        "toasty>=0.20",
        "tqdm>=4.60",
    ],
    extras_require={
        "docs": [
            "astropy-sphinx-theme",
            "numpydoc",
            "sphinx",
            "sphinx-automodapi",
        ],
    },
)

if __name__ == "__main__":
    setup(**setup_args)
