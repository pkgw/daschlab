#!/bin/bash

set -xeuo pipefail

git config --global user.email "binder@user.com"
git config --global user.name "Binder"

# Other necessary/useful packages

pip install \
  astropy \
  bokeh \
  dataclasses-json \
  ipyevents \
  ipykernel \
  jupyterlab \
  jupyterlab_widgets \
  nbclassic \
  notebook \
  pillow \
  pywwt \
  requests \
  scipy \
  shapely \
  toasty \
  wwt_api_client \
  wwt_data_formats \
  wwt_jupyterlab_extension \
  wwt-kernel-data-relay \
  --user

jupyter nbclassic-extension list
jupyter nbclassic-serverextension list
jupyter labextension list

pip install .
