#!/usr/bin/env bash

# --
# Setup environment

conda create -y -n locust_env python=3.7
conda activate locust_env


# Earth engine requires 'credentials' which can be gotten here: https://developers.google.com/earth-engine/python_install-conda
conda install -c conda-forge earthengine-api

conda install -y -c numpy
conda install -y -c pandas
conda install -y -c conda-forge matplotlib
conda install -y -c anaconda joblib
conda install -y -c conda-forge tifffile
conda install -y -c anaconda argparse
conda install -y -c conda-forge tqdm

pip install -U scikit-learn
pip install geopandas
pip install geohash
pip install shapely
pip install polygon_geohasher
