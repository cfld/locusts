#!/usr/bin/env bash

# --
# Setup environment

conda create -y -n locust_env python=3.7
conda activate locust_env

conda install -y -c numpy
conda install -y -c pandas
conda install matplotlib

pip install -U scikit-learn
pip install geopandas
pip install geohash
pip install shapely
pip install polygon_geohasher
