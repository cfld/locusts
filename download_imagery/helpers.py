#!/usr/bin/env python

"""
    helpers.py
"""
import os
import numpy as np
from zipfile import ZipFile
from tifffile import imread
from joblib import Parallel, delayed
from shapely import geometry
from itertools import product
from polygon_geohasher.polygon_geohasher import polygon_to_geohashes, geohashes_to_polygon
import ee
import backoff
import urllib
from urllib.request import urlretrieve


# --
# Generic

def parmap(fn, x, seed=False, **kwargs):
    if seed:
        jobs = [delayed(fn)(xx, seed=123 * i) for i, xx in enumerate(x)]
    else:
        jobs = [delayed(fn)(xx) for xx in x]

    return Parallel(**kwargs)(jobs)


@backoff.on_exception(backoff.constant, urllib.error.HTTPError, max_tries=4, interval=2)
def safe_urlretrieve(url, outpath):
    _ = urlretrieve(url, outpath)


# --
# Geohash

GEOHASH_CHARACTERS = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "b", "c", "d", "e", "f", "g",
    "h", "j", "k", "m", "n", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
]


def _geohash_descend(x):
    return [x + c for c in GEOHASH_CHARACTERS]


def geohash_descend(x, level=6):
    out = [x]
    while len(out[0]) < level:
        out = sum([[xx + c for c in GEOHASH_CHARACTERS] for xx in out], [])

    return out


def polygon2geohash(polygon, precision=6, coarse_precision=None, inner=True):
    polygon = geometry.shape(polygon.toGeoJSON())

    if coarse_precision is None:
        geohashes = polygon_to_geohashes(geojson, precision=precision, inner=inner)
    else:
        geohashes = polygon_to_geohashes(polygon, precision=coarse_precision, inner=inner)

        curr_precision = coarse_precision
        while curr_precision < precision:
            geohashes = [a + b for a, b in product(geohashes, GEOHASH_CHARACTERS)]
            curr_precision += 1

    return sorted(list(geohashes))


def geohashes2cell(geohashes):
    cells = [geohashes_to_polygon([h]) for h in geohashes]
    cells = [ee.Geometry(geometry.mapping(c)) for c in cells]
    return cells


# --
# Download

@backoff.on_exception(backoff.constant, urllib.error.HTTPError, max_tries=4, interval=2)
def safe_urlretrieve(url, outpath):
    _ = urlretrieve(url, outpath)


# --
# Sentinel
sentinel_channels = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12', 'QA60']


def maskS2clouds(image):
    qa = image.select('QA60')

    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11

    mask = qa.bitwiseAnd(cloudBitMask).eq(0)
    mask = mask.bitwiseAnd(cirrusBitMask).eq(0)

    return image.updateMask(mask)


def zip2numpy_sentinel(inpath):
    inpath = inpath.strip()
    outpath = inpath.replace('.zip', '.npy')
    ghash = os.path.basename(inpath).split('.')[0]
    ghash = ghash[0:5]
    with ZipFile(inpath) as handle:
        tile = np.stack([
            imread(handle.open(f'{ghash}.B1.tif')),
            imread(handle.open(f'{ghash}.B2.tif')),
            imread(handle.open(f'{ghash}.B3.tif')),
            imread(handle.open(f'{ghash}.B4.tif')),
            imread(handle.open(f'{ghash}.B5.tif')),
            imread(handle.open(f'{ghash}.B6.tif')),
            imread(handle.open(f'{ghash}.B7.tif')),
            imread(handle.open(f'{ghash}.B8.tif')),
            imread(handle.open(f'{ghash}.B8A.tif')),
            imread(handle.open(f'{ghash}.B9.tif')),
            #imread(handle.open(f'{ghash}.B10.tif')),
            imread(handle.open(f'{ghash}.B11.tif')),
            imread(handle.open(f'{ghash}.B12.tif')),
            imread(handle.open(f'{ghash}.QA60.tif'))
        ])
        tile = tile.astype(np.int32)

        np.save(outpath, tile)
        os.remove(inpath)


def get_one_sentinel(loc, outdir):
    outpath = os.path.join(outdir, loc['geohash'] + '_' + str(loc['date_start']) + '.zip')

    cell    = geohashes2cell([loc['geohash']])[0]
    mosaic  = (
        ee.ImageCollection('COPERNICUS/S2')
            .select(sentinel_channels)
            .filterDate(loc['date_start'], loc['date_end'])
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)).map(maskS2clouds)  # Apply cloud mask
    )
    mosaic = mosaic.sort('system:index', opt_ascending=False).mosaic()
    try:
        url = mosaic.clip(cell).getDownloadURL(params={"name": loc['geohash'], "crs": "EPSG:4326", "scale": 10})
        _ = safe_urlretrieve(url, outpath)
    except:
        pass


glads_bands = ['ECanop_tavg',  #Canopy water evaporation
               'Evap_tavg',  #Evapotranspiration
               'PotEvap_tavg', #Potential evaporation rate
               'Psurf_f_inst',  #Pressure
               'Qair_f_inst',  #Specific humidity
               'Qg_tavg',  #Heat flux
               'Rainf_f_tavg',  #Total precipitation rate
               'RootMoist_inst',  #Root zone soil moisture
               'SoilMoi0_10cm_inst',  #Soil moisture
               'SoilMoi10_40cm_inst',  # Soil moisture
               'SoilTMP0_10cm_inst',  #Soil temperature
               'SoilTMP10_40cm_inst',  # Soil temperature
               'Tair_f_inst',  #Air temperature
               'Tveg_tavg',  #Transpiration
               'Wind_f_inst',  #Wind speed
               ]

def get_one_gldas(loc, outdir):

    outpath = os.path.join(outdir, loc['geohash'] + '_' + str(loc['date_start']) + '.zip')

    cell    = geohashes2cell([loc['geohash']])[0]
    mosaic  = (
        ee.ImageCollection('NASA/GLDAS/V021/NOAH/G025/T3H')
            .select(glads_bands)
            .filterDate(loc['date_start'], loc['date_end'])
    ).reduce(ee.Reducer.percentile([0, 25, 50, 75, 100]))

    url = mosaic.clip(cell).getDownloadURL(params={"name": loc['geohash'], "crs": "EPSG:4326"})
    _ = safe_urlretrieve(url, outpath)
