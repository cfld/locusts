#!/usr/bin/env python

"""
    ethiopia_locust.py
"""

import os
import json
import random
import pandas as pd
import argparse
import geohash
from datetime import date, timedelta
from joblib import Parallel, delayed
from tqdm import tqdm

import ee
ee.Initialize()

from helpers import polygon2geohash, get_one_sentinel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str)
    '''hoppers_csv available here: https://locust-hub-hqfao.hub.arcgis.com/datasets/hoppers-1'''
    parser.add_argument('--hopper_csv', type=str)
    parser.add_argument('--country_geojson', type=str)
    parser.add_argument('--lat_min', type=int, default=34)
    parser.add_argument('--lon_max', type=int, default=15)
    parser.add_argument('--start_date', type=str, default='2016-01-01')
    parser.add_argument('--n_jobs', type=int, default=30)
    return parser.parse_args()


def main():
    args = parse_args()

    # get country geometry
    country = json.load(open(args.country_geojson))['features'][0]
    polygon = ee.Geometry.Polygon(country['geometry']['coordinates'])
    geohashes_country = polygon2geohash(polygon, precision=5, coarse_precision=5)

    # Get locations of sightings, and restrict to AOI
    df = pd.read_csv(args.hopper_csv)
    df['geohash'] = df[['Y', 'X']].apply(lambda x: geohash.encode(*x, precision=5), axis=1).values
    df = df.loc[df.STARTDATE > args.start_date].loc[df['geohash'].isin(geohashes_country)]
    df['STARTDATE'] = pd.to_datetime(df.STARTDATE)

    # Encode locations as geohashes and get surrounding geohashes
    gh = set(df['geohash'])
    for _ in range(30):
        for g in list(gh):
            gh |= set(geohash.expand(g))

    gh = list(gh.intersection(geohashes_country))

    random.shuffle(gh)
    gh = gh[:len(gh) // 3]
    gh.extend(list(df['geohash']))
    gh = list(set(gh))

    # Prepare to load data
    os.makedirs(args.outdir, exist_ok=True)

    # Get all geohashes of interest for around date where a hopper sighting occurs
    interval = 30
    delta = date.fromisoformat('2020-06-01') - date.fromisoformat(args.start_date)

    locs = []
    for i in range(int(delta.days/30)):
        start_date = date.fromisoformat(args.start_date) + timedelta(days=i*interval)
        end_date = start_date + timedelta(days=interval)
        for i in range(len(gh)):
            locs.append({'date_start': str(start_date),
                         'date_end': str(end_date),
                         'geohash': gh[i]})

    # Run jobs in parallel
    jobs = []
    for loc in locs:
        job = delayed(get_one_sentinel)(loc, outdir=args.outdir)
        jobs.append(job)

    random.shuffle(jobs)

    _ = Parallel(backend='multiprocessing', n_jobs=args.n_jobs, verbose=1, batch_size=4)(tqdm(jobs))

if __name__ == '__main__':
    main()

