#!/usr/bin/env python

"""
    ethiopia_climate.py
    
    designed to get climate data data to pair with outputs from ethiopia_locust.py
"""

import os
from glob import glob
import random
import pandas as pd
import argparse
import geohash
from datetime import date, timedelta
from joblib import Parallel, delayed
from tqdm import tqdm

import ee
ee.Initialize()

from helpers import polygon2geohash, get_one_gldas

def get_paths(path):
    nm = os.path.basename(path).split('_')
    date_start = str(nm[1][:-4])
    date_end = str(date.fromisoformat(date_start) + timedelta(days=30))
    d = {'geohash': nm[0],
         'date_start': date_start,
         'date_end': date_end}

    return d


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str)
    parser.add_argument('--imgsdir', type=str)
    parser.add_argument('--n_jobs', type=int, default=30)
    return parser.parse_args()


def main():
    args = parse_args()

    imgs_paths = glob(os.path.join(args.imgs_dir, '*.zip'))
    locs = [get_paths(img) for img in imgs_paths]
    sorted(locs, key = lambda i: i['date_start'])

    # Prepare to load data
    os.makedirs(args.outdir, exist_ok=True)

    # Run jobs in parallel
    jobs = []
    for loc in locs:
        job = delayed(get_one_gldas)(loc, outdir=args.outdir)
        jobs.append(job)



    _ = Parallel(backend='multiprocessing', n_jobs=args.n_jobs, verbose=1, batch_size=4)(tqdm(jobs))

if __name__ == '__main__':
    main()
