#!/usr/bin/env python

"""
    date_geohash.py
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
    parser.add_argument('--dl_list_json', type=str)
    parser.add_argument('--n_jobs', type=int, default=30)
    return parser.parse_args()


def main():
    args = parse_args()

    # Get list of dates and geohashes to download
    with open(args.dl_list_json, 'r') as f:
        dates_geos = json.load(f)

    # Prepare to load data
    os.makedirs(args.outdir, exist_ok=True)

    # Run jobs in parallel
    jobs = []
    for job in dates_geos:
        job = delayed(get_one_sentinel)(job, outdir=args.outdir)
        jobs.append(job)

    random.shuffle(jobs)

    _ = Parallel(backend='multiprocessing', n_jobs=args.n_jobs, verbose=1, batch_size=4)(tqdm(jobs))

if __name__ == '__main__':
    main()