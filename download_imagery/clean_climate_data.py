import os
import geohash
import argparse
import pandas as pd
import numpy as np
from tifffile import imread
from glob import glob
from zipfile import ZipFile
from tqdm import tqdm

from joblib import Parallel, delayed


from helpers import get_labels, get_locusthub_df

qts = [0, 25, 50, 75, 100]
glads_bands = ['ECanop_tavg',  # Canopy water evaporation
               'Evap_tavg',  # Evapotranspiration
               'PotEvap_tavg',  # Potential evaporation rate
               'Psurf_f_inst',  # Pressure
               'Qair_f_inst',  # Specific humidity
               'Qg_tavg',  # Heat flux
               'Rainf_f_tavg',  # Total precipitation rate
               'RootMoist_inst',  # Root zone soil moisture
               'SoilMoi0_10cm_inst',  # Soil moisture
               'SoilMoi10_40cm_inst',  # Soil moisture
               'SoilTMP0_10cm_inst',  # Soil temperature
               'SoilTMP10_40cm_inst',  # Soil temperature
               'Tair_f_inst',  # Air temperature
               'Tveg_tavg',  # Transpiration
               'Wind_f_inst',  # Wind speed
               ]

def get_paths(path):
    n = os.path.basename(path)
    d = {'path': path,
         'geohash': n[0:5],
         'date': n[6:16]}
    return d


def zip2numpy_gldas(path_dict):
    inpath = path_dict['path'].strip()
    ghash = path_dict['geohash']
    date = path_dict['date']
    with ZipFile(inpath) as handle:
        tile = np.stack([imread(handle.open(f'{ghash}.{b}_p{q}.tif')) for b in glads_bands for q in qts])
    tile = tile.astype(np.int32)
    tile = tile[:, 0, 0]
    return tile

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hoppers_path', type=str)
    parser.add_argumetn('--region_geojson_path', type=str)
    parser.add_argument('--metadata_path', type=str)
    parser.add_argument('--embs_path', type=str)
    parser.add_argument('--climate_data_path', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--n_jobs', type=int, default=30)
    return parser.parse_args()

args = parse_args()

df_hoppers = get_locusthub_df(args.hoppers_path, args.region_geojson_path)
df_meta = pd.read_csv(args.metadata_path)
Xn_orig = np.load(args.embs_path)

# Normalize
Xn_orig = Xn_orig / np.sqrt((Xn_orig ** 2).sum(axis=-1, keepdims=True))

# Remove corrupted if exist
drop_idx = np.unique(np.argwhere(np.isnan(Xn_orig))[:,0])
Xn = np.delete(Xn_orig, drop_idx, axis=0)
df_meta = df_meta.drop(drop_idx).reset_index(drop=True)

# Add missing info
df_meta['date'] = pd.to_datetime(df_meta.date)
df_meta['lat'] = df_meta.apply(lambda rec: geohash.decode(rec['geohash'])[1], axis=1)
df_meta['lon'] = df_meta.apply(lambda rec: geohash.decode(rec['geohash'])[0], axis=1)
df_meta = df_meta[['path', 'date', 'geohash', 'lat', 'lon']]

df_label = get_labels(df_meta, df_hoppers, 'hoppers', n_neighbor = 1)
y = df_label['hoppers'].values

img_paths = glob(os.path.join(args.climate_data_path, '*.zip'))
img_paths = [get_paths(img) for img in img_paths]

df_climate = pd.DataFrame(img_paths)
df_climate['date'] = pd.to_datetime(df_climate.date)
df_climate['climate_idx'] = df_climate.index
merge_idx = df_label.merge(df_climate, left_on=['geohash', 'date'], right_on=['geohash', 'date']).climate_idx


jobs = []
for img_path in img_paths:
    job = delayed(zip2numpy_gldas)(img_path)
    jobs.append(job)

climate_data = Parallel(backend='multiprocessing', n_jobs=40, verbose=0, batch_size=1)(tqdm(jobs))
climate_data = np.stack(climate_data)
climate_data = climate_data[merge_idx]

np.save(args.save_path, climate_data)
