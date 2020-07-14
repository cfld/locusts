import os
import json
import warnings
from datetime import timedelta
import pandas as pd
import numpy as np
import argparse

warnings.filterwarnings('ignore')

import ee
ee.Initialize()

import geopandas as gpd
import geohash
from shapely import geometry
from polygon_geohasher.polygon_geohasher import polygon_to_geohashes

from sklearn.svm import LinearSVC
from sklearn import metrics
from scipy.stats import rankdata

from matplotlib import pyplot as plt

def get_locusthub_df(csv_path, geojson_path):
    '''
    clean up raw csv data from locusthub data and subset to only geojson region
    '''
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df.STARTDATE)

    country = json.load(open(geojson_path))['features'][0]
    polygon = ee.Geometry.Polygon(country['geometry']['coordinates'])
    polygon = geometry.shape(polygon.toGeoJSON())
    geohashes = polygon_to_geohashes(polygon, precision=5, inner=5)
    geohashes_country = sorted(list(geohashes))

    df['gh'] = df[['Y', 'X']].apply(lambda x: geohash.encode(*x, precision=5), axis=1).values
    df = df.loc[df.STARTDATE > '2016-01-01'].loc[df['gh'].isin(geohashes_country)]
    df = df[['gh', 'Y', 'X', 'date']]

    return df

def get_labels(df, df_label, label_name='hoppers', n_neighbor=0):
    '''
        add locust information to the features metadata
    '''
    df[label_name] = 0
    for row in df_label.iterrows():
        start_day = row[1].date
        end_day = start_day + timedelta(days=30)
        gh = set([row[1].gh])
        if n_neighbor > 0:
            for _ in range(n_neighbor):
                for g in list(gh):
                    gh |= set(geohash.expand(g))
        gh = list(gh)
        idx = df[label_name].loc[df['geohash'].isin(gh)].loc[df['date'] >= start_day].loc[
            df['date'] < end_day].index.values
        df[label_name].iloc[idx] = 1
    return df

def plot_heatmap(df_train, df_valid, heatmap_val, titles, outpath, dot_size = 10,):
    '''
        plot heatmap of predictions for positive class in geography
    '''
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    ax.set_title(titles)
    gdf_train = gpd.GeoDataFrame(df_train,
                                 geometry=gpd.points_from_xy(df_train.lat, df_train.lon))
    gdf_valid = gpd.GeoDataFrame(df_valid,
                                 geometry=gpd.points_from_xy(df_valid.lat, df_valid.lon))

    gdf_valid.plot(ax           = ax,
                   column       = heatmap_val,
                   cmap         = 'viridis',
                   vmin         = gdf_valid[heatmap_val].min(),
                   vmax         = gdf_valid[heatmap_val].max(),
                   markersize   = dot_size)

    gdf_train.loc[df_train['hoppers'] == 1].plot(ax = ax, color = 'blue', markersize = int(dot_size))
    gdf_valid.loc[df_valid['hoppers'] == 1].plot(ax = ax, color = 'red', markersize = int(dot_size))

    fig.savefig(outpath)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hoppers_csv_path', type=str, help='get from https://locust-hub-hqfao.hub.arcgis.com/datasets/hoppers-1')
    parser.add_argument('--geojson_path', type=str)
    parser.add_argument('--metadata_path', type=str, help='Features metadata are dates and geohashes for each row of features.npy')
    parser.add_argument('--features_path', type=str)
    parser.add_argument('--save_path', type=str)
    
    return parser.parse_args()

args = parse_args()

print('getting features / labels / metadata')
# Read in hopper label data, feature metadata, and features
df_hoppers = get_locusthub_df(args.hoppers_csv_path, args.geojson_path)
df_meta = pd.read_csv(args.metadata_path)
Xn_orig = np.load(args.features_path)

# Normalize Features
Xn = Xn_orig / np.sqrt((Xn_orig ** 2).sum(axis=-1, keepdims=True))

# Remove corrupted if exists
drop_idx = np.unique(np.argwhere(np.isnan(Xn))[:,0])
Xn = np.delete(Xn, drop_idx, axis=0)
Xn_orig = np.delete(Xn_orig, drop_idx, axis=0)
df_meta = df_meta.drop(drop_idx).reset_index(drop=True)

# Add missing info
df_meta['date'] = pd.to_datetime(df_meta.date)
df_meta['lat'] = df_meta.apply(lambda rec: geohash.decode(rec['geohash'])[1], axis=1)
df_meta['lon'] = df_meta.apply(lambda rec: geohash.decode(rec['geohash'])[0], axis=1)
df_meta = df_meta[['path', 'date', 'geohash', 'lat', 'lon']]

# Create labels
df_label = get_labels(df_meta, df_hoppers, 'hoppers', n_neighbor = 0)
y = df_label['hoppers'].values

# Run experiment using past to predict current time step (only for dates with locusts)
print('running experiment')

# Train/val split is defined by time (past to predict present)
idx_train = df_label.loc[(df_label.date < '2019-11-11')].index
idx_valid = df_label.loc[df_label.date == '2019-11-11'].index

X_train, X_valid, y_train, y_valid = Xn[idx_train], Xn[idx_valid], y[idx_train], y[idx_valid]
df_train = df_label.iloc[idx_train]
df_valid = df_label.iloc[idx_valid]

# Train model
model = LinearSVC().fit(X_train, y_train)

# Predict on validation set
pred_valid = model.decision_function(X_valid)

# Remove predictions which
uq, cts = np.unique(pred_valid, return_counts=True)
blank_locs = np.where(np.isin(pred_valid, uq[cts > 20]))

# Compute ROC_AUC with only predictions not containing blanks
roc_auc = metrics.roc_auc_score(np.delete(y_valid, blank_locs), np.delete(pred_valid, blank_locs))

# Rank data
df_valid['rank'] = rankdata(pred_valid)
df_valid['rank'].iloc[blank_locs] = np.nan

print('plotting')
plot_heatmap(df_train, df_valid,
             heatmap_val = 'rank',
             dot_size    = 10,
             titles      = f'predictions for 2019-11-11, ROC AUC = {roc_auc})',
             outpath     = os.path.join(args.save_path, 'plot.csv'))

