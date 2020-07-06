import os
import json
from datetime import date, timedelta

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import bcolz

from matplotlib import pyplot as plt
from zipfile import ZipFile
from tifffile import imread

from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import geohash
from shapely import geometry
from polygon_geohasher.polygon_geohasher import polygon_to_geohashes
import geopandas as gpd


import ee
ee.Initialize()


# -
# Getting / labeling geohash timeseries data

def polygon2geohash(polygon, precision=6, coarse_precision=None, inner=True):
    polygon = geometry.shape(polygon.toGeoJSON())
    geohashes = polygon_to_geohashes(polygon, precision=coarse_precision, inner=inner)
    return sorted(list(geohashes))


def get_locusthub_df(path, country='ETH'):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df.STARTDATE)

    country = json.load(open(f'/home/ebarnett/naip_collect/geojson/countries/{country}.geo.json'))['features'][0]
    polygon = ee.Geometry.Polygon(country['geometry']['coordinates'])
    geohashes_country = polygon2geohash(polygon, precision=5, coarse_precision=5)

    df['gh'] = df[['Y', 'X']].apply(lambda x: geohash.encode(*x, precision=5), axis=1).values
    df = df.loc[df.STARTDATE > '2016-01-01'].loc[df['gh'].isin(geohashes_country)]

    df = df[['gh', 'Y', 'X', 'date']]
    return df


def get_labels(df, df_label, label_name='hoppers', n_neighbor=0):
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


# -
# Models

def random_guess(X_valid):
    '''randomly select points'''
    pred_valid = np.random.permutation(np.arange(X_valid.shape[0]))
    return pred_valid

def persistence(idx_valid, df_test, hashes):
    '''if a point has had a positive example, predict it again'''

    df_test = df_test.iloc[idx_valid].reset_index(drop=True)
    pos_idx = df_test.loc[df_test['geohash'].isin(hashes)].index
    pred_valid = np.zeros((len(idx_valid),))
    pred_valid[pos_idx] = 1

    return pred_valid

def train_val_SVM(X_train, X_valid, y_train, y_valid):
    model = LinearSVC(C=20, max_iter=5000).fit(X_train, y_train)

    pred_valid = model.decision_function(X_valid)
    pred_train = model.decision_function(X_train)
    thresh = np.median(pred_train[y_train])
    metrics_dict = get_metrics(pred_valid, y_valid, thresh)
    return metrics_dict, pred_valid


# -
# Metrics

def compute_topN(pred, true, N):
    idx = pred.argsort()[-N:][::-1]
    acc = np.sum(true[idx]) / N
    return acc

def get_conf(y_hat, y_actual):
    tp, fp, tn, fn = np.zeros(len(y_hat), ), np.zeros(len(y_hat), ), np.zeros(len(y_hat), ), np.zeros(len(y_hat), )
    for i in range(len(y_hat)):
        if y_hat[i] == 1:
            if y_actual[i] == 1:
                tp[i] = 1
            else:
                fp[i] = 1
        else:
            if y_actual[i] == 0:
                tn[i] = 1
            else:
                fn[i] = 1
    return tp, fp, tn, fn

def get_confusion_metrics(y_hat, y_actual):
    tp, fp, tn, fn = get_conf(y_hat, y_actual)
    tp = np.sum(tp)
    fp = np.sum(fp)
    tn = np.sum(tn)
    fn = np.sum(fn)
    prec = tp/(tp+fp)
    rec = tp/(tp+fn)
    return tp, fp, tn, fn, prec, rec



def get_metrics(pred, y, thresh):
    tp, fp, tn, fn, prec, rec = get_confusion_metrics(pred, y)
    metrics_dict = {'top5': compute_topN(pred, y, 5),
                    'top10': compute_topN(pred, y, 10),
                    'top50': compute_topN(pred, y, 50),
                    'top100': compute_topN(pred, y, 100),
                    'top500': compute_topN(pred, y, 500),
                    'top1000': compute_topN(pred, y, 1000),
                    'valid_auc': metrics.roc_auc_score(y, pred > thresh),
                    'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
                    'prec': prec, 'rec': rec}
    return metrics_dict


# -
# Image Plotting Tools

def zip2numpy_sentinel(inpath):
    inpath = inpath.strip()
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
            imread(handle.open(f'{ghash}.B11.tif')),
            imread(handle.open(f'{ghash}.B12.tif')),
            imread(handle.open(f'{ghash}.QA60.tif'))
        ])
        tile = tile.astype(np.int32)

    return tile

def prep_rgb(x):
    x = x[1:4].transpose(1, 2, 0).astype('float32')
    x /= x.max(axis=(0, 1), keepdims=True) + 1e-10
    return x


def plot_(img_paths_pred, title):
    fig, axes = plt.subplots(1, len(img_paths_pred), figsize=(10 * len(img_paths_pred), 10))
    for i in range(len(img_paths_pred)):
        img1 = prep_rgb(zip2numpy_sentinel(img_paths_pred[i]['path']))
        _ = axes[i].imshow(img1, aspect='auto')

        pred, truth, in_train, hp_in_train = '', '', '', ''
        if img_paths_pred[i]['pred'] == True:
            pred = 'Predicted Hopper'
        if img_paths_pred[i]['event'] >= 1:
            truth = 'Hopper GroundTruth'
        if img_paths_pred[i]['in_train'] >= 1:
            in_train = 'In train'
        else:
            in_train = 'In val'

        axes[i].title.set_text(f'({img_paths_pred[i]["date"]}) ({in_train}) ({truth}) ({pred})')
    fig.suptitle(title)

# -
# Map Plotting tools


def plot_map(df_all, title, pause=False):
    df_all['lat'] = df_all.apply(lambda rec: geohash.decode(rec['geohash'])[1], axis=1)
    df_all['lon'] = df_all.apply(lambda rec: geohash.decode(rec['geohash'])[0], axis=1)

    gdf = gpd.GeoDataFrame(df_all, geometry=gpd.points_from_xy(df_all.lat, df_all.lon))
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    fig.suptitle(f'East African Hopper Locust detection via Sentenel 2 Imagery', fontsize=20)
    ax.set_title(title)

    world[world.name.isin(['Ethiopia', 'Kenya', 'Somalia', 'Eritrea', 'Yemen', 'Djibouti', 'Somaliland'])].plot(
        color='white', edgecolor='Black', ax=ax)
    gdf.loc[gdf['tn'] == 1].plot(ax=ax, color='blue')
    gdf.loc[gdf['fp'] == 1].plot(ax=ax, color='red')
    gdf.loc[gdf['fn'] == 1].plot(ax=ax, color='yellow')
    gdf.loc[gdf['tp'] == 1].plot(ax=ax, color='green')

    ax.legend(['tn', 'fp', 'fn', 'tp'], loc="upper right", title="legend")
    if pause:
        plt.show(block=False)
        plt.pause(3)
        plt.close()


