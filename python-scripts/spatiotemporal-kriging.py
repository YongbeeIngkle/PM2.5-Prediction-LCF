import pandas as pd
import sys
import numpy as np
from pandas import DataFrame
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats.stats import pearsonr
from math import sqrt
import statistics
from scipy.stats import *
from scipy.spatial import distance

############################################################
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import ModelCheckpoint

from matplotlib.colors import ListedColormap
from matplotlib import cm
import geopandas as gpd

import numpy.ma as ma
import math

from pykrige.ok import OrdinaryKriging

print("Repositories uploaded!!")

############################################################

tag_names = ['day', 'month', 'cmaq_x', 'cmaq_y', 'cmaq_id', 'rid', 'elev', 'forest_cover', 'pd', 'local', 'limi', 'high', 'is',
    'nldas_pevapsfc','nldas_pressfc', 'nldas_cape', 'nldas_ugrd10m', 'nldas_vgrd10m', 'nldas_tmp2m', 'nldas_rh2m', 'nldas_dlwrfsfc',
    'nldas_dswrfsfc', 'nldas_pcpsfc', 'nldas_fpcsfc', 'gc_aod', 'aod_value', 'emissi11_pm25', 'pm25_value']

monitoring_data = pd.read_csv("US_data/BigUS/us_monitoring.csv")[tag_names]
monitoring_data = monitoring_data.fillna(monitoring_data.mean())

r2_corr_list = []
r2_list = []
rmse_list = []
mae_list = []

mn_range_list = [5,10,15,20,30,40,50]

for sp in mn_range_list:
    print(sp)

    kriging_file = open(f"results/results-no-transfer/kriging-east-results", "a+")

    for splitval in range(0,10):
        print(splitval)

        day_observations = np.array([])
        day_predictions = np.array([])

        for dayval in range(1,366):
            cmaq_ids = np.load(f"/Users/shrey/Downloads/PM25/PM25-dnntransfer/US_data/split-data/single/tl-cal-{sp}/split-{splitval}/target_cmaq.npz")
            test_cmaq = cmaq_ids["test"]
            test_df = monitoring_data[monitoring_data['cmaq_id'].isin(test_cmaq)]
            test_df = test_df[test_df['day'] == dayval]
            test_df = test_df.reset_index(drop = True)
            test_krig_data = test_df[['cmaq_x', 'cmaq_y']].to_numpy()
            test_krig_labels = test_df[['pm25_value']].to_numpy()

            source_monitors = np.load("/Users/shrey/Downloads/PM25/PM25-dnntransfer/US_data/split-data/single/source_cmaq.npy").tolist()
            source_df = monitoring_data[monitoring_data['cmaq_id'].isin(source_monitors)]
            east_source_df = source_df[source_df['cmaq_x'] > 0.0]
            east_source_monitors = east_source_df['cmaq_id'].unique().tolist()

            target_monitors = cmaq_ids["train"].tolist()

            train_cmaq = east_source_monitors + target_monitors
            train_df = monitoring_data[monitoring_data['cmaq_id'].isin(train_cmaq)]
            train_df = train_df[train_df['day'] == dayval]
            train_df = train_df.reset_index(drop = True)
            train_krig_data = train_df[['cmaq_x', 'cmaq_y', 'pm25_value']]
            train_krig_data = train_krig_data.to_numpy()

            OK = OrdinaryKriging(
            train_krig_data[:, 0],
            train_krig_data[:, 1],
            train_krig_data[:, 2],
            variogram_model="linear",
            verbose=False,
            enable_plotting=False,)

            z, ss = OK.execute("points", test_krig_data[:, 0], test_krig_data[:, 1])
            pred_vals = z.data

            day_predictions = np.concatenate([day_predictions, pred_vals])
            day_observations = np.concatenate([day_observations, test_krig_labels[:,0]])

        r2_ok = pearsonr(day_observations, day_predictions)
        r2_corr_ok = (r2_ok[0])**2
        r2_corr_list.append(r2_corr_ok)

        r2_ok = r2_score(day_observations, day_predictions)
        r2_list.append(r2_ok)

        rmse_ok = sqrt(mean_squared_error(day_observations, day_predictions))
        rmse_list.append(rmse_ok)

        mae_ok = mean_absolute_error(day_observations, day_predictions)
        mae_list.append(mae_ok)

        kriging_file.write('{0}\t{1}\t{2}\t{3}\n'.format(r2_corr_ok, r2_ok, rmse_ok, mae_ok))

    kriging_file.write('{0}\t{1}\t{2}\t{3}\n'.format(statistics.mean(r2_corr_list), statistics.mean(r2_list), statistics.mean(rmse_list), statistics.mean(mae_list)))
    kriging_file.write('===================================================\n')
