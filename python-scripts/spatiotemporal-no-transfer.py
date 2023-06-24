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
# from sklearn.kernel_mean import KernelMeanMatching

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats.stats import pearsonr
from math import sqrt
import statistics
from scipy.stats import *
from scipy.spatial import distance

from sklearn.preprocessing import StandardScaler, MinMaxScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import ModelCheckpoint

from matplotlib.colors import ListedColormap
from matplotlib import cm
import geopandas as gpd

import numpy.ma as ma
import math
from keras.models import Sequential, load_model, Model
from keras.layers import Input, Dense, Activation, Conv2D, Dropout, Flatten
from keras import optimizers, utils, initializers, regularizers

from scipy.spatial import distance_matrix
from sklearn.ensemble import RandomForestRegressor
from pykrige.rk import RegressionKriging

# from torch.utils.data import Dataset, DataLoader

print("Repositories uploaded!!")
####################################################################################

def create_distance_matrix(dt1: pd.DataFrame, dt2: pd.DataFrame):
    all_distance = distance_matrix(dt1, dt2)
    all_distance[all_distance==0] = np.inf
    all_distance = pd.DataFrame(all_distance, index=dt1.index, columns=dt2.index)
    return all_distance

class WeightAverage:
    def __init__(self, train_data, valid_data, train_label):
        self.train_data = train_data
        self.valid_data = valid_data
        self.train_label = train_label
        self.cmaq_cols = ["cmaq_x", "cmaq_y", "cmaq_id"]
        self._allocate_weight()
        self._compute_all_wa()

    def _allocate_weight(self):
        train_cmaq = pd.DataFrame(np.unique(self.train_data[self.cmaq_cols], axis=0), columns=self.cmaq_cols).set_index("cmaq_id")
        valid_cmaq = pd.DataFrame(np.unique(self.valid_data[self.cmaq_cols], axis=0), columns=self.cmaq_cols).set_index("cmaq_id")
        self.train_weight = 1 / create_distance_matrix(train_cmaq, train_cmaq)
        self.valid_weight = 1 / create_distance_matrix(valid_cmaq, train_cmaq)

    def _date_weight_average(self, data: pd.DataFrame, weight, train_label, train_cmaq):
        exist_weight = weight.loc[data["cmaq_id"], np.isin(weight.columns, train_cmaq)]
        weight_label = train_label[exist_weight.columns]
        weight_sum = np.sum(exist_weight, axis=1)
        cmaq_wa = np.sum(exist_weight*weight_label, axis=1)/weight_sum
        cmaq_wa.index = data.index
        return cmaq_wa

    def _compute_date_wa(self, date):
        date_train_data = self.train_data.loc[self.train_data["day"]==date].copy()
        date_valid_data = self.valid_data.loc[self.valid_data["day"]==date].copy()
        date_train_label = self.train_label.loc[self.train_data["day"]==date]
        date_train_label.index = date_train_data["cmaq_id"]
        train_wa = self._date_weight_average(date_train_data, self.train_weight, date_train_label, date_train_data["cmaq_id"])
        valid_wa = self._date_weight_average(date_valid_data, self.valid_weight, date_train_label, date_train_data["cmaq_id"])
        date_train_data["pm25_wa"] = train_wa
        date_valid_data["pm25_wa"] = valid_wa
        return date_train_data, date_valid_data

    def _compute_all_wa(self):
        all_dates = np.unique(self.train_data["day"])
        all_train_data, all_valid_data = [], []
        for date in all_dates:
            date_train, date_valid = self._compute_date_wa(date)
            all_train_data.append(date_train)
            all_valid_data.append(date_valid)
        all_train_data = pd.concat(all_train_data)
        all_valid_data = pd.concat(all_valid_data)
        self.weight_inputs = (all_train_data, all_valid_data)


################## Adaboost.R2 model ##################
def run_adaboost_model(train_x, train_y, test_x, test_y):
    from sklearn.ensemble import AdaBoostRegressor

    ada_params = {
    "max_leaf_nodes": 4,
    "max_depth": None,
    "min_samples_split": 5
    }

    params = dict(ada_params)

    model_adaboost = AdaBoostRegressor(DecisionTreeRegressor(**params), learning_rate = 0.1, n_estimators = 100)
    model_adaboost.fit(train_x, train_y)

    pred_adaboost = model_adaboost.predict(test_x)

    r2_correlation_adaboost = pearsonr(test_y, pred_adaboost)
    r2_corr_adaboost = (r2_correlation_adaboost[0])**2

    r2_adaboost = r2_score(test_y, pred_adaboost)
    rmse_adaboost = sqrt(mean_squared_error(test_y, pred_adaboost))
    mae_adaboost = mean_absolute_error(test_y, pred_adaboost)

    return r2_corr_adaboost, r2_adaboost, rmse_adaboost, mae_adaboost


################## Random forest regressor ##################
def run_rf_model(train_x, train_y, test_x, test_y):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV

    # Define the hyperparameter space for tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 3, 5],
        'max_features': ['sqrt', 'log2']
    }

    model_rf = RandomForestRegressor(n_estimators=100, max_depth= None, min_samples_split= 2, min_samples_leaf= 1)
    model_rf.fit(train_x, train_y)

    pred_rf = model_rf.predict(test_x)

    r2_correlation_rf = pearsonr(test_y, pred_rf)
    r2_corr_rf = (r2_correlation_rf[0])**2

    r2_rf = r2_score(test_y, pred_rf)
    rmse_rf = sqrt(mean_squared_error(test_y, pred_rf))
    mae_rf = mean_absolute_error(test_y, pred_rf)

    return r2_corr_rf, r2_rf, rmse_rf, mae_rf


################## Regularized gradient boosting regressor ##################
def run_gbr_model(train_x, train_y, test_x, test_y):

    # Define the hyperparameter space for tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 3, 5],
        'max_features': ['sqrt', 'log2'],
        'alpha': [0.01, 0.1, 1]
    }

    model_gbr = GradientBoostingRegressor(loss='huber', n_estimators=100, max_depth= None,
    min_samples_split= 2, min_samples_leaf= 1, validation_fraction=0.2, n_iter_no_change=5)
    model_gbr.fit(train_x, train_y)

    pred_gbr = model_gbr.predict(test_x)

    r2_correlation_gbr = pearsonr(test_y, pred_gbr)
    r2_corr_gbr = (r2_correlation_gbr[0])**2

    r2_gbr = r2_score(test_y, pred_gbr)
    rmse_gbr = sqrt(mean_squared_error(test_y, pred_gbr))
    mae_gbr = mean_absolute_error(test_y, pred_gbr)

    return r2_corr_gbr, r2_gbr, rmse_gbr, mae_gbr


################## Linear Support Vector Regression ##################
def run_svr_model(target_x, target_y, test_x, test_y):
    from sklearn.svm import LinearSVR

    # Hyperparameter Tuning
    param_grid = {'C': [0.1, 1, 10],
                  'epsilon': [0.1, 0.01, 0.001]}
    grid_search = GridSearchCV(LinearSVR(random_state=0, tol=1e-3), param_grid)
    grid_search.fit(target_x, target_y)
    best_model_svr = grid_search.best_estimator_

    # Predictions
    pred_svr = best_model_svr.predict(test_x)

    # model_svr = LinearSVR(random_state=0, tol=1e-5)
    # model_svr.fit(target_x, target_y)
    #
    # pred_svr = model_svr.predict(test_x)

    r2_correlation_svr = pearsonr(test_y, pred_svr)
    r2_corr_svr = (r2_correlation_svr[0])**2

    r2_svr = r2_score(test_y, pred_svr)
    rmse_svr = sqrt(mean_squared_error(test_y, pred_svr))
    mae_svr = mean_absolute_error(test_y, pred_svr)

    return r2_corr_svr, r2_svr, rmse_svr, mae_svr


################## Stochastic Gradient Descent Regression ##################
def run_sgd_model(target_x, target_y, test_x, test_y):
    from sklearn.linear_model import SGDRegressor

    # Hyperparameter Tuning
    param_grid = {'alpha': [0.1, 1, 10],
                  'learning_rate': ['constant', 'optimal', 'adaptive'],
                  'loss': ['huber', 'epsilon_insensitive', 'squared_loss'],
                  'epsilon': [0.1, 0.01, 0.001]}
    grid_search = GridSearchCV(SGDRegressor(max_iter=1000, tol=1e-3), param_grid)
    grid_search.fit(target_x, target_y)
    best_model_sgd = grid_search.best_estimator_

    # Predictions
    pred_sgd = best_model_sgd.predict(test_x)


    # model_sgd = SGDRegressor(loss= 'huber', max_iter = 1000, tol = 1e-3)
    # model_sgd.fit(target_x, target_y)
    #
    # pred_sgd = model_sgd.predict(test_x)

    r2_correlation_sgd = pearsonr(test_y, pred_sgd)
    r2_corr_sgd = (r2_correlation_sgd[0])**2

    r2_sgd = r2_score(test_y, pred_sgd)
    rmse_sgd = sqrt(mean_squared_error(test_y, pred_sgd))
    mae_sgd = mean_absolute_error(test_y, pred_sgd)

    return r2_corr_sgd, r2_sgd, rmse_sgd, mae_sgd


def _statistic_compute(train_input, valid_input):
    """
    Normalize the train (source and train target) and validation data.
    If standard deviation is 0, substitute to 1 to make the values 0 with just subtracting by mean.
    """
    mean, std = train_input.mean(axis=0), train_input.std(axis=0)
    std[std==0] = 1
    normalized_train = (train_input - mean) / std
    normalized_valid = (valid_input - mean) / std
    return normalized_train, normalized_valid


def main():
    tag_names = ['day', 'month', 'cmaq_x', 'cmaq_y', 'cmaq_id', 'rid', 'elev', 'forest_cover', 'pd', 'local', 'limi', 'high', 'is',
    'nldas_pevapsfc','nldas_pressfc', 'nldas_cape', 'nldas_ugrd10m', 'nldas_vgrd10m', 'nldas_tmp2m', 'nldas_rh2m', 'nldas_dlwrfsfc',
    'nldas_dswrfsfc', 'nldas_pcpsfc', 'nldas_fpcsfc', 'gc_aod', 'aod_value', 'emissi11_pm25', 'pm25_value']

    geo_tag_names = ['day', 'month', 'cmaq_x', 'cmaq_y', 'cmaq_id', 'lat', 'lon', 'rid', 'elev', 'forest_cover', 'pd', 'local', 'limi', 'high', 'is',
    'nldas_pevapsfc','nldas_pressfc', 'nldas_cape', 'nldas_ugrd10m', 'nldas_vgrd10m', 'nldas_tmp2m', 'nldas_rh2m', 'nldas_dlwrfsfc',
    'nldas_dswrfsfc', 'nldas_pcpsfc', 'nldas_fpcsfc', 'gc_aod', 'aod_value', 'emissi11_pm25', 'pm25_value']

    cols_to_scale = ['day', 'month', 'cmaq_x', 'cmaq_y', 'rid', 'elev', 'forest_cover', 'pd', 'local', 'limi', 'high', 'is',
    'nldas_pevapsfc','nldas_pressfc', 'nldas_cape', 'nldas_ugrd10m', 'nldas_vgrd10m', 'nldas_tmp2m', 'nldas_rh2m', 'nldas_dlwrfsfc',
    'nldas_dswrfsfc', 'nldas_pcpsfc', 'nldas_fpcsfc', 'gc_aod', 'emissi11_pm25'] ##'aod_value',

    monitoring_df = pd.read_csv("US_data/BigUS/us_monitoring.csv")[tag_names]
    geo_monitoring_df = pd.read_csv("US_data/BigUS/us_monitoring.csv")[geo_tag_names]

    ### intialize the standard scaler
    ss = MinMaxScaler()

    ### the monitor range to be created
    mn_range_list = [5,10,15,20,30,40,50]

    for sp in mn_range_list:
        ### open file for each of the algorithms
        # adaboost_file = open(f"results/results-no-transfer/adaboost-results", "a+")
        rf_file = open(f"results/results-no-transfer/rf-results-v2", "a+")
        gbr_file = open(f"results/results-no-transfer/gbr-results-v2", "a+")
        # svr_file = open(f"results/results-no-transfer/svr-results-v3", "a+")
        # sgd_file = open(f"results/results-no-transfer/sgd-results-v3", "a+")

        ### define the no. of splits using the for loop range
        for idx in range(0,10):
            print(sp, idx)

            #### read the target cmaqs
            cmaq_ids = np.load(f"US_data/split-data/single/tl-cal-{sp}/split-{idx}/target_cmaq.npz")
            target_cmaq = cmaq_ids["train"]
            target_df = monitoring_df[monitoring_df['cmaq_id'].isin(target_cmaq)]
            target_df = target_df.drop(columns=['cmaq_id', 'aod_value'])
            target_df = target_df.reset_index(drop = True)

            #### When using minmax scaler
            # target_df[cols_to_scale] = ss.fit_transform(target_df[cols_to_scale])
            # target_x = target_df.drop(columns=["pm25_value"]).to_numpy()
            # target_x = np.nan_to_num(target_x)
            # target_y = target_df["pm25_value"].to_numpy()


            #### read the test cmaqs
            test_cmaq = cmaq_ids["test"]
            test_df = monitoring_df[monitoring_df['cmaq_id'].isin(test_cmaq)]
            test_df = test_df.drop(columns=['cmaq_id', 'aod_value'])
            test_df = test_df.reset_index(drop = True)
            test_label = test_df["pm25_value"]
            test_df = test_df.drop(columns=["pm25_value"])

            #### When using minmax scaler
            # test_df[cols_to_scale] = ss.fit_transform(test_df[cols_to_scale])
            # test_x = test_df.to_numpy()
            # test_x = np.nan_to_num(test_x)
            # test_y = test_label.to_numpy()

            #### When using std, mean scaling
            target_df[cols_to_scale], test_df[cols_to_scale] = _statistic_compute(target_df[cols_to_scale], test_df[cols_to_scale])

            #### Update the target and test data by splitting them into features and y.
            target_x = target_df.drop(columns=["pm25_value"]).to_numpy()
            # target_x = np.nan_to_num(target_x)
            target_y = target_df["pm25_value"].to_numpy()

            test_x = test_df.to_numpy()
            # test_x = np.nan_to_num(test_x)
            test_y = test_label.to_numpy()


            print(target_x.shape, target_y.shape, test_x.shape, test_y.shape)

            #### function calls for executing all the algorithms
            # r2_corr_adaboost, r2_adaboost, rmse_adaboost, mae_adaboost = run_adaboost_model(target_x, target_y, test_x, test_y)
            r2_corr_rf, r2_rf, rmse_rf, mae_rf = run_rf_model(target_x, target_y, test_x, test_y)
            r2_corr_gbr, r2_gbr, rmse_gbr, mae_gbr = run_rf_model(target_x, target_y, test_x, test_y)
            # r2_corr_svr, r2_svr, rmse_svr, mae_svr = run_svr_model(target_x, target_y, test_x, test_y)
            # r2_corr_sgd, r2_sgd, rmse_sgd, mae_sgd = run_sgd_model(target_x, target_y, test_x, test_y)

            # print(r2_corr_adaboost, r2_adaboost, rmse_adaboost, mae_adaboost)
            # print(r2_corr_rf, r2_rf, rmse_rf, mae_rf)
            # print(r2_corr_gbr, r2_gbr, rmse_gbr, mae_gbr)
            # print(r2_corr_svr, r2_svr, rmse_svr, mae_svr)
            # print(r2_corr_sgd, r2_sgd, rmse_sgd, mae_sgd)


            #### writing the results to the file
            # adaboost_file.write('{0}\t{1}\t{2}\t{3}\n'.format(r2_corr_adaboost, r2_adaboost, rmse_adaboost, mae_adaboost))
            rf_file.write('{0}\t{1}\t{2}\t{3}\n'.format(r2_corr_rf, r2_rf, rmse_rf, mae_rf))
            gbr_file.write('{0}\t{1}\t{2}\t{3}\n'.format(r2_corr_gbr, r2_gbr, rmse_gbr, mae_gbr))
            # svr_file.write('{0}\t{1}\t{2}\t{3}\n'.format(r2_corr_svr, r2_svr, rmse_svr, mae_svr))
            # sgd_file.write('{0}\t{1}\t{2}\t{3}\n'.format(r2_corr_sgd, r2_sgd, rmse_sgd, mae_sgd))

        #### adding a marker line after every monitoring station range
        # adaboost_file.write('===================================================\n')
        rf_file.write('===================================================\n')
        gbr_file.write('===================================================\n')
        # svr_file.write('===================================================\n')
        # sgd_file.write('===================================================\n')

if __name__ == "__main__":
    main()
