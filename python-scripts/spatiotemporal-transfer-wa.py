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

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import ModelCheckpoint

from matplotlib.colors import ListedColormap
from matplotlib import cm
import geopandas as gpd

import multiprocessing

import numpy.ma as ma
import math
from keras.models import Sequential, load_model, Model
from keras.layers import Input, Dense, Activation, Conv2D, Dropout, Flatten
from keras import optimizers, utils, initializers, regularizers

from scipy.spatial import distance_matrix
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


### Tradaboost.R2 model
def run_tradaboost_model(source_x, source_y, target_x, target_y, test_x, test_y):
# def run_tradaboost_model(args_trada):
    from adapt.instance_based import TrAdaBoost, TrAdaBoostR2, TwoStageTrAdaBoostR2

    # source_x, source_y, target_x, target_y, test_x, test_y = args_trada

    trada_params = {
    "max_leaf_nodes": 4,
    "max_depth": None,
    "min_samples_split": 5
    }

    params = dict(trada_params)

    model_tradaboost = TrAdaBoostR2(DecisionTreeRegressor(**params), n_estimators = 100, Xt = target_x, yt = target_y, random_state = 2) ##DecisionTreeRegressor(max_depth = 6), n_estimators = 100, Xt = target_x, yt = target_y, random_state = 0)
    model_tradaboost.fit(source_x, source_y)

    pred_tradaboost = model_tradaboost.predict(test_x)

    r2_correlation_tradaboost = pearsonr(test_y, pred_tradaboost)
    r2_corr_tradaboost = (r2_correlation_tradaboost[0])**2

    r2_tradaboost = r2_score(test_y, pred_tradaboost)
    rmse_tradaboost = sqrt(mean_squared_error(test_y, pred_tradaboost))
    mae_tradaboost = mean_absolute_error(test_y, pred_tradaboost)

    return r2_corr_tradaboost, r2_tradaboost, rmse_tradaboost, mae_tradaboost

### Adaboost.R2 model
def run_adaboost_model(train_x, train_y, test_x, test_y):
# def run_adaboost_model(args_ada):
    from sklearn.ensemble import AdaBoostRegressor

    # train_x, train_y, test_x, test_y = args_ada

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


##### Kernel Mean Matching with DecisionTreeRegressor
def run_kmm_model(source_x, source_y, target_x, target_y, test_x, test_y):
# def run_kmm_model(args_kmm):
    from adapt.instance_based import KMM

    # source_x, source_y, target_x, target_y, test_x, test_y = args_kmm

    kmm_params = {
    "max_leaf_nodes": 4,
    "max_depth": None,
    "min_samples_split": 5
    }

    params = dict(kmm_params)

    model_kmm = KMM(DecisionTreeRegressor(**params), Xt=target_x, kernel="rbf", gamma=1., verbose=0, random_state=2)
    model_kmm.fit(source_x, source_y)

    pred_kmm = model_kmm.predict(test_x)

    r2_correlation_kmm = pearsonr(test_y, pred_kmm)
    r2_corr_kmm = (r2_correlation_kmm[0])**2

    r2_kmm = r2_score(test_y, pred_kmm)
    rmse_kmm = sqrt(mean_squared_error(test_y, pred_kmm))
    mae_kmm = mean_absolute_error(test_y, pred_kmm)

    return r2_corr_kmm, r2_kmm, rmse_kmm, mae_kmm

##### Nearest Neighbor Weighing
def run_nnw_model(source_x, source_y, target_x, target_y, test_x, test_y):
# def run_nnw_model(args_nnw):
    from adapt.instance_based import NearestNeighborsWeighting

    # source_x, source_y, target_x, target_y, test_x, test_y = args_nnw

    nnw_params = {
    "max_leaf_nodes": 4,
    "max_depth": None,
    "min_samples_split": 5
    }

    params = dict(nnw_params)

    model_nnw = NearestNeighborsWeighting(DecisionTreeRegressor(**params), n_neighbors=15, Xt = target_x, random_state=2)
    model_nnw.fit(source_x, source_y)

    pred_nnw = model_nnw.predict(test_x)

    r2_correlation_nnw = pearsonr(test_y, pred_nnw)
    r2_corr_nnw = (r2_correlation_nnw[0])**2

    r2_nnw = r2_score(test_y, pred_nnw)
    rmse_nnw = sqrt(mean_squared_error(test_y, pred_nnw))
    mae_nnw = mean_absolute_error(test_y, pred_nnw)

    return r2_corr_nnw, r2_nnw, rmse_nnw, mae_nnw

##### Unconstrained Least-Squares Importance Fitting
def run_ulsif_model(source_x, source_y, target_x, target_y, test_x, test_y):
# def run_ulsif_model(args_ulsif):
    from adapt.instance_based import ULSIF

    # source_x, source_y, target_x, target_y, test_x, test_y = args_ulsif

    ulsif_params = {
    "max_leaf_nodes": 4,
    "max_depth": None,
    "min_samples_split": 5
    }

    params = dict(ulsif_params)

    model_ulsif = ULSIF(DecisionTreeRegressor(**params), kernel="rbf",
              lambdas=[0.1, 1., 10.], gamma=[0.1, 1., 10.], Xt = target_x, random_state=2)
    model_ulsif.fit(source_x, source_y)

    pred_ulsif = model_ulsif.predict(test_x)

    r2_correlation_ulsif = pearsonr(test_y, pred_ulsif)
    r2_corr_ulsif = (r2_correlation_ulsif[0])**2

    r2_ulsif = r2_score(test_y, pred_ulsif)
    rmse_ulsif = sqrt(mean_squared_error(test_y, pred_ulsif))
    mae_ulsif = mean_absolute_error(test_y, pred_ulsif)

    return r2_corr_ulsif, r2_ulsif, rmse_ulsif, mae_ulsif

##### Relative Unconstrained Least-Squares Importance Fitting
def run_rulsif_model(source_x, source_y, target_x, target_y, test_x, test_y):
# def run_rulsif_model(args_rulsif):
    from adapt.instance_based import RULSIF

    # source_x, source_y, target_x, target_y, test_x, test_y = args_rulsif

    rulsif_params = {
    "max_leaf_nodes": 4,
    "max_depth": None,
    "min_samples_split": 5
    }

    params = dict(rulsif_params)

    model_rulsif = RULSIF(DecisionTreeRegressor(**params), kernel="rbf", alpha=0.1,
              lambdas=[0.1, 1., 10.], gamma=[0.1, 1., 10.], Xt = target_x, random_state=2) ### previously random_state = 0
    model_rulsif.fit(source_x, source_y)

    pred_rulsif = model_rulsif.predict(test_x)

    r2_correlation_rulsif = pearsonr(test_y, pred_rulsif)
    r2_corr_rulsif = (r2_correlation_rulsif[0])**2

    r2_rulsif = r2_score(test_y, pred_rulsif)
    rmse_rulsif = sqrt(mean_squared_error(test_y, pred_rulsif))
    mae_rulsif = mean_absolute_error(test_y, pred_rulsif)

    return r2_corr_rulsif, r2_rulsif, rmse_rulsif, mae_rulsif


##### Random forest regressor
def run_rf_model(train_x, train_y, test_x, test_y):
# def run_rf_model(args_rf):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV

    # train_x, train_y, test_x, test_y = args_rf

    # Create a random forest regressor with regularization
    # rf_reg = RandomForestRegressor(n_estimators=100, max_depth= None, min_samples_split= 2, min_samples_leaf= 1)

    # Define the hyperparameter space for tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 3, 5],
        'max_features': ['sqrt', 'log2']
    }

    # Create a grid search object and fit it to the data
    # model_rf = GridSearchCV(rf_reg, param_grid=param_grid, cv=5, n_jobs=-1)
    model_rf = RandomForestRegressor(n_estimators=100, max_depth= None, min_samples_split= 2, min_samples_leaf= 1)
    model_rf.fit(train_x, train_y)

    pred_rf = model_rf.predict(test_x)

    r2_correlation_rf = pearsonr(test_y, pred_rf)
    r2_corr_rf = (r2_correlation_rf[0])**2

    r2_rf = r2_score(test_y, pred_rf)
    rmse_rf = sqrt(mean_squared_error(test_y, pred_rf))
    mae_rf = mean_absolute_error(test_y, pred_rf)

    return r2_corr_rf, r2_rf, rmse_rf, mae_rf


#### Regularized gradient boosting regressor
def run_gbr_model(train_x, train_y, test_x, test_y):
# def run_gbr_model(args_gbr):
    # train_x, train_y, test_x, test_y = args_gbr

    # Define the hyperparameter space for tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 3, 5],
        'max_features': ['sqrt', 'log2'],
        'alpha': [0.01, 0.1, 1]
    }

    # Create a gradient boosting regressor with a validation split
    # gb_reg = GradientBoostingRegressor(loss='huber', validation_fraction=0.2, n_iter_no_change=5)

    # Create a grid search object and fit it to the data
    # model_gbr = GridSearchCV(gb_reg, param_grid=param_grid, cv=5, n_jobs=-1)
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



##### Domain adaptation using kernel mean matching and gradient boosting regression.
def run_kmm_gbr_model(source_x, source_y, target_x, target_y, test_x, test_y):
# def run_kmm_gbr_model(args_kgbr):
    from adapt.instance_based import KMM

    # source_x, source_y, target_x, target_y, test_x, test_y = args_kgbr

    model_kmm = KMM(DecisionTreeRegressor(max_depth = 6), Xt=target_x, kernel="rbf", gamma=1., verbose=0, random_state=0)
    weights_kmm = model_kmm.fit_weights(source_x, target_x)
    weights_kmm = np.asarray(weights_kmm)
    weights_kmm = weights_kmm.reshape((weights_kmm.size, 1))

    # Adapt the source data to the target distribution using KMM
    source_adapt_x = source_x*weights_kmm

    # Concatenate the adapted source data and target data
    X = np.concatenate((source_adapt_x, target_x))

    # Concatenate the source and target labels
    y = np.concatenate((source_y, target_y))

    gbr_params = {
    "n_estimators": 400,
    "max_leaf_nodes": 4,
    "max_depth": None,
    "random_state": 2,
    "min_samples_split": 5,
    "learning_rate": 0.1,
    "subsample": 0.5
    }

    # Initialize the Gradient Boosting Regressor object
    params = dict(gbr_params)
    model_kgbr = GradientBoostingRegressor(**params) ##learning_rate = 0.1, max_depth = 6, n_estimators = 100, subsample=0.5)

    # Fit the Gradient Boosting Regressor object on the concatenated data
    model_kgbr.fit(X, y)

    pred_kgbr = model_kgbr.predict(test_x)

    r2_correlation_kgbr = pearsonr(test_y, pred_kgbr)
    r2_corr_kgbr = (r2_correlation_kgbr[0])**2

    r2_kgbr = r2_score(test_y, pred_kgbr)
    rmse_kgbr = sqrt(mean_squared_error(test_y, pred_kgbr))
    mae_kgbr = mean_absolute_error(test_y, pred_kgbr)

    return r2_corr_kgbr, r2_kgbr, rmse_kgbr, mae_kgbr

##### Domain adaptation using Kulback Leibler Importance Estimation and gradient boosting regression.
def run_kliep_gbr_model(source_x, source_y, target_x, target_y, test_x, test_y):
    from adapt.instance_based import KLIEP

    model_kliep = KLIEP(DecisionTreeRegressor(max_depth = 6), Xt=target_x, kernel="rbf", gamma= 0.1, random_state=0)
    weights_kliep = model_kliep.fit_weights(source_x, target_x)

    weights_kliep = np.asarray(weights_kliep)
    weights_kliep = weights_kliep.reshape((weights_kliep.size, 1))

    # Adapt the source data to the target distribution using KLIEP
    source_adapt_x = source_x*weights_kliep

    # Concatenate the adapted source data and target data
    X = np.concatenate((source_adapt_x, target_x))

    # Concatenate the source and target labels
    y = np.concatenate((source_y, target_y))

    gbr_params = {
    "n_estimators": 400,
    "max_leaf_nodes": 4,
    "max_depth": None,
    "random_state": 2,
    "min_samples_split": 5,
    "learning_rate": 0.1,
    "subsample": 0.5
    }

    # Initialize the Gradient Boosting Regressor object
    params = dict(gbr_params)
    model_kliep_gbr = GradientBoostingRegressor(**params) ##learning_rate = 0.1, max_depth = 6, n_estimators = 100, subsample=0.5)

    # Fit the Gradient Boosting Regressor object on the concatenated data
    model_kliep_gbr.fit(X, y)

    pred_kliep_gbr = model_kliep_gbr.predict(test_x)

    r2_correlation_kliep_gbr = pearsonr(test_y, pred_kliep_gbr)
    r2_corr_kliep_gbr = (r2_correlation_kliep_gbr[0])**2

    r2_kliep_gbr = r2_score(test_y, pred_kliep_gbr)
    rmse_kliep_gbr = sqrt(mean_squared_error(test_y, pred_kliep_gbr))
    mae_kliep_gbr = mean_absolute_error(test_y, pred_kliep_gbr)

    return r2_corr_kliep_gbr, r2_kliep_gbr, rmse_kliep_gbr, mae_kliep_gbr


##### Domain adaptation using RULSIF and gradient boosting regression.
def run_rulsif_gbr_model(source_x, source_y, target_x, target_y, test_x, test_y):
    from adapt.instance_based import RULSIF

    model_rulsif = RULSIF(DecisionTreeRegressor(max_depth = 6), kernel="rbf", alpha=0.1,
              lambdas=[0.1, 1., 10.], gamma=[0.1, 1., 10.], Xt = target_x, random_state=2) ### previously random_state = 0
    weights_rulsif = model_rulsif.fit_weights(source_x, target_x)

    weights_rulsif = np.asarray(weights_rulsif)
    weights_rulsif = weights_rulsif.reshape((weights_rulsif.size, 1))

    # Adapt the source data to the target distribution using NNW
    source_adapt_x = source_x*weights_rulsif

    # Concatenate the adapted source data and target data
    X = np.concatenate((source_adapt_x, target_x))

    # Concatenate the source and target labels
    y = np.concatenate((source_y, target_y))

    gbr_params = {
    "n_estimators": 400,
    "max_leaf_nodes": 4,
    "max_depth": None,
    "random_state": 2,
    "min_samples_split": 5,
    "learning_rate": 0.1,
    "subsample": 0.5
    }

    # Initialize the Gradient Boosting Regressor object
    params = dict(gbr_params)
    model_rulsif_gbr = GradientBoostingRegressor(**params) ##learning_rate = 0.1, max_depth = 6, n_estimators = 100, subsample=0.5)

    # Fit the Gradient Boosting Regressor object on the concatenated data
    model_rulsif_gbr.fit(X, y)

    pred_rulsif_gbr = model_rulsif_gbr.predict(test_x)

    r2_correlation_rulsif_gbr = pearsonr(test_y, pred_rulsif_gbr)
    r2_corr_rulsif_gbr = (r2_correlation_rulsif_gbr[0])**2

    r2_rulsif_gbr = r2_score(test_y, pred_rulsif_gbr)
    rmse_rulsif_gbr = sqrt(mean_squared_error(test_y, pred_rulsif_gbr))
    mae_rulsif_gbr = mean_absolute_error(test_y, pred_rulsif_gbr)

    return r2_corr_rulsif_gbr, r2_rulsif_gbr, rmse_rulsif_gbr, mae_rulsif_gbr



##### Domain adaptation using Nearest Neighbor Weighing and gradient boosting regression.
def run_nnw_gbr_model(source_x, source_y, target_x, target_y, test_x, test_y):
    from adapt.instance_based import NearestNeighborsWeighting

    model_nnw = NearestNeighborsWeighting(DecisionTreeRegressor(max_depth = 6), Xt=target_x, n_neighbors=6, random_state=0)
    weights_nnw = model_nnw.fit_weights(source_x, target_x)

    weights_nnw = np.asarray(weights_nnw)
    weights_nnw = weights_nnw.reshape((weights_nnw.size, 1))

    # Adapt the source data to the target distribution using NNW
    source_adapt_x = source_x*weights_nnw

    # Concatenate the adapted source data and target data
    X = np.concatenate((source_adapt_x, target_x))

    # Concatenate the source and target labels
    y = np.concatenate((source_y, target_y))

    gbr_params = {
    "n_estimators": 400,
    "max_leaf_nodes": 4,
    "max_depth": None,
    "random_state": 2,
    "min_samples_split": 5,
    "learning_rate": 0.1,
    "subsample": 0.5
    }

    # Initialize the Gradient Boosting Regressor object
    params = dict(gbr_params)
    model_nnw_gbr = GradientBoostingRegressor(**params) ##learning_rate = 0.1, max_depth = 6, n_estimators = 100, subsample=0.5)

    # Fit the Gradient Boosting Regressor object on the concatenated data
    model_nnw_gbr.fit(X, y)

    pred_nnw_gbr = model_nnw_gbr.predict(test_x)

    r2_correlation_nnw_gbr = pearsonr(test_y, pred_nnw_gbr)
    r2_corr_nnw_gbr = (r2_correlation_nnw_gbr[0])**2

    r2_nnw_gbr = r2_score(test_y, pred_nnw_gbr)
    rmse_nnw_gbr = sqrt(mean_squared_error(test_y, pred_nnw_gbr))
    mae_nnw_gbr = mean_absolute_error(test_y, pred_nnw_gbr)

    return r2_corr_nnw_gbr, r2_nnw_gbr, rmse_nnw_gbr, mae_nnw_gbr


############## Kriging Regression ##############
def run_kriging_model(train_x, train_y, test_x, test_y):


    model_kriging = RegressionKriging(regression_model = RandomForestRegressor(n_estimators=100, max_depth= None,
    min_samples_split= 2, min_samples_leaf= 1), n_closest_points=10, verbose = 1)
    model_kriging.fit(train_x, train_y)

    pred_kriging = model_kriging.predict()

    r2_correlation_kriging = pearsonr(test_y, pred_kriging)
    r2_corr_riging = (r2_correlation_kriging[0])**2

    r2_kriging = r2_score(test_y, pred_kriging)
    rmse_kriging = sqrt(mean_squared_error(test_y, pred_kriging))
    mae_kriging = mean_absolute_error(test_y, pred_kriging)

    return r2_corr_riging, r2_kriging, rmse_kriging, mae_kriging


def main():
    tag_names = ['day', 'month', 'cmaq_x', 'cmaq_y', 'cmaq_id', 'rid', 'elev', 'forest_cover', 'pd', 'local', 'limi', 'high', 'is',
    'nldas_pevapsfc','nldas_pressfc', 'nldas_cape', 'nldas_ugrd10m', 'nldas_vgrd10m', 'nldas_tmp2m', 'nldas_rh2m', 'nldas_dlwrfsfc',
    'nldas_dswrfsfc', 'nldas_pcpsfc', 'nldas_fpcsfc', 'gc_aod', 'aod_value', 'emissi11_pm25', 'pm25_value']

    cols_to_scale = ['day', 'month', 'cmaq_x', 'cmaq_y', 'rid', 'elev', 'forest_cover', 'pd', 'local', 'limi', 'high', 'is',
    'nldas_pevapsfc','nldas_pressfc', 'nldas_cape', 'nldas_ugrd10m', 'nldas_vgrd10m', 'nldas_tmp2m', 'nldas_rh2m', 'nldas_dlwrfsfc',
    'nldas_dswrfsfc', 'nldas_pcpsfc', 'nldas_fpcsfc', 'gc_aod', 'aod_value', 'emissi11_pm25', 'pm25_wa']

    monitoring_df = pd.read_csv("US_data/BigUS/us_monitoring.csv")[tag_names]

    ### intialize the standard scaler
    # ss = StandardScaler()
    ss = MinMaxScaler()

    ### create the source data
    source_cmaq = np.load("US_data/split-data/single/source_cmaq.npy")  ## /Users/shrey/Downloads/PM25/PM25-dnntransfer/US_data/split-data/single/source_cmaq.npy
    source_cmaq = source_cmaq.tolist()
    source_data = monitoring_df[monitoring_df['cmaq_id'].isin(source_cmaq)]
    # source_data = source_data.drop(columns=['cmaq_id'])
    # source_data = source_data.reset_index(drop = True)
    # source_x = source_data.drop(columns=["pm25_value"]).to_numpy()
    # source_x = np.nan_to_num(source_x)
    # source_y = source_data["pm25_value"].to_numpy()
    # source_y = np.nan_to_num(source_y)

    ### the monitor range to be created
    mn_range_list = [5,10,15,20,30,40,50]

    ### parallel processing loop.
    # with multiprocessing.Pool(processes=9) as pool:

    for sp in mn_range_list:
        ### open file for each of the algorithms
        # tradaboost_file = open(f"results/results-minmax-wa/tradaboost-wa-results", "a+") ## results/
        # adaboost_file = open(f"results/results-minmax-wa/adaboost-wa-results", "a+")
        # kmm_file = open(f"results/results-minmax-wa/kmm-wa-results", "a+")
        # nnw_file = open(f"results/results-minmax-wa/nnw-wa-results", "a+")
        # ulsif_file = open(f"results/results-minmax-wa/ulsif-wa-results", "a+")
        # rulsif_file = open(f"results/results-minmax-wa/rulsif-wa-results", "a+")
        # rf_file = open(f"results/results-minmax-wa/rf-wa-results", "a+")
        # gbr_file = open(f"results/results-minmax-wa/gbr-wa-results", "a+")
        kgbr_file = open(f"results/results-minmax-wa/kgbr-results", "a+")
        kliep_gbr_file = open(f"results/results-minmax-wa/kliep-gbr-results", "a+")
        rulsif_gbr_file = open(f"results/results-minmax-wa/rulsif-gbr-results", "a+")
        nnw_gbr_file = open(f"results/results-minmax-wa/nnw-gbr-results", "a+")

        # kriging_file = open(f"results/results-minmax-wa/kriging-wa-results", "a+")

        ### define the no. of splits using the for loop range
        for idx in range(0,10):
            print(sp, idx)

            #### read the target cmaqs
            cmaq_ids = np.load(f"US_data/split-data/single/tl-cal-{sp}/split-{idx}/target_cmaq.npz")
            target_cmaq = cmaq_ids["train"]
            target_cmaq = target_cmaq.tolist()
            target_data = monitoring_df[monitoring_df['cmaq_id'].isin(target_cmaq)]
            target_data = target_data.reset_index(drop = True)

            #### create training data, train = target + source
            train_cmaq = target_cmaq + source_cmaq
            train_df = monitoring_df[monitoring_df['cmaq_id'].isin(train_cmaq)]
            train_df = train_df.reset_index(drop = True)
            train_label = train_df["pm25_value"]
            train_df = train_df.drop(columns=["pm25_value"])


            #### read the test cmaqs
            test_cmaq = cmaq_ids["test"]
            test_df = monitoring_df[monitoring_df['cmaq_id'].isin(test_cmaq)]
            test_df = test_df.reset_index(drop = True)
            test_label = test_df["pm25_value"]
            test_df = test_df.drop(columns=["pm25_value"])

            #### create the weighted average feature for the train and test data
            weight_average = WeightAverage(train_df, test_df, train_label)
            train_wa_df, test_wa_df = weight_average.weight_inputs

            #### create the individual source, target, test numpy arrays after weighted average has been obtained

            ## creating the target data with weighted average
            target_wa_df = train_wa_df[train_wa_df['cmaq_id'].isin(target_cmaq)]
            target_wa_df = target_wa_df.drop(columns=['cmaq_id'])
            target_wa_df = target_wa_df.reset_index(drop = True)
            target_wa_df[cols_to_scale] = ss.fit_transform(target_wa_df[cols_to_scale]) ### scaling target columns

            target_x = target_wa_df.to_numpy()
            target_x = np.nan_to_num(target_x)
            target_y = target_data["pm25_value"].to_numpy()
            # target_y = np.nan_to_num(target_y)

            ## creating the source data with weighted average
            source_wa_df = train_wa_df[train_wa_df['cmaq_id'].isin(source_cmaq)]
            # source_wa_df = source_wa_df.drop(columns=["level_0"])
            source_wa_df = source_wa_df.drop(columns=['cmaq_id'])
            source_wa_df = source_wa_df.reset_index(drop = True)
            source_wa_df[cols_to_scale] = ss.fit_transform(source_wa_df[cols_to_scale]) ### scaling source columns

            source_x = source_wa_df.to_numpy()
            source_x = np.nan_to_num(source_x)
            source_y = source_data["pm25_value"].to_numpy()
            # source_y = np.nan_to_num(source_y)

            ## test data is scaled and its numpy arrays are obtained
            test_wa_df = test_wa_df.drop(columns=['cmaq_id'])
            test_wa_df[cols_to_scale] = ss.fit_transform(test_wa_df[cols_to_scale]) ### scaling test columns
            test_x = test_wa_df.to_numpy()
            test_x = np.nan_to_num(test_x)
            test_y = test_label.to_numpy()
            # test_y = np.nan_to_num(test_y)

            ## scale the train data and create its numpy arrays for algorithms which need source + target
            train_wa_df = train_wa_df.drop(columns=['cmaq_id'])
            train_wa_df[cols_to_scale] = ss.fit_transform(train_wa_df[cols_to_scale]) ### scaling train columns
            train_x = train_wa_df.to_numpy()
            train_x = np.nan_to_num(train_x)
            train_y = train_label.to_numpy()
            # train_y = np.nan_to_num(train_y)


            print(train_x.shape, train_y.shape, source_x.shape, source_y.shape, target_x.shape, target_y.shape, test_x.shape, test_y.shape)

            #### function calls for executing all the algorithms
            # r2_corr_tradaboost, r2_tradaboost, rmse_tradaboost, mae_tradaboost = run_tradaboost_model(source_x, source_y, target_x, target_y, test_x, test_y)
            # r2_corr_adaboost, r2_adaboost, rmse_adaboost, mae_adaboost = run_adaboost_model(train_x, train_y, test_x, test_y)
            # r2_corr_kmm, r2_kmm, rmse_kmm, mae_kmm = run_kmm_model(source_x, source_y, target_x, target_y, test_x, test_y)
            # r2_corr_nnw, r2_nnw, rmse_nnw, mae_nnw = run_nnw_model(source_x, source_y, target_x, target_y, test_x, test_y)
            # r2_corr_ulsif, r2_ulsif, rmse_ulsif, mae_ulsif = run_ulsif_model(source_x, source_y, target_x, target_y, test_x, test_y)
            # r2_corr_rulsif, r2_rulsif, rmse_rulsif, mae_rulsif = run_rulsif_model(source_x, source_y, target_x, target_y, test_x, test_y)
            # r2_corr_rf, r2_rf, rmse_rf, mae_rf = run_rf_model(train_x, train_y, test_x, test_y)
            # r2_corr_gbr, r2_gbr, rmse_gbr, mae_gbr = run_rf_model(train_x, train_y, test_x, test_y)
            r2_corr_kgbr, r2_kgbr, rmse_kgbr, mae_kgbr = run_kmm_gbr_model(source_x, source_y, target_x, target_y, test_x, test_y)
            r2_corr_kliep_gbr, r2_kliep_gbr, rmse_kliep_gbr, mae_kliep_gbr = run_kliep_gbr_model(source_x, source_y, target_x, target_y, test_x, test_y)
            r2_corr_rulsif_gbr, r2_rulsif_gbr, rmse_rulsif_gbr, mae_rulsif_gbr = run_rulsif_gbr_model(source_x, source_y, target_x, target_y, test_x, test_y)
            r2_corr_nnw_gbr, r2_nnw_gbr, rmse_nnw_gbr, mae_nnw_gbr = run_nnw_gbr_model(source_x, source_y, target_x, target_y, test_x, test_y)
            # r2_corr_kriging, r2_kriging, rmse_kriging, mae_kriging = run_kriging_model(train_x, train_y, test_x, test_y)



            # async_result_trada = pool.map_async(run_tradaboost_model, [(source_x, source_y, target_x, target_y, test_x, test_y)])
            # async_result_ada = pool.map_async(run_adaboost_model, [(train_x, train_y, test_x, test_y)])
            # async_result_kmm = pool.map_async(run_kmm_model, [(source_x, source_y, target_x, target_y, test_x, test_y)])
            # async_result_nnw = pool.map_async(run_nnw_model, [(source_x, source_y, target_x, target_y, test_x, test_y)])
            # async_result_ulsif = pool.map_async(run_ulsif_model, [(source_x, source_y, target_x, target_y, test_x, test_y)])
            # async_result_rulsif = pool.map_async(run_rulsif_model, [(source_x, source_y, target_x, target_y, test_x, test_y)])
            # async_result_rf = pool.map_async(run_rf_model, [(train_x, train_y, test_x, test_y)])
            # async_result_gbr = pool.map_async(run_gbr_model, [(train_x, train_y, test_x, test_y)])
            # async_result_kgbr = pool.map_async(run_kmm_gbr_model, [(source_x, source_y, target_x, target_y, test_x, test_y)])
            #
            # result_trada = async_result_trada.get()
            # r2_corr_tradaboost, r2_tradaboost, rmse_tradaboost, mae_tradaboost  = result_trada[0]
            #
            # result_ada = async_result_ada.get()
            # r2_corr_adaboost, r2_adaboost, rmse_adaboost, mae_adaboost = result_ada[0]
            #
            # result_kmm = async_result_kmm.get()
            # r2_corr_kmm, r2_kmm, rmse_kmm, mae_kmm  = result_kmm[0]
            #
            # result_nnw = async_result_nnw.get()
            # r2_corr_nnw, r2_nnw, rmse_nnw, mae_nnw  = result_nnw[0]
            #
            # result_ulsif = async_result_ulsif.get()
            # r2_corr_ulsif, r2_ulsif, rmse_ulsif, mae_ulsif  = result_ulsif[0]
            #
            # result_rulsif = async_result_rulsif.get()
            # r2_corr_rulsif, r2_rulsif, rmse_rulsif, mae_rulsif   = result_rulsif[0]
            #
            # result_rf = async_result_rf.get()
            # r2_corr_rf, r2_rf, rmse_rf, mae_rf = result_rf[0]
            #
            # result_gbr = async_result_gbr.get()
            # r2_corr_gbr, r2_gbr, rmse_gbr, mae_gbr  = result_gbr[0]
            #
            # result_kgbr = async_result_kgbr.get()
            # r2_corr_kgbr, r2_kgbr, rmse_kgbr, mae_kgbr  = result_kgbr[0]


            # print(r2_tradaboost, rmse_tradaboost, mae_tradaboost)
            # print(r2_adaboost, rmse_adaboost, mae_adaboost)
            # print(r2_kliep, rmse_kliep, mae_kliep)
            # print(r2_kmm, rmse_kmm, mae_kmm)
            # print(r2_ldm, rmse_ldm, mae_ldm)
            # print(r2_nnw, rmse_nnw, mae_nnw)
            # print(r2_ulsif, rmse_ulsif, mae_ulsif)
            # print(r2_rulsif, rmse_rulsif, mae_rulsif)
            # print(r2_iwn, rmse_iwn, mae_iwn)
            # print(r2_wann, rmse_wann, mae_wann)
            # print(r2_kgbr, rmse_kgbr, mae_kgbr)
            # print(r2_corr_kriging, r2_kriging, rmse_kriging, mae_kriging)

            #### writing the results to the file
            # tradaboost_file.write('{0}\t{1}\t{2}\n'.format(r2_corr_tradaboost, r2_tradaboost, rmse_tradaboost, mae_tradaboost))
            # adaboost_file.write('{0}\t{1}\t{2}\n'.format(r2_corr_adaboost, r2_adaboost, rmse_adaboost, mae_adaboost))
            # kmm_file.write('{0}\t{1}\t{2}\n'.format(r2_corr_kmm, r2_kmm, rmse_kmm, mae_kmm))
            # nnw_file.write('{0}\t{1}\t{2}\n'.format(r2_corr_nnw, r2_nnw, rmse_nnw, mae_nnw))
            # ulsif_file.write('{0}\t{1}\t{2}\n'.format(r2_corr_ulsif, r2_ulsif, rmse_ulsif, mae_ulsif))
            # rulsif_file.write('{0}\t{1}\t{2}\n'.format(r2_corr_rulsif, r2_rulsif, rmse_rulsif, mae_rulsif))
            # rf_file.write('{0}\t{1}\t{2}\n'.format(r2_corr_rf, r2_rf, rmse_rf, mae_rf))
            # gbr_file.write('{0}\t{1}\t{2}\n'.format(r2_corr_gbr, r2_gbr, rmse_gbr, mae_gbr))
            kgbr_file.write('{0}\t{1}\t{2}\t{3}\n'.format(r2_corr_kgbr, r2_kgbr, rmse_kgbr, mae_kgbr))
            kliep_gbr_file.write('{0}\t{1}\t{2}\t{3}\n'.format(r2_corr_kliep_gbr, r2_kliep_gbr, rmse_kliep_gbr, mae_kliep_gbr))
            rulsif_gbr_file.write('{0}\t{1}\t{2}\t{3}\n'.format(r2_corr_rulsif_gbr, r2_rulsif_gbr, rmse_rulsif_gbr, mae_rulsif_gbr))
            nnw_gbr_file.write('{0}\t{1}\t{2}\t{3}\n'.format(r2_corr_nnw_gbr, r2_nnw_gbr, rmse_nnw_gbr, mae_nnw_gbr))
            # kriging_file.write('{0}\t{1}\t{2}\n'.format(r2_corr_kriging, r2_kriging, rmse_kriging, mae_kriging))

        #### adding a marker line after every monitoring station range
        # tradaboost_file.write('===================================================\n')
        # adaboost_file.write('===================================================\n')
        # kmm_file.write('===================================================\n')
        # nnw_file.write('===================================================\n')
        # ulsif_file.write('===================================================\n')
        # rulsif_file.write('===================================================\n')
        # rf_file.write('===================================================\n')
        # gbr_file.write('===================================================\n')
        kgbr_file.write('===================================================\n')
        kliep_gbr_file.write('===================================================\n')
        rulsif_gbr_file.write('===================================================\n')
        nnw_gbr_file.write('===================================================\n')
        # kriging_file.write('===================================================\n')


        # pool.close()
        # pool.join()


if __name__ == "__main__":
    main()
