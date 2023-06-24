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

from sklearn.preprocessing import StandardScaler

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

import libpysal as ps
from mgwr.gwr import GWR, MGWR
from mgwr.sel_bw import Sel_BW
# import geopandas as gp

print("Repositories uploaded!!")
############################################################

fold = 5
tr_batch_size = 100
ev_batch_size = 100
noise_std = 0.1
n_conv = 64
n_hidden = 64
SEED_NUM = 10000
Input_width = 5
Input_height = 5
num_channels = 28

################################################# Model Transfer Learning Models #################################################

def get_model_pm25():
    model = Sequential([
    Conv2D(n_conv, (3,3), kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=SEED_NUM), bias_initializer=initializers.Constant(0.1), input_shape=(Input_width, Input_height, num_channels), padding='same', kernel_regularizer=regularizers.l1(0.005)),
    Activation('relu'),
    Flatten(),
    Dropout(0.2, seed=SEED_NUM), ### originally 0.5 was used for dropout.
    Dense(n_hidden, activation='elu', kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=SEED_NUM), bias_initializer=initializers.Constant(0.1), kernel_regularizer=regularizers.l1(0.005)),
    Dense(1, activation='linear')
    ])
    nadam = optimizers.Nadam(learning_rate=0.0002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    model.compile(optimizer=nadam, loss='mse', metrics=['mae'])
    return model

def model_target_lessdense():
    modelSrc = load_model("US_data/us-2011/target_california_grid_split/model_weights_cal/grid_CNN.weights.best.hdf5")

    for i in range(3):
        modelSrc.layers[i].trainable = False

    ll = modelSrc.layers[4].output
    ll = Dropout(0.2, seed=SEED_NUM)(ll)

    ll = Dense(64, activation='elu',
              kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=SEED_NUM),
              bias_initializer=initializers.Constant(0.1), kernel_regularizer=regularizers.l1(0.005))(ll)
    ll = Dropout(0.2, seed=SEED_NUM)(ll)

    ll = Dense(32, activation='elu',
              kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=SEED_NUM),
              bias_initializer=initializers.Constant(0.1), kernel_regularizer=regularizers.l1(0.005))(ll)
    ll = Dropout(0.2, seed=SEED_NUM)(ll)

    ll = Dense(16, activation='elu',
              kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=SEED_NUM),
              bias_initializer=initializers.Constant(0.1), kernel_regularizer=regularizers.l1(0.005))(ll)
    ll = Dropout(0.2, seed=SEED_NUM)(ll)

    ll = Dense(8, activation='elu',
              kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=SEED_NUM),
              bias_initializer=initializers.Constant(0.1), kernel_regularizer=regularizers.l1(0.005))(ll)
    ll = Dropout(0.2, seed=SEED_NUM)(ll)

    ll = Dense(1, activation='linear')(ll)

    model = Model(inputs = modelSrc.input, outputs=ll)
    nadam = optimizers.Nadam(learning_rate=0.0002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    model.compile(optimizer=nadam, loss='mse', metrics=['mae'])
    return model

def get_model():
    model = Sequential([Conv2D(n_conv, (3,3), kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=SEED_NUM),
                               bias_initializer=initializers.Constant(0.1), input_shape=(Input_width, Input_height, num_channels),
                               padding='same', kernel_regularizer=regularizers.l1(0.005)),
                        Activation('relu'),
                        Flatten(),
                        Dropout(0.5, seed=SEED_NUM),
                        Dense(n_hidden, activation='elu',
                              kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=SEED_NUM),
                              bias_initializer=initializers.Constant(0.1), kernel_regularizer=regularizers.l1(0.005)),
                        Dense(1, activation='linear')])
    nadam = optimizers.Nadam(learning_rate=0.0002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    model.compile(optimizer=nadam, loss='mse', metrics=['mae'])
    return model

def run_source_model(x_source, y_source):
    epochs_ = 30

    model_pm25 = get_model_pm25()

    print(x_source.shape, y_source.shape)
    x_train, x_test, y_train, y_test = train_test_split(x_source, y_source, test_size=0.3, random_state=42)
    x_train, x_test = norm_by_std_nan(x_train, x_test)
    x_train = x_train.reshape(len(x_train), Input_width, Input_height, num_channels)
    x_test   = x_test.reshape(len(x_test), Input_width, Input_height, num_channels)
    print(x_train.shape, x_test.shape)

    checkpointer = ModelCheckpoint(filepath='US_data/us-2011/target_california_grid_split/model_weights_cal/grid_CNN.weights.best.hdf5', save_best_only=True) ##verbose=1,
    history = model_pm25.fit(x_train, y_train, epochs=epochs_, batch_size=tr_batch_size, validation_split=0.1, callbacks=[checkpointer]) ##verbose=1,

    pred = model_pm25.predict(x_test, batch_size=ev_batch_size).reshape(len(x_test),)

    val_r2 = 1 - (np.sum(np.square(y_test - pred)) / np.sum(np.square(y_test - np.mean(y_test))))
    print("epoch:{}, validation set r-squared:{}".format(epochs_, val_r2))

    val_r2_v2 = pearsonr(y_test, pred)
    val_r2_v2 = (val_r2_v2[0])**2
    print("epoch:{}, validation set r-squared-version2:{}".format(epochs_, val_r2_v2))

    val_mse = sqrt(mean_squared_error(y_test, pred))
    print("RMSE", val_mse)

    val_mae = mean_absolute_error(y_test, pred)
    print("MAE", val_mae)

    return val_r2, val_r2_v2, val_mse, val_mae

def run_targetTL_model(x_target, y_target, x_test, y_test):

    model_source_target = model_target_lessdense()

    print(x_target.shape, x_test.shape)
    x_target, x_test = norm_by_std_nan(x_target, x_test)
    x_target = x_target.reshape(len(x_target), Input_width, Input_height, num_channels)
    x_test   = x_test.reshape(len(x_test), Input_width, Input_height, num_channels)
    print(x_target.shape, x_test.shape)

    history = model_source_target.fit(x_target, y_target, epochs=200, batch_size=tr_batch_size, validation_split=0.1)
    pred = model_source_target.predict(x_test, batch_size=ev_batch_size).reshape(len(x_test),)

    val_r2 = 1 - (np.sum(np.square(y_test - pred)) / np.sum(np.square(y_test - np.mean(y_test))))
    print("R2 version-1",val_r2)

    val_r2_v2 = pearsonr(y_test, pred)
    val_r2_v2 = (val_r2_v2[0])**2
    print("R2 version-2", val_r2_v2)

    val_mse = sqrt(mean_squared_error(y_test, pred))
    print("RMSE", val_mse)

    val_mae = mean_absolute_error(y_test, pred)
    print("MAE", val_mae)

    return val_r2, val_r2_v2, val_mse, val_mae

def run_target_model(x_target, y_target, x_test, y_test):
    model_target = get_model()

    print(x_target.shape, x_test.shape)
    x_target, x_test = norm_by_std_nan(x_target, x_test)
    x_target = x_target.reshape(len(x_target), Input_width, Input_height, num_channels)
    x_test   = x_test.reshape(len(x_test), Input_width, Input_height, num_channels)
    print(x_target.shape, x_test.shape)

    history = model_target.fit(x_target, y_target, epochs=200, batch_size=tr_batch_size, validation_split=0.1)
    pred = model_target.predict(x_test, batch_size=ev_batch_size).reshape(len(x_test),)

    val_r2 = 1 - (np.sum(np.square(y_test - pred)) / np.sum(np.square(y_test - np.mean(y_test))))
    print("R2 version-1",val_r2)

    val_r2_v2 = pearsonr(y_test, pred)
    val_r2_v2 = (val_r2_v2[0])**2
    print("R2 version-2", val_r2_v2)

    val_mse = sqrt(mean_squared_error(y_test, pred))
    print("RMSE", val_mse)

    val_mae = mean_absolute_error(y_test, pred)
    print("MAE", val_mae)

    return val_r2, val_r2_v2, val_mse, val_mae


################################################# Instance Transfer Learning Models #################################################

def run_tradaboost_model(source_x, source_y, target_x, target_y, test_x, test_y):
    from adapt.instance_based import TrAdaBoost, TrAdaBoostR2, TwoStageTrAdaBoostR2

    trada_params = {
    "max_leaf_nodes": 4,
    "max_depth": None,
    "min_samples_split": 5
    }

    params = dict(trada_params)

    model_tradaboost = TrAdaBoostR2(DecisionTreeRegressor(**params), n_estimators = 100, Xt = target_x, yt = target_y, random_state = 2)
    model_tradaboost.fit(source_x, source_y)

    pred_tradaboost = model_tradaboost.predict(test_x)

    r2_correlation_tradaboost = pearsonr(test_y, pred_tradaboost)
    r2_tradaboost = (r2_correlation_tradaboost[0])**2

    rmse_tradaboost = sqrt(mean_squared_error(test_y, pred_tradaboost))

    mae_tradaboost = mean_absolute_error(test_y, pred_tradaboost)

    return r2_tradaboost, rmse_tradaboost, mae_tradaboost


def run_adaboost_model(source_x, source_y, target_x, target_y, test_x, test_y):
    from sklearn.ensemble import AdaBoostRegressor

    ada_params = {
    "max_leaf_nodes": 4,
    "max_depth": None,
    "min_samples_split": 5,
    "random_state": 2,
    }

    params = dict(ada_params)

    model_adaboost = AdaBoostRegressor(DecisionTreeRegressor(**params), learning_rate = 0.1, n_estimators = 100)
    model_adaboost.fit(source_x, source_y)

    pred_adaboost = model_adaboost.predict(test_x)

    r2_correlation_adaboost = pearsonr(test_y, pred_adaboost)
    r2_adaboost = (r2_correlation_adaboost[0])**2

    rmse_adaboost = sqrt(mean_squared_error(test_y, pred_adaboost))

    mae_adaboost = mean_absolute_error(test_y, pred_adaboost)

    return r2_adaboost, rmse_adaboost, mae_adaboost

# ##### Kullback Leibler Importance Estimation
# def run_kliep_model(source_x, source_y, target_x, target_y, test_x, test_y):
#     from adapt.instance_based import KLIEP
#
#     model_kliep = KLIEP(DecisionTreeRegressor(max_depth = 6), Xt = target_x, kernel = "rbf", gamma = [0.1, 1.0], random_state = 0)
#     model_kliep.fit(source_x, source_y)
#
#     pred_kliep = model_kliep.predict(test_x)
#
#     r2_correlation_kliep = pearsonr(test_y, pred_kliep)
#     r2_kliep = (r2_correlation_kliep[0])**2
#
#     rmse_kliep = sqrt(mean_squared_error(test_y, pred_kliep))
#
#     mae_kliep = mean_absolute_error(test_y, pred_kliep)
#
#     return r2_kliep, rmse_kliep, mae_kliep

##### Kernel Mean Matching
def run_kmm_model(source_x, source_y, target_x, target_y, test_x, test_y):
    from adapt.instance_based import KMM

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
    r2_kmm = (r2_correlation_kmm[0])**2

    rmse_kmm = sqrt(mean_squared_error(test_y, pred_kmm))

    mae_kmm = mean_absolute_error(test_y, pred_kmm)

    return r2_kmm, rmse_kmm, mae_kmm

# ##### Linear Discrepancy Minimization
# def run_ldm_model(source_x, source_y, target_x, target_y, test_x, test_y):
#     from adapt.instance_based import LDM
#
#
#     model_ldm = LDM(Xt = target_x, random_state=0) #####DecisionTreeRegressor(max_depth = 4)
#     model_ldm.fit(source_x, source_y)
#
#     pred_ldm = model_ldm.predict(test_x)
#
#     r2_correlation_ldm = pearsonr(test_y, pred_ldm)
#     r2_ldm = (r2_correlation_ldm[0])**2
#
#     rmse_ldm = sqrt(mean_squared_error(test_y, pred_ldm))
#
#     mae_ldm = mean_absolute_error(test_y, pred_ldm)
#
#     return r2_ldm, rmse_ldm, mae_ldm

##### Nearest Neighbor Weighing
def run_nnw_model(source_x, source_y, target_x, target_y, test_x, test_y):
    from adapt.instance_based import NearestNeighborsWeighting

    nnw_params = {
    "max_leaf_nodes": 4,
    "max_depth": None,
    "min_samples_split": 5
    }

    params = dict(nnw_params)

    model_nnw = NearestNeighborsWeighting(DecisionTreeRegressor(**params), n_neighbors=10, Xt = target_x, random_state=2)
    model_nnw.fit(source_x, source_y)

    pred_nnw = model_nnw.predict(test_x)

    r2_correlation_nnw = pearsonr(test_y, pred_nnw)
    r2_nnw = (r2_correlation_nnw[0])**2

    rmse_nnw = sqrt(mean_squared_error(test_y, pred_nnw))

    mae_nnw = mean_absolute_error(test_y, pred_nnw)

    return r2_nnw, rmse_nnw, mae_nnw

##### Unconstrained Least-Squares Importance Fitting
def run_ulsif_model(source_x, source_y, target_x, target_y, test_x, test_y):
    from adapt.instance_based import ULSIF

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
    r2_ulsif = (r2_correlation_ulsif[0])**2

    rmse_ulsif = sqrt(mean_squared_error(test_y, pred_ulsif))

    mae_ulsif = mean_absolute_error(test_y, pred_ulsif)

    return r2_ulsif, rmse_ulsif, mae_ulsif

##### Relative Unconstrained Least-Squares Importance Fitting
def run_rulsif_model(source_x, source_y, target_x, target_y, test_x, test_y):
    from adapt.instance_based import RULSIF

    rulsif_params = {
    "max_leaf_nodes": 4,
    "max_depth": None,
    "min_samples_split": 5
    }

    params = dict(rulsif_params)

    model_rulsif = RULSIF(DecisionTreeRegressor(**params), kernel="rbf", alpha=0.1,
              lambdas=[0.1, 1., 10.], gamma=[0.1, 1., 10.], Xt = target_x, random_state=2)
    model_rulsif.fit(source_x, source_y)

    pred_rulsif = model_rulsif.predict(test_x)

    r2_correlation_rulsif = pearsonr(test_y, pred_rulsif)
    r2_rulsif = (r2_correlation_rulsif[0])**2

    rmse_rulsif = sqrt(mean_squared_error(test_y, pred_rulsif))

    mae_rulsif = mean_absolute_error(test_y, pred_rulsif)

    return r2_rulsif, rmse_rulsif, mae_rulsif

# ##### Importance Weighting Network
# def run_iwn_model(source_x, source_y, target_x, target_y, test_x, test_y):
#     from adapt.instance_based import IWN
#
#     model_iwn = IWN(DecisionTreeRegressor(max_depth = 6), sigma_init=0.1, pretrain=True,
#                     pretrain__epochs=100, pretrain__verbose=0, Xt = target_x, random_state=0)
#     model_iwn.fit(source_x, source_y, epochs=100, batch_size=256, verbose=1)
#
#     pred_iwn = model_iwn.predict(test_x)
#
#     r2_correlation_iwn = pearsonr(test_y, pred_iwn)
#     r2_iwn = (r2_correlation_iwn[0])**2
#
#     rmse_iwn = sqrt(mean_squared_error(test_y, pred_iwn))
#
#     mae_iwn = mean_absolute_error(test_y, pred_iwn)
#
#     return r2_iwn, rmse_iwn, mae_iwn

# ##### Weighting Adversarial Neural Network
# def run_wann_model(source_x, source_y, target_x, target_y, test_x, test_y):
#     from adapt.instance_based import WANN
#
#     model_wann = WANN(Xt = target_x, yt = target_y, random_state=0)
#     model_wann.fit(source_x, source_y, epochs=20, verbose=1)
#
#     pred_wann = model_wann.predict(test_x)
#
#     r2_correlation_wann = pearsonr(test_y, pred_wann)
#     r2_wann = (r2_correlation_wann[0])**2
#
#     rmse_wann = sqrt(mean_squared_error(test_y, pred_wann))
#
#     mae_wann = mean_absolute_error(test_y, pred_wann)
#
#     return r2_wann, rmse_wann, mae_wann


##### Domain adaptation using kernel mean matching and gradient boosting regression.
def run_kmm_gbr_model(source_x, source_y, target_x, target_y, test_x, test_y):
    from adapt.instance_based import KMM

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

    # Initialize the Gradient Boosting Regressor object
    model_kgbr = GradientBoostingRegressor(**params) ##learning_rate = 0.1, max_depth = 6, n_estimators = 100) ###, subsample=0.5)

    # Fit the Gradient Boosting Regressor object on the concatenated data
    model_kgbr.fit(X, y)

    pred_kgbr = model_kgbr.predict(test_x)

    r2_correlation_kgbr = pearsonr(test_y, pred_kgbr)
    r2_kgbr = (r2_correlation_kgbr[0])**2

    rmse_kgbr = sqrt(mean_squared_error(test_y, pred_kgbr))

    mae_kgbr = mean_absolute_error(test_y, pred_kgbr)

    return r2_kgbr, rmse_kgbr, mae_kgbr


def run_gwr_model(train_x, train_y, train_coords, test_x, test_y, test_coords):
    #Calibrate GWR model

    gwr_selector = Sel_BW(train_coords, train_y, train_x)
    gwr_bw = gwr_selector.search(bw_min=2)

    model = GWR(train_coords, train_y, train_x, gwr_bw)
    gwr_results = model.fit()

    scale = gwr_results.scale
    residuals = gwr_results.resid_response

    pred_gwr = model.predict(test_coords, test_x, scale, residuals)

    r2_correlation_gwr = pearsonr(test_y, pred_gwr)
    r2_gwr = (r2_correlation_gwr[0])**2

    rmse_gwr = sqrt(mean_squared_error(test_y, pred_gwr))

    mae_gwr = mean_absolute_error(test_y, pred_gwr)

    return r2_gwr, rmse_gwr, mae_gwr




############################### Processing functions ###############################

def split_train_validation(data_set, label_set, fold, k):
    """split train set and validation set"""
    quo = int(len(data_set) / k)
    x_train = np.delete(data_set, range(quo*fold,quo*(fold+1)), 0)
    y_train = np.delete(label_set, range(quo*fold,quo*(fold+1)), 0)
    x_test = data_set[quo*fold:quo*(fold+1)]
    y_test = label_set[quo*fold:quo*(fold+1)]
    return x_train, y_train, x_test, y_test

def norm_by_std_nan(train, val):
    mask = np.ma.array(train, mask=np.isnan(train))
    mean = np.mean(mask, 0)
    std = np.std(mask, 0)

    train = (train - mean) / std
    train = np.where(train == np.nan, 0, train)
    train = np.nan_to_num(train)

    val = (val-mean)/std
    val = np.where(val == np.nan, 0, val)
    val = np.nan_to_num(val)
    return train, val

def norm_by_std_nan_1arr(train):
    mask = np.ma.array(train, mask=np.isnan(train))
    mean = np.mean(mask, 0)
    std = np.std(mask, 0)

    train = (train - mean) / std
    train = np.where(train == np.nan, 0, train)
    train = np.nan_to_num(train)

    return train

def get_in_clusters(data_path: str, train_num: int):
    target_path = f"{data_path}tl-cal-{train_num}/"
    train_test_id = {}

    source_cmaq = np.load(f"{data_path}source_cmaq.npy")
    for set_id in range(10):
        target_cmaq = np.load(f"{target_path}split-{set_id}/target_cmaq.npz")
        train_cmaq = target_cmaq["train"]
        test_cmaq = target_cmaq["test"]
        train_test_id[set_id] = {
            "train_in_cluster":train_cmaq,
            "train_out_cluster":source_cmaq,
            "test_cluster":test_cmaq
        }
    return train_test_id

def _get_input_label(input_dt: pd.DataFrame, label_dt: pd.DataFrame, train_test_data_id: dict):
    all_inputs, all_labels = [], []
    for cluster_id in train_test_data_id.keys():
        cluster_test_cmaq = train_test_data_id[cluster_id]['test_cluster']
        all_labels.append(label_dt[np.isin(input_dt["cmaq_id"], cluster_test_cmaq)])
        all_inputs.append(input_dt.loc[np.isin(input_dt["cmaq_id"], cluster_test_cmaq)])
    all_labels = pd.concat(all_labels)
    all_inputs = pd.concat(all_inputs)
    return all_inputs, all_labels

def main():
    # x_source = np.load("US_data/us-2011/temp-cluster/source_cal.npy")
    # y_source = np.load("US_data/us-2011/temp-cluster/source_cal_label.npy")
    # run_source_model(x_source, y_source)

    tag_names = ['day', 'month', 'cmaq_x', 'cmaq_y', 'cmaq_id', 'rid', 'elev', 'forest_cover', 'pd', 'local', 'limi', 'high', 'is',
    'nldas_pevapsfc','nldas_pressfc', 'nldas_cape', 'nldas_ugrd10m', 'nldas_vgrd10m', 'nldas_tmp2m', 'nldas_rh2m', 'nldas_dlwrfsfc',
    'nldas_dswrfsfc', 'nldas_pcpsfc', 'nldas_fpcsfc', 'gc_aod', 'aod_value', 'emissi11_pm25', 'pm25_value'] 

    cols_to_scale = ['elev', 'forest_cover', 'pd', 'local', 'limi', 'high', 'is',
    'nldas_pevapsfc','nldas_pressfc', 'nldas_cape', 'nldas_ugrd10m', 'nldas_vgrd10m', 'nldas_tmp2m', 'nldas_rh2m', 'nldas_dlwrfsfc',
    'nldas_dswrfsfc', 'nldas_pcpsfc', 'nldas_fpcsfc', 'gc_aod', 'aod_value', 'emissi11_pm25']

    monitoring_df = pd.read_csv("/Users/shrey/Downloads/PM25/PM25-dnntransfer/US_data/BigUS/us_monitoring.csv")[tag_names]

    ss = StandardScaler()

    source_cmaq = np.load("/Users/shrey/Downloads/PM25/PM25-dnntransfer/US_data/split-data/single/source_cmaq.npy")
    source_df = monitoring_df[monitoring_df['cmaq_id'].isin(source_cmaq)]
    source_df = source_df.reset_index()

    source_df[cols_to_scale] = ss.fit_transform(source_df[cols_to_scale])

    source_x = source_df.drop(columns=["pm25_value"]).to_numpy()
    source_x = np.nan_to_num(source_x)
    source_y = source_df["pm25_value"].to_numpy()
    source_y = np.nan_to_num(source_y)

    mn_range_list = [5,10,15,20,30,40,50]

    for sp in mn_range_list:
        # myfile = open(f"US_data/us-2011/tl-cal-{sp}/results.txt", "a+")
        # tradaboost_file = open(f"results/tradaboost-results", "a+")
        # adaboost_file = open(f"results/adaboost-results", "a+")
        # kmm_file = open(f"results/kmm-results", "a+")
        # nnw_file = open(f"results/nnw-results", "a+")
        # ulsif_file = open(f"results/ulsif-results", "a+")
        # rulsif_file = open(f"results/rulsif-results", "a+")
        # kgbr_file = open(f"results/kgbr-results", "a+")
        #wann_file = open(f"wann-results", "a+")
        gwr_file = open(f"/Users/shrey/Downloads/PM25/PM25-dnntransfer/results/gwr-results", "a+")

        for idx in range(0,10):
            print(sp, idx)
            # x_target = np.load(f"US_data/us-2011/tl-cal-{sp}/split-{idx}/target_cal.npy")
            # y_target = np.load(f"US_data/us-2011/tl-cal-{sp}/split-{idx}/target_cal_label.npy")
            # x_test = np.load(f"US_data/us-2011/tl-cal-{sp}/split-{idx}/test_cal.npy")
            # y_test = np.load(f"US_data/us-2011/tl-cal-{sp}/split-{idx}/test_cal_label.npy")

            cmaq_ids = np.load(f"/Users/shrey/Downloads/PM25/PM25-dnntransfer/US_data/split-data/single/tl-cal-{sp}/split-{idx}/target_cmaq.npz")
            train_cmaq = cmaq_ids["train"]
            target_df = monitoring_df[monitoring_df['cmaq_id'].isin(train_cmaq)]
            target_df = target_df.reset_index()
            target_df[cols_to_scale] = ss.fit_transform(target_df[cols_to_scale])
            target_x = target_df.drop(columns=["pm25_value"]).to_numpy()
            target_x = np.nan_to_num(target_x)
            target_y = target_df["pm25_value"].to_numpy()
            target_y = np.nan_to_num(target_y)

            test_cmaq = cmaq_ids["test"]
            test_df = monitoring_df[monitoring_df['cmaq_id'].isin(test_cmaq)]
            test_df = test_df.reset_index()
            test_df[cols_to_scale] = ss.fit_transform(test_df[cols_to_scale])
            test_x = test_df.drop(columns=["pm25_value"]).to_numpy()
            test_x = np.nan_to_num(test_x)
            test_y = test_df["pm25_value"].to_numpy()
            test_y = np.nan_to_num(test_y)

            test_coords = list(zip(test_df.cmaq_x, test_df.cmaq_y))

            ############################ For GWR #######
            source_monitors = np.load("/Users/shrey/Downloads/PM25/PM25-dnntransfer/US_data/split-data/single/source_cmaq.npy").tolist()
            monitors = np.load(f"/Users/shrey/Downloads/PM25/PM25-dnntransfer/US_data/split-data/single/tl-cal-{sp}/split-{idx}/target_cmaq.npz")
            target_monitors = cmaq_ids["train"].tolist()

            train_cmaq = source_monitors + target_monitors
            train_df = monitoring_df[monitoring_df['cmaq_id'].isin(train_cmaq)]
            train_df = train_df.reset_index()

            train_df[cols_to_scale] = ss.fit_transform(train_df[cols_to_scale])

            train_x = train_df.drop(columns=["pm25_value"]).to_numpy()
            train_x = np.nan_to_num(train_x)
            train_y = train_df["pm25_value"].to_numpy()
            train_y = np.nan_to_num(train_y).reshape((-1,1))

            train_coords = list(zip(train_df.cmaq_x, train_df.cmaq_y))


            print(source_x.shape, source_y.shape, target_x.shape, target_y.shape, test_x.shape, test_y.shape)
            print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

            # val_r2_tgtTL, val_r2_v2_tgtTL, val_mse_tgtTL, val_mae_tgtTL = run_targetTL_model(x_target, y_target, x_test, y_test)
            # val_r2_tgt, val_r2_v2_tgt, val_mse_tgt, val_mae_tgt = run_target_model(x_target, y_target, x_test, y_test)

            # r2_tradaboost, rmse_tradaboost, mae_tradaboost = run_tradaboost_model(source_x, source_y, target_x, target_y, test_x, test_y)
            # r2_adaboost, rmse_adaboost, mae_adaboost = run_adaboost_model(source_x, source_y, target_x, target_y, test_x, test_y)
            # # r2_kliep, rmse_kliep, mae_kliep = run_kliep_model(source_x, source_y, target_x, target_y, test_x, test_y)
            # r2_kmm, rmse_kmm, mae_kmm = run_kmm_model(source_x, source_y, target_x, target_y, test_x, test_y)
            # # r2_ldm, rmse_ldm, mae_ldm = run_ldm_model(source_x, source_y, target_x, target_y, test_x, test_y)
            # r2_nnw, rmse_nnw, mae_nnw = run_nnw_model(source_x, source_y, target_x, target_y, test_x, test_y)
            # r2_ulsif, rmse_ulsif, mae_ulsif = run_ulsif_model(source_x, source_y, target_x, target_y, test_x, test_y)
            # r2_rulsif, rmse_rulsif, mae_rulsif = run_rulsif_model(source_x, source_y, target_x, target_y, test_x, test_y)
            # r2_iwn, rmse_iwn, mae_iwn = run_iwn_model(source_x, source_y, target_x, target_y, test_x, test_y)
            # r2_wann, rmse_wann, mae_wann = run_wann_model(source_x, source_y, target_x, target_y, test_x, test_y)
            # r2_kgbr, rmse_kgbr, mae_kgbr = run_kmm_gbr_model(source_x, source_y, target_x, target_y, test_x, test_y)

            new_test_y = test_y.reshape((-1,1))
            r2_gwr, rmse_gwr, mae_gwr = run_gwr_model(train_x, train_y, train_coords, test_x, new_test_y, test_coords)


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

            # myfile.write('{0}\t{1}\t{2}\t{3}\n'.format(val_r2_tgtTL, val_r2_v2_tgtTL, val_mse_tgtTL, val_mae_tgtTL))
            # myfile.write('{0}\t{1}\t{2}\t{3}\n'.format(val_r2_tgt, val_r2_v2_tgt, val_mse_tgt, val_mae_tgt))
            # file.write('=========================================================\n')

            # tradaboost_file.write('{0}\t{1}\t{2}\n'.format(r2_tradaboost, rmse_tradaboost, mae_tradaboost))
            # adaboost_file.write('{0}\t{1}\t{2}\n'.format(r2_adaboost, rmse_adaboost, mae_adaboost))
            # kmm_file.write('{0}\t{1}\t{2}\n'.format(r2_kmm, rmse_kmm, mae_kmm))
            # nnw_file.write('{0}\t{1}\t{2}\n'.format(r2_nnw, rmse_nnw, mae_nnw))
            # ulsif_file.write('{0}\t{1}\t{2}\n'.format(r2_ulsif, rmse_ulsif, mae_ulsif))
            # rulsif_file.write('{0}\t{1}\t{2}\n'.format(r2_rulsif, rmse_rulsif, mae_rulsif))
            # #wann_file.write('{0}\t{1}\t{2}\t{3}\n'.format(r2_wann, rmse_wann, mae_wann))
            # # kgbr_file.write('{0}\t{1}\t{2}\n'.format(r2_kgbr, rmse_kgbr, mae_kgbr))
            gwr_file.write('{0}\t{1}\t{2}\n'.format(r2_gwr, rmse_gwr, mae_gwr))


        # tradaboost_file.write('===================================================\n')
        # adaboost_file.write('===================================================\n')
        # kmm_file.write('===================================================\n')
        # nnw_file.write('===================================================\n')
        # ulsif_file.write('===================================================\n')
        # rulsif_file.write('===================================================\n')
        # # kgbr_file.write('===================================================\n')
        gwr_file.write('===================================================\n')

        # myfile.close()

if __name__ == "__main__":
    main()
