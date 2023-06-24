import pandas as pd
import numpy as np
import sklearn.metrics.pairwise as pairwise
import sklearn.datasets as datasets
# import xgboost as xgb
from sklearn import metrics
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from scipy.stats.stats import pearsonr
from math import sqrt
import statistics
from scipy.stats import *
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error



###### compute_mmd and gaussian_kernel calculated seperately.
# Define a function for computing the MMD between two distributions
def compute_mmd(Xs, Xt, kernel):
    Kxx = kernel(Xs, Xs)
    Kxy = kernel(Xs, Xt)
    Kyy = kernel(Xt, Xt)
    mmd = np.mean(Kxx) - 2 * np.mean(Kxy) + np.mean(Kyy)
    return mmd

# Define a kernel function
def gaussian_kernel(X, Y, sigma=1.0):
    pairwise_dists = pairwise.euclidean_distances(X, Y)
    K = np.exp(-pairwise_dists ** 2 / (2 * (sigma ** 2)))
    return K

####### used by mmd_rbf()
def rbf_kernel(X, Y, gamma):
    X_norm = np.sum(X ** 2, axis=-1)
    Y_norm = np.sum(Y ** 2, axis=-1)
    K = np.exp(-gamma * (X_norm[:, None] + Y_norm[None, :] - 2 * np.dot(X, Y.T)))
    return K

###### mmd code taken from: https://github.com/jindongwang/transferlearning/blob/master/code/distance/mmd_numpy_sklearn.py
def mmd_linear(X, Y):
    """MMD using linear kernel (i.e., k(x,y) = <x,y>)
    Note that this is not the original linear MMD, only the reformulated and faster version.
    The original version is:
        def mmd_linear(X, Y):
            XX = np.dot(X, X.T)
            YY = np.dot(Y, Y.T)
            XY = np.dot(X, Y.T)
            return XX.mean() + YY.mean() - 2 * XY.mean()
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
    Returns:
        [scalar] -- [MMD value]
    """
    delta = X.mean(0) - Y.mean(0)
    return delta.dot(delta.T)


def mmd_rbf(X, Y, gamma=1.0):
    """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})
    Returns:
        [scalar] -- [MMD value]
    """
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    # XX = rbf_kernel(X, X, gamma)
    # print(XX)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    # YY = rbf_kernel(Y, Y, gamma)
    # print(YY)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    # XY = rbf_kernel(X, Y, gamma)
    # print(XY)
    return XX.mean() + YY.mean() - 2 * XY.mean()


def mmd_poly(X, Y, degree=2, gamma=1, coef0=0):
    """MMD using polynomial kernel (i.e., k(x,y) = (gamma <X, Y> + coef0)^degree)
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
    Keyword Arguments:
        degree {int} -- [degree] (default: {2})
        gamma {int} -- [gamma] (default: {1})
        coef0 {int} -- [constant item] (default: {0})
    Returns:
        [scalar] -- [MMD value]
    """
    XX = metrics.pairwise.polynomial_kernel(X, X, degree, gamma, coef0)
    YY = metrics.pairwise.polynomial_kernel(Y, Y, degree, gamma, coef0)
    XY = metrics.pairwise.polynomial_kernel(X, Y, degree, gamma, coef0)
    return XX.mean() + YY.mean() - 2 * XY.mean()


def concrete_data():
    ConcreteData_df = pd.read_excel('/Users/shrey/Downloads/PM25/PM_25_TL/UCI_regression/Concrete/Concrete_Data.xls') ## 'Cement' found to be correlated at 0.4 :: 100
    drop_col_concrete = ['Cement']

    concrete_tgt_df = ConcreteData_df.loc[(ConcreteData_df['Cement'] <= 225)]
    concrete_tgt_df = concrete_tgt_df.drop(drop_col_concrete, axis = 1)
    concrete_tgt_df = concrete_tgt_df.reset_index(drop=True)

    concrete_source1_df = ConcreteData_df.loc[(ConcreteData_df['Cement'] > 225) & (ConcreteData_df['Cement'] <= 350)]
    concrete_source1_df = concrete_source1_df.drop(drop_col_concrete, axis = 1)
    concrete_source1_df = concrete_source1_df.reset_index(drop=True)

    concrete_source2_df = ConcreteData_df.loc[(ConcreteData_df['Cement'] > 350)]
    concrete_source2_df = concrete_source2_df.drop(drop_col_concrete, axis = 1)
    concrete_source2_df = concrete_source2_df.reset_index(drop=True)

    concrete_cols = concrete_tgt_df.columns.difference(['ConcreteCompressiveStrength'])

    ss = StandardScaler()
    concrete_tgt_df[concrete_cols] = ss.fit_transform(concrete_tgt_df[concrete_cols])
    concrete_source1_df[concrete_cols] = ss.fit_transform(concrete_source1_df[concrete_cols])
    concrete_source2_df[concrete_cols] = ss.fit_transform(concrete_source2_df[concrete_cols])

    concrete_source_df = pd.concat([concrete_source1_df, concrete_source2_df], ignore_index = True)
    concrete_source_df = concrete_source_df.reset_index(drop = True)

    return concrete_source_df, concrete_tgt_df

def synthetic_reg_data():
    # Load source and target data
    Xs, ys = datasets.make_regression(n_samples=1000, n_features=20, n_informative=10, noise=0.1, random_state=1)
    Xt, yt = datasets.make_regression(n_samples=500, n_features=20, n_informative=8, noise=0.2, random_state=2)

    return Xs, ys, Xt, yt

def us_data():
    tag_names = ['day', 'month', 'cmaq_x', 'cmaq_y', 'cmaq_id', 'rid', 'elev', 'forest_cover', 'pd', 'local', 'limi', 'high', 'is',
    'nldas_pevapsfc','nldas_pressfc', 'nldas_cape', 'nldas_ugrd10m', 'nldas_vgrd10m', 'nldas_tmp2m', 'nldas_rh2m', 'nldas_dlwrfsfc',
    'nldas_dswrfsfc', 'nldas_pcpsfc', 'nldas_fpcsfc', 'gc_aod', 'aod_value', 'emissi11_pm25', 'pm25_value_k', 'pm25_value']

    monitoring_df = pd.read_csv("US_data/BigUS/us_monitoring.csv")[tag_names]

    source_cmaq = np.load("US_data/split-data/single/source_cmaq.npy")
    source_df = monitoring_df[monitoring_df['cmaq_id'].isin(source_cmaq)]
    source_df = source_df.reset_index()
    Xs = source_df.drop(columns=["pm25_value"]).to_numpy()
    Xs = np.nan_to_num(Xs)
    ys = source_df["pm25_value"].to_numpy()
    ys = np.nan_to_num(ys)

    cmaq_ids = np.load(f"US_data/split-data/single/tl-cal-5/split-5/target_cmaq.npz")
    train_cmaq = cmaq_ids["train"]
    target_df = monitoring_df[monitoring_df['cmaq_id'].isin(train_cmaq)]
    target_df = target_df.reset_index()
    Xt = target_df.drop(columns=["pm25_value"]).to_numpy()
    Xt = np.nan_to_num(Xt)
    yt = target_df["pm25_value"].to_numpy()
    yt = np.nan_to_num(yt)

    return Xs, ys, Xt, yt


def main():

    # ##### Load synthetic regression data
    # Xs, ys, Xt, yt = synthetic_reg_data()
    # print(Xs.shape, Xt.shape)
    #
    #
    # ##### Load us split dataset
    # Xs, ys, Xt, yt = us_data()
    # print(Xs.shape, Xt.shape)

    ###### Load concrete data
    concrete_source_df, concrete_tgt_df = concrete_data()

    target_concrete = ['ConcreteCompressiveStrength']
    ys = concrete_source_df[target_concrete].to_numpy()
    Xs = concrete_source_df.drop(target_concrete, axis = 1).to_numpy()

    yt = concrete_tgt_df[target_concrete]
    Xt = concrete_tgt_df.drop(target_concrete, axis = 1)

    Xt, Xv, yt, yv = train_test_split(Xt, yt, test_size=0.80, random_state=42)
    yt = yt.to_numpy()
    Xt = Xt.to_numpy()
    yv = yv.to_numpy()
    yv = yv.ravel()
    Xv = Xv.to_numpy()

    print(Xs.shape, Xt.shape)

    # Compute the MMD between the source and target distributions
    # mmd = compute_mmd(Xs, Xt, gaussian_kernel)
    mmd = mmd_rbf(Xs, Xt)
    print(mmd)

    # Adapt the source data to the target data using MMD
    # Xs_new = Xs - np.mean(Xs, axis=0) + np.mean(Xt, axis=0)
    # mmd_adapted = compute_mmd(Xs_new, Xt, gaussian_kernel)

    Xs_adapted = np.dot(mmd, Xs)
    ys_adapted = np.dot(mmd, ys)
    print(Xs_adapted)

    # Concatenate the adapted source data and the target data
    X = np.concatenate((Xs_adapted, Xt), axis=0)
    # y = np.concatenate((ys, yt), axis=0)
    y = np.concatenate((ys_adapted, yt), axis=0)
    # y = np.concatenate((ys.ravel(), np.zeros(len(yt))), axis=0)


    # Define the domain label vector
    d = np.concatenate((np.zeros(Xs_adapted.shape[0]), np.ones(Xt.shape[0])), axis=0)

    # Train the gradient boosting regression with MMD regularization
    params = {'loss': 'squared_error', 'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'random_state': 42}
    model = GradientBoostingRegressor(**params)
    model.fit(X, y) ##, sample_weight=np.exp(-mmd * d))

    # Evaluate the model on the test data
    y_pred = model.predict(Xv)
    # mse = np.mean((yv - y_pred) ** 2)
    # print('MSE on adapted source data:', mse)

    r2_correlation_mmd = pearsonr(yv, y_pred)
    r2_mmd = (r2_correlation_mmd[0])**2
    print(r2_mmd)

    rmse_mmd = sqrt(mean_squared_error(yv, y_pred))
    print(rmse_mmd)

    mae_mmd = mean_absolute_error(yv, y_pred)
    print(mae_mmd)


    ########## Regular Gradient Boost
    # Concatenate the adapted source data and the target data
    X_reg = np.concatenate((Xs, Xt), axis=0)
    y_reg = np.concatenate((ys, yt), axis=0)

    # Train the gradient boosting regression with MMD regularization
    params = {'loss': 'squared_error', 'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'random_state': 42}
    model_regular = GradientBoostingRegressor(**params)
    model_regular.fit(X_reg, y_reg, sample_weight=np.exp(-mmd * d))

    # Evaluate the model on the test data
    y_pred_regular = model_regular.predict(Xv)
    # mse = np.mean((yv - y_pred) ** 2)
    # print('MSE on adapted source data:', mse)

    r2_correlation_regular = pearsonr(yv, y_pred_regular)
    r2_regular = (r2_correlation_regular[0])**2
    print(r2_regular)

    rmse_regular = sqrt(mean_squared_error(yv, y_pred_regular))
    print(rmse_regular)

    mae_regular = mean_absolute_error(yv, y_pred_regular)
    print(mae_regular)

    # sigma = 1.0
    # K_XX = rbf_kernel(Xs, Xs, gamma=1.0/(2.0*sigma**2))
    # K_XY = rbf_kernel(Xs, Xt, gamma=1.0/(2.0*sigma**2))
    #
    # # Compute the weights for the source samples
    # alpha = np.mean(K_XY, axis=1) - np.mean(K_XX, axis=1)
    #
    # # Scale the weights with the MMD value and the lambda parameter
    # lambda_param = 0.5
    # alpha = alpha / (mmd + lambda_param) ### param_grid = {'adaptation__lambda_param': [0.1, 1.0, 10.0]}
    # print(len(alpha))

    # Xs_adapted = Xs + alpha.dot(K_XX)

    # Xs_adapted = np.dot(alpha, Xs)
    # print(Xs_adapted)

    # ############ New MMD Model
    # from sklearn.metrics.pairwise import rbf_kernel
    # from scipy.optimize import minimize
    #
    #
    # K_source = rbf_kernel(Xs)
    # K_target = rbf_kernel(Xt)
    #
    # def loss(w):
    #     y_pred = np.dot(K_source, w)
    #     mse_source = mean_squared_error(ys, y_pred)
    #     return mse_source + mmd*w.dot(np.dot(K_target, w))
    #
    # # Initialize the weights for optimization
    # w0 = np.zeros(Xs.shape[0])
    #
    # # Optimize the loss function
    # res = minimize(loss, w0)
    #
    # # Get the optimized weights
    # w_opt = res.x
    #
    # # Use the optimized weights to fit the gradient boosting regressor
    # params = {'loss': 'squared_error', 'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'random_state': 42}
    # model_ngbm = GradientBoostingRegressor(**params)
    # model_ngbm.fit(Xs, ys - np.dot(K_source, w_opt))
    #
    # # Make predictions on the target samples
    # y_pred = model_ngbm.predict(Xt) + np.dot(K_target, w_opt)
    #
    # r2_correlation_mmd = pearsonr(yv, y_pred)
    # r2_mmd = (r2_correlation_mmd[0])**2
    # print(r2_mmd)
    #
    # rmse_mmd = sqrt(mean_squared_error(yv, y_pred))
    # print(rmse_mmd)
    #
    # mae_mmd = mean_absolute_error(yv, y_pred)
    # print(mae_mmd)

    #
    # # Evaluate the model on the target data
    # y_pred = model.predict(Xt)
    # mse = np.mean((yt - y_pred) ** 2)
    # print('MSE on target data:', mse)

if __name__ == "__main__":
    main()
