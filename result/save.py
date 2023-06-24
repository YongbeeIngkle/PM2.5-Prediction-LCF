import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def save_split_results(all_pred: dict, model_name: str, save_name: str):
    save_dir = f"result/{model_name}/prediction/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.savez(f"{save_dir}{save_name}.npz", **all_pred)

def save_accuracy(all_label, all_pred, model_name, train_num):
    save_dir = f"result/accuracy/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_name = f"{model_name}_mean_accuracy.csv"
    file_full_path = save_dir + file_name
    all_r2, all_r2_pearson, all_rmse, all_mae = [], [], [], []
    for cluster_id in all_label.keys():
        clsuter_label = all_label[cluster_id]
        cluster_pred = all_pred[cluster_id]
        r2 = r2_score(clsuter_label, cluster_pred)
        r2_pearson = pearsonr(clsuter_label, cluster_pred)[0]**2
        rmse = np.sqrt(mean_squared_error(clsuter_label, cluster_pred))
        mae = mean_absolute_error(clsuter_label, cluster_pred)
        all_r2.append(r2)
        all_r2_pearson.append(r2_pearson)
        all_rmse.append(rmse)
        all_mae.append(mae)
    all_accuracy = np.vstack([all_r2, all_r2_pearson, all_rmse, all_mae]).T
    mean_accuracy = all_accuracy.mean(axis=0)
    all_accuracy = np.vstack([all_accuracy, mean_accuracy])
    tuple_index = [(train_num, f"split{i}") for i in range(len(all_r2))] + [(train_num, "mean")]
    if os.path.exists(file_full_path):
        file_dt = pd.read_csv(file_full_path, index_col=0)
    else:
        file_dt = pd.DataFrame(columns=["Mean R2", "Mean R2 - Pearson", "Mean RMSE", "Mean MAE"], index=tuple_index) 
    file_dt = pd.concat([file_dt, pd.DataFrame(all_accuracy, columns=["Mean R2", "Mean R2 - Pearson", "Mean RMSE", "Mean MAE"], index=tuple_index)]).dropna(axis=0)
    file_dt.to_csv(file_full_path)
