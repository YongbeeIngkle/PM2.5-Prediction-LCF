import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_process.data import tag_names, PredCompose
from data_process.spatial_validation import get_clusters, ClusterGrid, TargetGrid

def _get_target_cmaqs(monitoring_whole_data, coord_whole_data, target_cluster):
    whole_cmaq = coord_whole_data.drop_duplicates().reset_index(drop=True)[['cmaq_x', 'cmaq_y', "cmaq_id"]]
    input_dt = monitoring_whole_data.drop(columns=["pm25_value"])
    label_dt = monitoring_whole_data["pm25_value"]
    _, cluster_model = get_clusters(input_dt, label_dt)
    whole_clsuter = cluster_model.predict(whole_cmaq[['cmaq_x', 'cmaq_y']])
    target_cmaq = whole_cmaq.iloc[whole_clsuter==target_cluster]
    return target_cmaq["cmaq_id"]

if __name__=='__main__':
    sp = 15
    target_cluster = 4
    monitoring_whole_data = pd.read_csv("data/us_monitoring.csv")[tag_names]
    coord_whole_data = pd.read_csv("data/largeUS_coords_pred.csv", index_col=0)
    target_cmaqs = _get_target_cmaqs(monitoring_whole_data, coord_whole_data, target_cluster)

    for date in range(1, 366):
        print(f"date {date}")
        whole_date_data = pd.read_csv(f"D:/split-by-day/us-2011-satellite-day-{date}.csv")[tag_names[:-1]]
        target_date_data = whole_date_data[np.isin(whole_date_data["cmaq_id"], target_cmaqs)]
        target_date_data.to_csv(f"D:/split-by-day/us-2011-satellite-target4-day-{date}.csv", index=False)
