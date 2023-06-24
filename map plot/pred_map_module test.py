import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import torch
from data_process.data import tag_names, DataCompose, PredCompose, StationAllocate, PredStationAllocate
from data_process.spatial_validation import get_in_clusters
from model.autoencoder import TrainWholeModel
from model.adapt import TrainTest

if __name__=='__main__':
    date = 365

    train_num = 15
    target_cluster = 4
    source_type = "east half"
    compose_data = False
    coord_pm = False
    compose_path = f"data/split-data/tl-cal-{train_num}/"

    monitoring_whole_data = pd.read_csv("data/us_monitoring.csv")[tag_names]
    input_dt = monitoring_whole_data.drop(columns=["pm25_value"])
    label_dt = monitoring_whole_data["pm25_value"]
    target_cmaqs = pd.read_csv(f"D:/split-by-day/us-2011-satellite-target4-day-{date}.csv")[["cmaq_x", "cmaq_y", "cmaq_id"]].set_index("cmaq_id")
    train_test_data_id = get_in_clusters("data/split-data/", train_num)[0]

    dict_id = {0:train_test_data_id}
    date_input = input_dt[input_dt["day"]==date]
    date_label = label_dt[input_dt["day"]==date]

    save_compose = DataCompose(input_dt, label_dt, dict_id, source_type, True, False, False, True, coord_pm, False, compose_path)
    save_date_valid = save_compose.valid_dt[0]["input"]
    save_date_valid = save_date_valid[save_date_valid[:,0,6]==max(save_date_valid[:,0,6])]
    # data_compose = DataCompose(date_input, date_label, dict_id, source_type, True, False, False, True, coord_pm, True, compose_path)
    # station_allocate = StationAllocate(
    #     data_compose.source_dt[0]["input"], data_compose.train_target_dt[0]["input"], data_compose.valid_dt[0]["input"], 
    #     data_compose.source_dt[0]["label"], data_compose.train_target_dt[0]["label"], data_compose.valid_dt[0]["label"], 12
    # )
    # _, _, valid_data = station_allocate.composed_inputs

    statistics = {"mean":save_compose.mean, "std":save_compose.std}
    # statistics = {"mean":0, "std":1}
    pred_compose = PredCompose(input_dt, label_dt, target_cmaqs, train_test_data_id, source_type, True, statistics)
    valid_data_compose = DataCompose(date_input, date_label, dict_id, source_type, False)
    target_data = pred_compose.allocate_near_data(valid_data_compose.valid_dt[0]["input"], False, True, f"D:/target-encode/tl-cal-{train_num}/")
    # station_allocate = PredStationAllocate(
    #     data_compose.source_dt[0]["input"], data_compose.train_target_dt[0]["input"], target_cmaqs, 
    #     data_compose.source_dt[0]["label"], data_compose.train_target_dt[0]["label"], 12)
    # target_data = station_allocate.compute_date_set(data_compose.valid_dt[0]["input"])
    a=3
