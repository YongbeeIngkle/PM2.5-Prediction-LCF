import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from data_process.data import tag_names
from data_process.pred_data import SimpleTrainCompose, SimplePredCompose
from data_process.spatial_validation import get_in_clusters
from model.scikit import TrainTest

def _unite_target_data(train_data, valid_data):
    unite_input = pd.concat([train_data["input"], valid_data["input"]])
    unite_label = pd.concat([train_data["label"], valid_data["label"]])
    return {"input":unite_input, "label":unite_label}

def _rf_whole_train_predict(input_dt, label_dt, train_test_data_id):
    save_dir = f"D:/prediction map/RF/whole/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    train_data_compose = SimpleTrainCompose(input_dt, label_dt, train_test_data_id, "whole", False, False)
    model_train_test = TrainTest("RF")
    train_data = _unite_target_data(train_data_compose.train_target_dt, train_data_compose.valid_dt)
    model_train_test.train({0:train_data})
    all_pred, all_label = model_train_test.predict({0:train_data_compose.valid_dt})
    print(r2_score(all_label["cluster0"], all_pred["cluster0"]))

    model_pred_compose = SimplePredCompose("whole", {"mean":0, "std":1}, False)
    for date in range(1,366):
        target_date_data = pd.read_csv(f"D:/split-by-day/us-2011-satellite-target4-day-{date}.csv")[train_data_compose.columns]
        composed_input = model_pred_compose.compose_input(target_date_data)
        prediction = model_train_test.predict({0:{"input":composed_input, "label":np.zeros(len(composed_input))}})[0]
        pred_val = pd.DataFrame(prediction["cluster0"], index=target_date_data["cmaq_id"], columns=["prediction"])
        pred_val.to_csv(f"{save_dir}date{date}_prediction.csv")

def _rf_train_predict(input_dt, label_dt, train_test_data_id):
    save_dir = f"D:/prediction map/RF/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    train_data_compose = SimpleTrainCompose(input_dt, label_dt, train_test_data_id, "whole", False, False)
    model_train_test = TrainTest("RF")
    model_train_test.train({0:train_data_compose.train_target_dt})
    all_pred, all_label = model_train_test.predict({0:train_data_compose.valid_dt})
    print(r2_score(all_label["cluster0"], all_pred["cluster0"]))

    model_pred_compose = SimplePredCompose("whole", {"mean":0, "std":1}, False)
    for date in range(1,366):
        target_date_data = pd.read_csv(f"D:/split-by-day/us-2011-satellite-target4-day-{date}.csv")[train_data_compose.columns]
        composed_input = model_pred_compose.compose_input(target_date_data)
        prediction = model_train_test.predict({0:{"input":composed_input, "label":np.zeros(len(composed_input))}})[0]
        pred_val = pd.DataFrame(prediction["cluster0"], index=target_date_data["cmaq_id"], columns=["prediction"])
        pred_val.to_csv(f"{save_dir}date{date}_prediction.csv")
        # plt.scatter(target_date_data["cmaq_x"], target_date_data["cmaq_y"], c=prediction["cluster0"], s=5, cmap="rainbow", vmin=4, vmax=16)
        # plt.colorbar()
        # plt.show()

if __name__=='__main__':
    train_num = 15
    compose_path = f"data/split-data/tl-cal-{train_num}/"

    monitoring_whole_data = pd.read_csv("data/us_monitoring.csv")[tag_names]
    input_dt = monitoring_whole_data.drop(columns=["pm25_value"])
    label_dt = monitoring_whole_data["pm25_value"]
    train_test_data_id = get_in_clusters("data/split-data/", train_num)[0]

    # _rf_train_predict(input_dt, label_dt, train_test_data_id)
    _rf_whole_train_predict(input_dt, label_dt, train_test_data_id)
    