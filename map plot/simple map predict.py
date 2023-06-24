import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from data_process.data import tag_names
from data_process.pred_data import SimpleTrainCompose, SimplePredCompose
from data_process.spatial_validation import get_in_clusters
from model.adapt import TrainTest

def _transfer_train_predict(input_dt, label_dt, train_test_data_id, source_type, weight_average):
    if weight_average:
        model_name = "NNW_SWA"
    else:
        model_name = "NNW"
    save_dir = f"D:/prediction map/{source_type}/{model_name}/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    train_data_compose = SimpleTrainCompose(input_dt, label_dt, train_test_data_id, source_type, True, weight_average)
    model_train_test = TrainTest("NnwGbr")
    model_train_test.train({0:train_data_compose.source_dt}, {0:train_data_compose.train_target_dt})
    all_pred, all_label = model_train_test.predict({0:train_data_compose.valid_dt})
    print(r2_score(all_label["cluster0"], all_pred["cluster0"]))

    model_pred_compose = SimplePredCompose(source_type, {"mean":train_data_compose.mean, "std":train_data_compose.std}, True, weight_average)
    for date in range(1,366):
        target_date_data = pd.read_csv(f"D:/split-by-day/us-2011-satellite-target4-day-{date}.csv")[train_data_compose.columns]
        if weight_average:
            original_data_compose = SimpleTrainCompose(input_dt, label_dt, train_test_data_id, source_type, False, False)
            model_pred_compose.set_weight_model(target_cmaqs, original_data_compose.train_dt["input"], original_data_compose.train_dt["label"])
        composed_input = model_pred_compose.compose_input(target_date_data)
        prediction = model_train_test.predict({0:{"input":composed_input, "label":np.zeros(len(composed_input))}})[0]
        pred_val = pd.DataFrame(prediction["cluster0"], index=target_date_data["cmaq_id"], columns=["prediction"])
        pred_val.to_csv(f"{save_dir}date{date}_prediction.csv")
        # plt.scatter(target_date_data["cmaq_x"], target_date_data["cmaq_y"], c=prediction["cluster0"], s=5, cmap="rainbow", vmin=4, vmax=16)
        # plt.colorbar()
        # plt.show()

if __name__=='__main__':
    train_num = 15
    target_cluster = 4
    source_type = "whole"
    compose_data = False
    compose_path = f"data/split-data/tl-cal-{train_num}/"

    monitoring_whole_data = pd.read_csv("data/us_monitoring.csv")[tag_names]
    input_dt = monitoring_whole_data.drop(columns=["pm25_value"])
    label_dt = monitoring_whole_data["pm25_value"]
    target_cmaqs = pd.read_csv("D:/split-by-day/us-2011-satellite-target4-day-1.csv")[["cmaq_x", "cmaq_y", "cmaq_id"]].set_index("cmaq_id")
    train_test_data_id = get_in_clusters("data/split-data/", train_num)[0]

    for weight_average in [True, False]:
        _transfer_train_predict(input_dt, label_dt, train_test_data_id, source_type, weight_average)
    