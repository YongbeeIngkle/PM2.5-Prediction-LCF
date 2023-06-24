import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import torch
from data_process.data import tag_names, PredCompose, DataCompose
from data_process.spatial_validation import get_in_clusters
from model.autoencoder import TrainWholeModel
from model.adapt import TrainTest

def _train_autoencoder(data_compose: DataCompose, train_whole_model: TrainWholeModel, train_num, train_test_data_id, train_encoder: bool):
    """
    Train LCF autoencoder model.
    """
    if train_encoder:
        autoencode_train_dt, _ = data_compose.regression_data_convert_loader(True)
    else:
        autoencode_train_dt = {i:None for i in train_test_data_id.keys()}
    train_whole_model.train(autoencode_train_dt, train_num, 20)

def _encode_dataset(data_compose: DataCompose, train_whole_model: TrainWholeModel):
    """
    Output the LCF result of source, train-target, and test data.
    """
    autoencode_source_dt, autoencode_train_target_dt, autoencode_valid_dt = data_compose.regression_data_convert_loader(False)
    source_encode = train_whole_model.compose_feature(autoencode_source_dt)
    train_target_encode = train_whole_model.compose_feature(autoencode_train_target_dt)
    valid_encode = train_whole_model.compose_feature(autoencode_valid_dt)
    return source_encode, train_target_encode, valid_encode

def _combine_encode_input(train_test_data_id, source_type: str, source_encode: dict, train_target_encode: dict, valid_encode: dict):
    """
    Combine the LCF with original regression-objective predictors.
    """
    data_compose = DataCompose(input_dt, label_dt, train_test_data_id, source_type, True)
    source_input, train_target_input, valid_input = {}, {}, {}
    for split_id in data_compose.train_valid_data_id.keys():
        source_data = data_compose.source_dt[split_id]["input"]
        train_target_data = data_compose.train_target_dt[split_id]["input"]
        valid_data = data_compose.valid_dt[split_id]["input"]
        source_input[split_id] = np.hstack([source_data, source_encode[split_id]])
        train_target_input[split_id] = np.hstack([train_target_data, train_target_encode[split_id]])
        valid_input[split_id] = np.hstack([valid_data, valid_encode[split_id]])
    all_statistics = {"mean":data_compose.mean, "std":data_compose.std}
    return source_input, train_target_input, valid_input, all_statistics

def _transfer_train_predict(input_dt, label_dt, train_test_data_id, source_type, weight_average):
    if weight_average:
        model_name = "NNW_SWA"
    else:
        model_name = "NNW"
    save_dir = f"D:/prediction map/{model_name}/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    dict_id = {0:train_test_data_id}
    data_compose = DataCompose(input_dt, label_dt, dict_id, source_type, True, True, weight_average)
    model_train_test = TrainTest("NnwGbr")
    model_train_test.train(data_compose.source_dt, data_compose.train_target_dt)
    statistics = {"mean":data_compose.mean, "std":data_compose.std}

    pred_compose = PredCompose(input_dt, label_dt, target_cmaqs, train_test_data_id, source_type, True, statistics)
    for date in range(1,366):
        target_date_data = pd.read_csv(f"D:/split-by-day/us-2011-satellite-target4-day-{date}.csv")[tag_names[:-1]]
        if weight_average:
            composed_input = pred_compose.weight_average_data(target_date_data)
        else:
            composed_input = pred_compose.normalize_input(target_date_data)
        prediction = model_train_test.predict({0:{"input":composed_input, "label":np.zeros(len(composed_input))}})[0]
        pred_val = pd.DataFrame(prediction["cluster0"], index=target_date_data["cmaq_id"], columns=["prediction"])
        pred_val.to_csv(f"{save_dir}date{date}_prediction.csv")

def _LCF_train_predict(input_dt, label_dt, train_test_data_id, source_type, coord_pm, compose_path):
    compose_save = True
    target_compose_path = f"D:/target-encode/tl-cal-{train_num}/"
    if not os.path.exists(target_compose_path):
        os.makedirs(target_compose_path)
    if coord_pm:
        model_name = "LCF_coord"
        encoder_save_dir = "trained models/encode feature/east/"
    else:
        model_name = "LCF"
        encoder_save_dir = "trained models/whole encode feature/east/"
    save_dir = f"D:/prediction map/{model_name}/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    dict_id = {0:train_test_data_id}
    data_compose = DataCompose(input_dt, label_dt, dict_id, source_type, True, False, False, True, coord_pm, False, compose_path)

    # Define LCF composer (Autoencoder) model and train, output the LCF.
    train_whole_model = TrainWholeModel("LF", 0, data_compose.input_shape, encoder_save_dir)
    _train_autoencoder(data_compose, train_whole_model, train_num, train_test_data_id, True)
    source_encode, train_target_encode, valid_encode = _encode_dataset(data_compose, train_whole_model)
    train_whole_model.all_models = {0:torch.load(f"{encoder_save_dir}train{train_num}_split0")}
    source_encode, train_target_encode, valid_encode = _encode_dataset(data_compose, train_whole_model)

    # Define input data for transfer learning
    source_input, train_target_input, valid_input, regress_statistics = _combine_encode_input(dict_id, source_type, source_encode, train_target_encode, valid_encode)
    source_dt = {c:{"input":source_input[c], "label":data_compose.source_dt[c]["label"]} for c in source_encode.keys()}
    train_target_dt = {c:{"input":train_target_input[c], "label":data_compose.train_target_dt[c]["label"]} for c in train_target_encode.keys()}
    valid_dt = {c:{"input":valid_input[c], "label":data_compose.valid_dt[c]["label"]} for c in valid_encode.keys()}

    # Train transfer learning model and output the results
    model_train_test = TrainTest("NnwGbr")
    model_train_test.train(source_dt, train_target_dt)
    all_pred, all_label = model_train_test.predict(valid_dt)
    print(r2_score(all_label["cluster0"], all_pred["cluster0"]))

    encode_statistics = {"mean":data_compose.mean, "std":data_compose.std}
    pred_compose = PredCompose(input_dt, label_dt, target_cmaqs, train_test_data_id, source_type, True, regress_statistics)
    variable_cols = pred_compose.source_dt["input"].columns

    # valid_data_compose = DataCompose(input_dt, label_dt, dict_id, source_type, False)
    # valid_data = valid_data_compose.valid_dt[0]["input"]
    for date in range(1,366):
        print(f"date{date}")
        target_date_data = pd.read_csv(f"D:/split-by-day/us-2011-satellite-target4-day-{date}.csv")[variable_cols]
        target_date_data = target_date_data.set_index("cmaq_id", drop=False)
        encode_input = np.load(f"D:/target-encode/tl-cal-15/date{date}.npz")["dataset"]
        if coord_pm:
            encode_input = encode_input[:,[2,3,-1]]
        target_date_data = (target_date_data - regress_statistics["mean"]) / regress_statistics["std"]
        encode_input = (encode_input - encode_statistics["mean"]) / encode_statistics["std"]

        autoencode_target_dt = pred_compose.autoencode_data_convert_loader(encode_input)
        target_encode = train_whole_model.compose_feature({0:autoencode_target_dt})[0]
        composed_input = np.hstack([target_date_data, target_encode])

        prediction = model_train_test.predict({0:{"input":composed_input, "label":np.zeros(len(composed_input))}})[0]
        pred_val = pd.DataFrame(prediction["cluster0"], index=target_date_data.index, columns=["prediction"])
        pred_val.to_csv(f"{save_dir}date{date}_prediction.csv")
        # plt.scatter(target_date_data["cmaq_x"], target_date_data["cmaq_y"], c=prediction["cluster0"], s=5, cmap="rainbow", vmin=4, vmax=16)
        # plt.colorbar()
        # plt.show()

if __name__=='__main__':
    train_num = 15
    target_cluster = 4
    source_type = "east half"
    compose_data = False
    compose_path = f"data/split-data/tl-cal-{train_num}/"

    monitoring_whole_data = pd.read_csv("data/us_monitoring.csv")[tag_names]
    input_dt = monitoring_whole_data.drop(columns=["pm25_value"])
    label_dt = monitoring_whole_data["pm25_value"]
    target_cmaqs = pd.read_csv("D:/split-by-day/us-2011-satellite-target4-day-1.csv")[["cmaq_x", "cmaq_y", "cmaq_id"]].set_index("cmaq_id")
    train_test_data_id = get_in_clusters("data/split-data/", train_num)[0]

    # for weight_average in [False, True]:
    #     _transfer_train_predict(input_dt, label_dt, train_test_data_id, source_type, weight_average)
    for coord_pm in [False, True]:
        _LCF_train_predict(input_dt, label_dt, train_test_data_id, source_type, coord_pm, compose_path)

    