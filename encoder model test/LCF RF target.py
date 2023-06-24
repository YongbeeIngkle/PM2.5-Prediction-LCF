import numpy as np
import pandas as pd
from data_process.spatial_validation import get_in_clusters
from data_process.data import DataCompose, tag_names
from model.autoencoder import TrainWholeModel
from model.scikit import TrainTest
from result.save import save_split_results, save_accuracy

def _train_autoencoder(data_compose: DataCompose, train_whole_model: TrainWholeModel, train_num, train_test_data_id, train_encoder: bool):
    if train_encoder:
        autoencode_train_dt, _ = data_compose.regression_data_convert_loader(True)
    else:
        autoencode_train_dt = {i:None for i in train_test_data_id.keys()}
    train_whole_model.train(autoencode_train_dt, train_num, 20)

def _encode_dataset(data_compose: DataCompose, train_whole_model: TrainWholeModel):
    _, autoencode_train_dt, autoencode_valid_dt = data_compose.regression_data_convert_loader(False)
    train_encode = train_whole_model.compose_feature(autoencode_train_dt)
    valid_encode = train_whole_model.compose_feature(autoencode_valid_dt)
    return train_encode, valid_encode

def _combine_encode_input(train_test_data_id, source_type: bool, train_encode: dict, valid_encode: dict):
    data_compose = DataCompose(input_dt, label_dt, train_test_data_id, source_type, True)
    train_input, valid_input = {}, {}
    for split_id in data_compose.train_valid_data_id.keys():
        train_target_data = data_compose.train_target_dt[split_id]["input"]
        valid_data = data_compose.valid_dt[split_id]["input"]
        train_input[split_id] = np.hstack([train_target_data, train_encode[split_id]])
        valid_input[split_id] = np.hstack([valid_data, valid_encode[split_id]])
    return train_input, valid_input

def _one_cluster_model(model_name, train_num, input_dt, label_dt, coord_pm, source_type, save_preds=False):
    compose_data = False
    autoencoder_train = True
    data_path = "data/split-data/"
    compose_path = f"{data_path}tl-cal-{train_num}/"
    save_name = f"split_num{train_num}"
    model_save_name = f"{model_name}_{source_type}"
    if coord_pm:
        model_save_name = f"LCF_coord-{model_save_name}"
        encoder_save_dir = "trained models/encode feature/"
    else:
        model_save_name = f"LCF_whole-{model_save_name}"
        encoder_save_dir = "trained models/whole encode feature/"
    if "west" in source_type:
        encoder_save_dir = f"{encoder_save_dir}west/"
    elif "east" in source_type:
        encoder_save_dir = f"{encoder_save_dir}east/"

    train_test_data_id = get_in_clusters(data_path, train_num)
    # train_test_data_id = {i:train_test_data_id[i] for i in range(5)}
    data_compose = DataCompose(input_dt, label_dt, train_test_data_id, source_type, True, False, False, True, coord_pm, compose_data, compose_path)
    
    train_whole_model = TrainWholeModel("LF", 0, data_compose.input_shape, encoder_save_dir)
    _train_autoencoder(data_compose, train_whole_model, train_num, train_test_data_id, autoencoder_train)
    train_encode, valid_encode = _encode_dataset(data_compose, train_whole_model)
    train_input, valid_input = _combine_encode_input(train_test_data_id, source_type, train_encode, valid_encode)
    train_dt = {c:{"input":train_input[c], "label":data_compose.train_target_dt[c]["label"]} for c in train_encode.keys()}
    valid_dt = {c:{"input":valid_input[c], "label":data_compose.valid_dt[c]["label"]} for c in valid_encode.keys()}
    model_train_test = TrainTest(model_name)
    model_train_test.train(train_dt)
    all_pred, all_label = model_train_test.predict(valid_dt)

    if save_preds:
        save_split_results(all_pred, model_save_name, save_name)
    else:
        save_accuracy(all_label, all_pred, model_save_name, train_num)

if __name__=='__main__':
    model_name = "RF"
    for source_type in ["whole", "east half"]:
        for coord_pm in [True, False]:
            train_numbers = [5, 10, 15, 20, 30, 40, 50]

            monitoring_whole_data = pd.read_csv("data/us_monitoring.csv")[tag_names]
            input_dt = monitoring_whole_data.drop(columns=["pm25_value"])
            label_dt = monitoring_whole_data["pm25_value"]

            for train_num in train_numbers:
                _one_cluster_model(model_name, train_num, input_dt, label_dt, coord_pm, source_type)
