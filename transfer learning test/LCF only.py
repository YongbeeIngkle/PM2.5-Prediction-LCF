import numpy as np
import pandas as pd
from data_process.spatial_validation import get_in_clusters
from data_process.data import DataCompose, tag_names
from model.autoencoder import TrainWholeModel
from model.adapt import TrainTest
from result.save import save_split_results, save_accuracy

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
    return source_input, train_target_input, valid_input

def _one_cluster_model(model_name: str, train_num: int, input_dt, label_dt, coord_pm: bool, source_type: bool, save_preds=False):
    """
    One cluster (California-Nevada) target model running function.

    model_name: name of model - One of ["NnwGbr", "RulsifGbr", "Kmm", "Kliep"].
    train_num: number of train-target monitoring station number.
    input_dt: original input data. Not used if compose_data=False.
    label_dt: original label data Not used if compose_data=False.
    coord_pm: whether only use coordinate and PM2.5 or whole data for LCF.
    east_source: whether use the east or whole area source data use.
    save_preds: save whole prediction or just accuracy.
    """
    compose_data = False
    autoencoder_train = True

    # Define the path to save results
    data_path = "data/split-data/"
    compose_path = f"{data_path}tl-cal-{train_num}/"
    save_name = f"split_num{train_num}"
    model_name = f"{model_name}_{source_type}"
    if coord_pm:
        model_save_name = f"LCF_only_coord-{model_name}"
        encoder_save_dir = "trained models/encode feature/"
    else:
        model_save_name = f"LCF_only_whole-{model_name}"
        encoder_save_dir = "trained models/whole encode feature/"
    if "west" in source_type:
        encoder_save_dir = f"{encoder_save_dir}west/"
    elif "east" in source_type:
        encoder_save_dir = f"{encoder_save_dir}east/"
    
    # Get cluster information and define data composer module 
    train_test_data_id = get_in_clusters(data_path, train_num)
    data_compose = DataCompose(input_dt, label_dt, train_test_data_id, source_type, True, False, False, True, coord_pm, compose_data, compose_path)
    
    # Define LCF composer (Autoencoder) model and train, output the LCF.
    train_whole_model = TrainWholeModel("LF", 0, data_compose.input_shape, encoder_save_dir)
    _train_autoencoder(data_compose, train_whole_model, train_num, train_test_data_id, autoencoder_train)
    source_encode, train_target_encode, valid_encode = _encode_dataset(data_compose, train_whole_model)

    # Define input data for transfer learning
    source_dt = {c:{"input":source_encode[c], "label":data_compose.source_dt[c]["label"]} for c in source_encode.keys()}
    train_target_dt = {c:{"input":train_target_encode[c], "label":data_compose.train_target_dt[c]["label"]} for c in train_target_encode.keys()}
    valid_dt = {c:{"input":valid_encode[c], "label":data_compose.valid_dt[c]["label"]} for c in valid_encode.keys()}

    # Train transfer learning model and output the results
    model_train_test = TrainTest(model_name)
    model_train_test.train(source_dt, train_target_dt)
    all_pred, all_label = model_train_test.predict(valid_dt)
    if save_preds:
        save_split_results(all_pred, model_save_name, save_name)
    else:
        save_accuracy(all_label, all_pred, model_save_name, train_num)

if __name__=='__main__':
    all_models = ["KliepGbr"]
    for source_type in ["whole"]:
        for coord_pm in [True, False]:
            for model_name in all_models:
                train_numbers = [5, 10, 15, 20, 30, 40, 50]

                monitoring_whole_data = pd.read_csv("data/us_monitoring.csv")[tag_names]
                input_dt = monitoring_whole_data.drop(columns=["pm25_value"])
                label_dt = monitoring_whole_data["pm25_value"]

                for train_num in train_numbers:
                    _one_cluster_model(model_name, train_num, input_dt, label_dt, coord_pm, source_type)
