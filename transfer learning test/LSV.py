import pandas as pd
from data_process.spatial_validation import get_in_clusters
from data_process.data import DataCompose, tag_names
from model.autoencoder import TrainWholeModel
from model.adapt import TrainTest
from result.save import save_split_results, save_accuracy

def _train_autoencoder(data_compose: DataCompose, train_whole_model: TrainWholeModel, train_test_data_id, train_encoder: bool):
    if train_encoder:
        autoencode_train_dt, _ = data_compose.regression_data_convert_loader(True)
    else:
        autoencode_train_dt = {i:None for i in train_test_data_id.keys()}
    train_whole_model.train(autoencode_train_dt, 20)

def _encode_dataset(data_compose: DataCompose, train_whole_model: TrainWholeModel):
    autoencode_source_dt, autoencode_train_target_dt, autoencode_valid_dt = data_compose.regression_data_convert_loader(False)
    source_encode = train_whole_model.test_encode(autoencode_source_dt)
    train_target_encode = train_whole_model.test_encode(autoencode_train_target_dt)
    valid_encode = train_whole_model.test_encode(autoencode_valid_dt)
    return source_encode, train_target_encode, valid_encode

def _one_cluster_model(model_name, train_num, input_dt, label_dt, save_preds=False):
    compose_data = False
    autoencoder_train = True
    data_path = "data/split-data/"
    compose_path = f"{data_path}tl-cal-{train_num}/"
    save_name = f"split_num{train_num}"
    model_save_name = f"LSV-{model_name}"

    train_test_data_id = get_in_clusters(data_path, train_num)
    train_test_data_id = {i:train_test_data_id[i] for i in range(5)}
    data_compose = DataCompose(input_dt, label_dt, train_test_data_id, True, False, False, True, False, compose_data, compose_path)
    
    train_whole_model = TrainWholeModel("LSV", 2, data_compose.input_shape, "trained models/")
    _train_autoencoder(data_compose, train_whole_model, train_test_data_id, autoencoder_train)
    source_encode, train_target_encode, valid_encode = _encode_dataset(data_compose, train_whole_model)
    source_dt = {c:{"input":source_encode[c], "label":data_compose.source_dt[c]["label"]} for c in source_encode.keys()}
    train_target_dt = {c:{"input":train_target_encode[c], "label":data_compose.train_target_dt[c]["label"]} for c in train_target_encode.keys()}
    valid_dt = {c:{"input":valid_encode[c], "label":data_compose.valid_dt[c]["label"]} for c in valid_encode.keys()}

    model_train_test = TrainTest(model_name)
    model_train_test.train(source_dt, train_target_dt)
    all_pred, all_label = model_train_test.predict(valid_dt)
    if save_preds:
        save_split_results(all_pred, model_save_name, save_name)
    else:
        save_accuracy(all_label, all_pred, model_save_name, train_num)

if __name__=='__main__':
    # all_models = ["NnwGbr", "RulsifGbr"]
    all_models = ["NnwGbr"]
    for model_name in all_models:
        train_numbers = [5, 10, 15, 20, 30, 40, 50]

        monitoring_whole_data = pd.read_csv("data/us_monitoring.csv")[tag_names]
        input_dt = monitoring_whole_data.drop(columns=["pm25_value"])
        label_dt = monitoring_whole_data["pm25_value"]

        for train_num in train_numbers:
            _one_cluster_model(model_name, train_num, input_dt, label_dt)
