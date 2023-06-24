import pandas as pd
import numpy as np
from data_process.spatial_validation import get_in_clusters
from data_process.data import DataCompose, tag_names
from model.adapt import TrainTest
from result.save import save_split_results, save_accuracy

def _extract_swa(data_dict):
    swa_dict = {}
    for set_id in data_dict.keys():
        swa_dict[set_id] = data_dict[set_id]
        swa_dict[set_id]["input"] = np.array(data_dict[set_id]["input"]["pm25_wa"]).reshape(-1,1)
    return swa_dict

def _one_cluster_model(model_name, train_num, input_dt, label_dt, source_type, save_preds=False):
    data_path = "data/split-data/"
    save_name = f"split_num{train_num}"
    model_save_name = f"{model_name} SWA only"
    model_save_name = f"{model_save_name}_{source_type}"
    
    train_test_data_id = get_in_clusters(data_path, train_num)
    data_compose = DataCompose(input_dt, label_dt, train_test_data_id, source_type, True, True, True)
    model_train_test = TrainTest(model_name)
    source_dt = _extract_swa(data_compose.source_dt)
    train_target_dt = _extract_swa(data_compose.train_target_dt)
    valid_dt = _extract_swa(data_compose.valid_dt)
    model_train_test.train(source_dt, train_target_dt)
    all_pred, all_label = model_train_test.predict(valid_dt)
    if save_preds:
        save_split_results(all_pred, model_save_name, save_name)
    else:
        save_accuracy(all_label, all_pred, model_save_name, train_num)

if __name__=='__main__':
    all_models = ["NnwGbr"]
    for source_type in ["east half"]:
        for model_name in all_models:
            train_numbers = [5, 10, 15, 20, 30, 40, 50]

            monitoring_whole_data = pd.read_csv("data/us_monitoring.csv")[tag_names]
            input_dt = monitoring_whole_data.drop(columns=["pm25_value"])
            label_dt = monitoring_whole_data["pm25_value"]

            for train_num in train_numbers:
                _one_cluster_model(model_name, train_num, input_dt, label_dt, source_type)
