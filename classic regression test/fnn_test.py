import pandas as pd
from data_process.spatial_validation import get_in_clusters
from data_process.data import DataCompose, tag_names
from model.nn_regressor import RegressionTrainTest
from result.save import save_split_results, save_accuracy

def _one_cluster_model(model_name, train_num, input_dt, label_dt, weight_average, save_preds=False):
    data_path = "data/split-data/"
    save_name = f"split_num{train_num}"
    if weight_average:
        model_save_name = f"{model_name}_weighted_average"
    else:
        model_save_name = model_name

    train_test_data_id = get_in_clusters(data_path, train_num)
    data_compose = DataCompose(input_dt, label_dt, train_test_data_id, True, weight_average)
    train_dt, valid_dt = data_compose.regression_data_convert_loader()
    model_train_test = RegressionTrainTest(model_name, data_compose.input_shape)
    model_train_test.train(train_dt, 25)
    all_pred, all_label = model_train_test.predict(valid_dt)
    if save_preds:
        save_split_results(all_pred, model_save_name, save_name)
    else:
        save_accuracy(all_label, all_pred, model_save_name, train_num)

if __name__=='__main__':
    model_name = "FNN"
    weight_average = True
    train_numbers = [5, 10, 15, 20, 30, 40, 50]

    monitoring_whole_data = pd.read_csv("data/us_monitoring.csv")[tag_names]
    input_dt = monitoring_whole_data.drop(columns=["pm25_value"])
    label_dt = monitoring_whole_data["pm25_value"]

    for train_num in train_numbers:
        _one_cluster_model(model_name, train_num, input_dt, label_dt, weight_average)
