import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from data_process.data import tag_names
from data_process.pred_data import SimpleTrainCompose, SimplePredCompose, EncodeTrainCompose, EncodePredCompose, LcfTrainCompose, LcfPredCompose
from data_process.spatial_validation import get_in_clusters
from model.scikit import TrainTest
from model.autoencoder import TrainWholeModel

def _train_autoencoder(data_compose: EncodeTrainCompose, train_whole_model: TrainWholeModel, train_num, train_test_data_id, train_encoder: bool):
    """
    Train LCF autoencoder model.
    """
    if train_encoder:
        autoencode_train_loader, _ = data_compose.convert_loader(True)
        autoencode_train_dt = {0:autoencode_train_loader}
    else:
        autoencode_train_dt = {0:train_test_data_id}
    train_whole_model.train(autoencode_train_dt, train_num, 20)

def _encode_dataset(data_compose: EncodeTrainCompose, train_whole_model: TrainWholeModel):
    """
    Output the LCF result of source, train-target, and test data.
    """
    autoencode_source_dt, autoencode_train_target_dt, autoencode_valid_dt = data_compose.convert_loader(False)
    source_encode = train_whole_model.compose_feature({0:autoencode_source_dt})[0]
    train_target_encode = train_whole_model.compose_feature({0:autoencode_train_target_dt})[0]
    valid_encode = train_whole_model.compose_feature({0:autoencode_valid_dt})[0]
    return source_encode, train_target_encode, valid_encode

def _combine_encode_input(data_compose: LcfTrainCompose, source_encode: dict, train_target_encode: dict, valid_encode: dict):
    """
    Combine the LCF with original regression-objective predictors.
    """
    source_data = data_compose.source_dt["input"].copy()
    train_target_data = data_compose.train_target_dt["input"].copy()
    valid_data = data_compose.valid_dt["input"].copy()
    source_data["latent"] = source_encode
    train_target_data["latent"] = train_target_encode
    valid_data["latent"] = valid_encode
    return source_data, train_target_data, valid_data

def _rf_train_predict(input_dt, label_dt, train_test_data_id, source_type, coord_pm):
    save_dir = f"D:/prediction map/RF/LCF/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if coord_pm:
        encoder_save_dir = "trained models/encode feature/east/"
    else:
        encoder_save_dir = "trained models/whole encode feature/east/"
    encode_train_compose = EncodeTrainCompose(train_test_data_id, source_type, True, False, coord_pm)
    train_data_compose = SimpleTrainCompose(input_dt, label_dt, train_test_data_id, "whole", False, False)

    # Define LCF composer (Autoencoder) model and train, output the LCF.
    train_whole_model = TrainWholeModel("LF", 0, {0:encode_train_compose.input_shape}, encoder_save_dir)
    _train_autoencoder(encode_train_compose, train_whole_model, train_num, train_test_data_id, True)
    source_encode, train_target_encode, valid_encode = _encode_dataset(encode_train_compose, train_whole_model)
    
    # Define input data for transfer learning
    lcf_train_compose = LcfTrainCompose(input_dt, label_dt, train_test_data_id, source_type, True)
    source_input, train_target_input, valid_input = _combine_encode_input(lcf_train_compose, source_encode, train_target_encode, valid_encode)
    source_dt = {0:{"input":source_input, "label":lcf_train_compose.source_dt["label"]}}
    train_target_dt = {0:{"input":train_target_input, "label":lcf_train_compose.train_target_dt["label"]}}
    valid_dt = {0:{"input":valid_input, "label":lcf_train_compose.valid_dt["label"]}}

    # Train transfer learning model and output the results
    model_train_test = TrainTest("RF")
    model_train_test.train(train_target_dt)
    all_pred, all_label = model_train_test.predict(valid_dt)
    print(r2_score(all_label["cluster0"], all_pred["cluster0"]))

    encode_pred_compose = EncodePredCompose(source_type, {"mean":encode_train_compose.mean, "std":encode_train_compose.std}, True, coord_pm)
    lcf_pred_compose = LcfPredCompose(source_type, {"mean":lcf_train_compose.mean, "std":lcf_train_compose.std}, True, coord_pm)

    for date in range(1,366):
        target_date_data = pd.read_csv(f"D:/split-by-day/us-2011-satellite-target4-day-{date}.csv")[train_data_compose.columns]
        target_date_data = target_date_data.set_index("cmaq_id", drop=False)
        encode_input = encode_pred_compose.read_encoder_data(date)

        autoencode_target_dt = encode_pred_compose.convert_loader(encode_input)
        target_encode = train_whole_model.compose_feature({0:autoencode_target_dt})[0]
        composed_input = lcf_pred_compose.combine_input_latent(target_date_data, target_encode)
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

    _rf_train_predict(input_dt, label_dt, train_test_data_id, "east half", False)
    