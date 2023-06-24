import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from adapt.instance_based import TrAdaBoostR2, KMM, KLIEP, RULSIF, NearestNeighborsWeighting

class TrainTest:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def _define_tradaboost(self, target_x, target_y):
        params = dict({
            "max_leaf_nodes": 4,
            "max_depth": None,
            "min_samples_split": 5
        })
        model = TrAdaBoostR2(
            DecisionTreeRegressor(**params), 
            n_estimators = 400, Xt = target_x, yt = target_y, random_state = 2
        )
        return model

    def _define_gbr(self):
        params = dict({
            "n_estimators": 400,
            "max_leaf_nodes": 4,
            "max_depth": None,
            "random_state": 2,
            "min_samples_split": 5,
            "learning_rate": 0.1,
            "subsample": 0.5
        })
        model = GradientBoostingRegressor(**params)
        return model
    
    def _compute_adapt(self, model_name: str, source_x, target_x):
        if model_name == "Kmm":
            weight_model = KMM(DecisionTreeRegressor(max_depth = 6), Xt=target_x, kernel="rbf", gamma=1., verbose=0, random_state=0)
        elif model_name == "Kliep":
            weight_model = KLIEP(DecisionTreeRegressor(max_depth = 6), Xt=target_x, kernel="rbf", gamma=1., verbose=0, random_state=0)
        elif model_name == "Rulsif":
            weight_model = RULSIF(DecisionTreeRegressor(max_depth = 6), kernel="rbf", alpha=0.1, lambdas=[0.1, 1., 10.], gamma=[0.1, 1., 10.], Xt = target_x, random_state=2)
        elif model_name == "Nnw":
            weight_model = NearestNeighborsWeighting(DecisionTreeRegressor(max_depth = 6), Xt=target_x, n_neighbors=6, random_state=0)
        weights = weight_model.fit_weights(source_x, target_x)
        weights = np.asarray(weights)
        weights = weights.reshape((weights.size, 1))
        source_adapt_x = source_x*weights
        return source_adapt_x

    def _train_model(self, source_x, source_y, target_x, target_y):
        if self.model_name == 'Tradaboost':
            model = self._define_tradaboost(target_x, target_y)
            input_data = source_x
            label_data = source_y
        elif "Gbr" in self.model_name:
            model = self._define_gbr()
            weight_model = self.model_name.split("Gbr")[0]
            source_adapt_x = self._compute_adapt(weight_model, source_x, target_x)
            if type(source_adapt_x) == pd.DataFrame:
                input_data = pd.concat([source_adapt_x, target_x])
                label_data = pd.concat([source_y, target_y])
            else:
                input_data = np.vstack([source_adapt_x, target_x])
                label_data = np.hstack([source_y, target_y])
        model.fit(input_data, label_data)
        return model

    def train(self, source_dataset: dict, train_target_dataset: dict):
        self.all_models = {}
        for cluster_id in source_dataset.keys():
            print(f"Cluster{cluster_id} Train")
            source_input = source_dataset[cluster_id]["input"]
            source_label = source_dataset[cluster_id]["label"]
            train_target_input = train_target_dataset[cluster_id]["input"]
            train_target_label = train_target_dataset[cluster_id]["label"]
            model = self._train_model(source_input, source_label, train_target_input, train_target_label)
            self.all_models[cluster_id] = model

    def predict(self, pred_dataset: dict):
        all_pred_vals, all_labels = {}, {}
        for cluster_id in pred_dataset.keys():
            input_dt = pred_dataset[cluster_id]["input"]
            label_dt = pred_dataset[cluster_id]["label"]
            pred_val = self.all_models[cluster_id].predict(input_dt)
            all_pred_vals[f"cluster{cluster_id}"] = pred_val
            all_labels[f"cluster{cluster_id}"] = np.array(label_dt)
            mse_val = mean_squared_error(np.array(label_dt), pred_val)
            print(f"MSE value: {mse_val}")
        return all_pred_vals, all_labels
