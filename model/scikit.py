import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

class TrainTest:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.hyperparameter = None

    def define_model(self):
        if self.model_name == 'RF':
            if self.hyperparameter is None:
                model = RandomForestRegressor(max_features=12, n_estimators=1000, random_state=1000)
            else:
                model = RandomForestRegressor(random_state=1000, **self.hyperparameter)
        elif self.model_name == "GBM":
            if self.hyperparameter is None:
                model = GradientBoostingRegressor(learning_rate=0.1, n_estimators=100, max_depth=6, random_state=1000)
            else:
                model = GradientBoostingRegressor(random_state=1000, **self.hyperparameter)
        return model

    def train(self, train_dataset: dict):
        self.all_models = {}
        for cluster_id in train_dataset.keys():
            print(f"Cluster{cluster_id} Regressor Train")
            input_dt = train_dataset[cluster_id]["input"]
            label_dt = train_dataset[cluster_id]["label"]
            model = self.define_model()
            model.fit(input_dt, label_dt)
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
