import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from torch.utils.data import Dataset, DataLoader
from data_process.spatial_validation import get_clusters, ClusterGrid

tag_names = ['day', 'month', 'cmaq_x', 'cmaq_y', 'cmaq_id', 'rid', 'elev', 'forest_cover', 'pd', 'local', 'limi', 'high', 'is', 
    'nldas_pevapsfc','nldas_pressfc', 'nldas_cape', 'nldas_ugrd10m', 'nldas_vgrd10m', 'nldas_tmp2m', 'nldas_rh2m', 'nldas_dlwrfsfc', 
    'nldas_dswrfsfc', 'nldas_pcpsfc', 'nldas_fpcsfc', 'gc_aod', 'aod_value', 'emissi11_pm25', 'pm25_value']

def _convert_loader(input_dt:np.ndarray, output_dt:np.ndarray, batch:int):
    """
    Convert input-output dataset as torch DataLoader.

    input_dt: input dataset
    output_dt: output dataset
    batch: batch size
    """
    if len(input_dt) < 1:
        raise Exception("input_dt length is 0.")
    if len(output_dt) < 1:
        raise Exception("output_dt length is 0.")
    dt_set = InputOutputSet(input_dt, output_dt)
    dt_loader = DataLoader(dt_set, batch_size=batch, shuffle=False, pin_memory=True)
    return dt_loader

def _drop_constant_col(train_dt: pd.DataFrame):
    """
    Drop the constant columns, which has no information.

    train_dt: train dataset
    """
    _std = train_dt.std(axis=0)
    train_dt_variable = train_dt.loc[:,_std>0]
    return train_dt_variable.columns

def _drop_na_col(train_dt: pd.DataFrame):
    """
    Drop the columns with one or more nan value.

    train_dt: train dataset
    """
    train_drop_dt = train_dt.dropna(axis=1)
    return train_drop_dt.columns

def _drop_useless_col(source_data: pd.DataFrame, train_target_data: pd.DataFrame, valid_data: pd.DataFrame):
    """
    Drop useless columns (i.e. nan-including columns and constant columns) for source, train target, and validation dataset.
    Compose train dataset with combining source and train target dataset and remove the useless columns of source, train target, and validation dataset.

    source_data: source dataset
    train_target_data: train target dataset
    valid_data: validation dataset
    """
    train_data = pd.concat([source_data, train_target_data])
    train_cols = _drop_na_col(train_data)
    train_cols = _drop_constant_col(train_data[train_cols])
    source_drop_const = source_data[train_cols]
    train_drop_const = train_target_data[train_cols]
    valid_drop_const = valid_data[train_cols]
    return source_drop_const, train_drop_const, valid_drop_const

def _sort_distance_stations(distance_data: pd.DataFrame):
    """
    Sort the columns of distance matrix row by row.

    distance_data: distance matrix data
    """
    nearest_stations = pd.DataFrame(columns=range(distance_data.shape[1]), index=distance_data.index)
    for row in distance_data.index:
        nearest_stations.loc[row] = distance_data.columns[distance_data.loc[row].argsort()]
    return nearest_stations

def _unite_train_data(source_data: dict, train_target_data: dict):
    """
    Unite source and train target data as train data. Stack both input and label, respecitvely.

    source_data: source data of all splits
    train_target_data: target data of all splits
    """
    train_data = {}
    source_input = source_data["input"]
    train_target_input = train_target_data["input"]
    source_label = source_data["label"]
    train_target_label = train_target_data["label"]
    if type(source_input) == pd.DataFrame:
        train_data["input"] = pd.concat([source_input, train_target_input])
        train_data["label"] = pd.concat([source_label, train_target_label])
    else:
        train_data["input"] = np.vstack([source_input, train_target_input])
        train_data["label"] = np.hstack([source_label, train_target_label])
    return train_data

def create_distance_matrix(dt1: pd.DataFrame, dt2: pd.DataFrame):
    """
    Create distance matrix between two dataframe. If the distance is 0 (i.e. same coordinate), replace to inf.

    dt1: first data
    dt2: second data
    """
    all_distance = distance_matrix(dt1, dt2)
    all_distance[all_distance==0] = np.inf
    all_distance = pd.DataFrame(all_distance, index=dt1.index, columns=dt2.index)
    return all_distance

class WeightAverage:
    def __init__(self, source_data: pd.DataFrame, train_target_data: pd.DataFrame, valid_data: pd.DataFrame, train_data:pd.DataFrame, train_label: pd.DataFrame):
        """
        Compute weighted average.

        train_data: train input
        valid_data: validation input
        trian_label: train label
        """
        self.source_data = source_data
        self.train_target_data = train_target_data
        self.valid_data = valid_data
        self.train_data = train_data
        self.train_label = train_label
        self.cmaq_cols = ["cmaq_x", "cmaq_y", "cmaq_id"]
        self._allocate_weight()
        self._compute_all_wa()

    def _allocate_weight(self):
        """
        Allocate the weight (1/distance) for train data and validation data.
        """
        train_cmaq = pd.DataFrame(np.unique(self.train_data[self.cmaq_cols], axis=0), columns=self.cmaq_cols).set_index("cmaq_id")
        source_cmaq = pd.DataFrame(np.unique(self.source_data[self.cmaq_cols], axis=0), columns=self.cmaq_cols).set_index("cmaq_id")
        train_target_cmaq = pd.DataFrame(np.unique(self.train_target_data[self.cmaq_cols], axis=0), columns=self.cmaq_cols).set_index("cmaq_id")
        valid_cmaq = pd.DataFrame(np.unique(self.valid_data[self.cmaq_cols], axis=0), columns=self.cmaq_cols).set_index("cmaq_id")
        self.source_weight = 1 / create_distance_matrix(source_cmaq, train_cmaq)
        self.train_target_weight = 1 / create_distance_matrix(train_target_cmaq, train_cmaq)
        self.valid_weight = 1 / create_distance_matrix(valid_cmaq, train_cmaq)

    def _date_weight_average(self, data: pd.DataFrame, weight: pd.DataFrame, train_label: pd.DataFrame, train_cmaq):
        """
        Compute weighted average of given data for one date.

        data: input data for computing weighted average
        weight: weight for weighted average
        train_label: label of train data
        train_cmaq: cmaq id of train data
        """
        exist_weight = weight.loc[data["cmaq_id"], np.isin(weight.columns, train_cmaq)]
        weight_label = train_label[exist_weight.columns]
        weight_sum = np.sum(exist_weight, axis=1)
        cmaq_wa = np.sum(exist_weight*weight_label, axis=1)/weight_sum
        cmaq_wa.index = data.index
        return cmaq_wa
        
    def _compute_date_wa(self, date):
        """
        Compute weighted average of train and validation data for given date.

        date: weighted average computing date
        """
        date_train_data = self.train_data.loc[self.train_data["day"]==date].copy()
        date_source_data = self.source_data.loc[self.source_data["day"]==date].copy()
        date_train_target_data = self.train_target_data.loc[self.train_target_data["day"]==date].copy()
        date_valid_data = self.valid_data.loc[self.valid_data["day"]==date].copy()
        date_train_label = self.train_label.loc[self.train_data["day"]==date]
        date_train_label.index = date_train_data["cmaq_id"]
        source_wa = self._date_weight_average(date_source_data, self.source_weight, date_train_label, date_train_data["cmaq_id"])
        train_target_wa = self._date_weight_average(date_train_target_data, self.train_target_weight, date_train_label, date_train_data["cmaq_id"])
        valid_wa = self._date_weight_average(date_valid_data, self.valid_weight, date_train_label, date_train_data["cmaq_id"])
        date_source_data["pm25_wa"] = source_wa
        date_train_target_data["pm25_wa"] = train_target_wa
        date_valid_data["pm25_wa"] = valid_wa
        return date_source_data, date_train_target_data, date_valid_data

    def _compute_all_wa(self):
        """
        Compute weighted average of train and validation data for all dates.
        """
        all_dates = np.unique(self.train_data["day"])
        all_source_data, all_train_target_data, all_valid_data = [], [], []
        for date in all_dates:
            date_source, date_train_target, date_valid = self._compute_date_wa(date)
            all_source_data.append(date_source)
            all_train_target_data.append(date_train_target)
            all_valid_data.append(date_valid)
        all_source_data = pd.concat(all_source_data)
        all_train_target_data = pd.concat(all_train_target_data)
        all_valid_data = pd.concat(all_valid_data)
        self.weight_inputs = (all_source_data, all_train_target_data, all_valid_data)

class PredWeightAverage:
    def __init__(self, target_coord: pd.DataFrame, train_data:pd.DataFrame, train_label: pd.DataFrame):
        """
        Compute weighted average.

        train_data: train input
        target_coord: target coordination
        trian_label: train label
        """
        self.target_coord = target_coord
        self.train_data = train_data
        self.train_label = train_label
        self.cmaq_cols = ["cmaq_x", "cmaq_y", "cmaq_id"]
        self._allocate_weight()
        
    def _allocate_weight(self):
        """
        Allocate the weight (1/distance) for train data and validation data.
        """
        train_cmaq = pd.DataFrame(np.unique(self.train_data[self.cmaq_cols], axis=0), columns=self.cmaq_cols).set_index("cmaq_id")
        self.target_weight = 1 / create_distance_matrix(self.target_coord, train_cmaq)

    def _date_weight_average(self, data: pd.DataFrame, weight: pd.DataFrame, train_label: pd.DataFrame, train_cmaq):
        """
        Compute weighted average of given data for one date.

        data: input data for computing weighted average
        weight: weight for weighted average
        train_label: label of train data
        train_cmaq: cmaq id of train data
        """
        exist_weight = weight.loc[data["cmaq_id"], np.isin(weight.columns, train_cmaq)]
        weight_label = train_label[exist_weight.columns]
        weight_sum = np.sum(exist_weight, axis=1)
        cmaq_wa = np.sum(exist_weight*weight_label, axis=1)/weight_sum
        cmaq_wa.index = data.index
        return cmaq_wa
        
    def compute_date_wa(self, target_data: pd.DataFrame):
        """
        Compute weighted average of train and validation data for given date.

        date: weighted average computing date
        """
        pm_add_target_data = target_data.copy()
        date = target_data["day"].iloc[0]
        date_train_data = self.train_data.loc[self.train_data["day"]==date].copy()
        date_train_label = self.train_label.loc[self.train_data["day"]==date]
        date_train_label.index = date_train_data["cmaq_id"]
        target_data_wa = self._date_weight_average(target_data, self.target_weight, date_train_label, date_train_data["cmaq_id"])
        pm_add_target_data["pm25_wa"] = target_data_wa
        return pm_add_target_data

class InputOutputSet(Dataset):
    def __init__(self, input_dt, output_dt):
        super().__init__()
        self.input_dt = input_dt
        self.output_dt = output_dt

    def __getitem__(self, i):
        return self.input_dt[i], self.output_dt[i]

    def __len__(self):
        return len(self.input_dt)

class EncodeTrainCompose:
    def __init__(self, train_valid_data_id: dict, source_type:str, normalize=False, pm_normalize=False, coord_pm=False):
        self.train_valid_data_id = train_valid_data_id
        self.source_type = source_type
        self.pm_normalize = pm_normalize
        self.coord_pm = coord_pm
        self._allocate_near_data()
        if normalize:
            self._normalize_train_valid()
        # self.train_dt = _unite_train_data(self.source_dt, self.train_target_dt)

    def _normalize_train_valid(self):
        """
        Normalize the train (source and train target) and validation data.
        If standard deviation is 0, substitute to 1 to make the values 0 with just subtracting by mean.
        """
        source_input = self.source_dt["input"]
        train_target_input = self.train_target_dt["input"]
        valid_input = self.valid_dt["input"]
        train_input = np.vstack([source_input, train_target_input])
        mean, std = train_input.mean(axis=0), train_input.std(axis=0)
        std[std==0] = 1
        if not self.pm_normalize:
            mean[-1,:] = 0
            std[-1,:] = 1
        self.source_dt["input"] = (source_input - mean) / std
        self.train_target_dt["input"] = (train_target_input - mean) / std
        self.valid_dt["input"] = (valid_input - mean) / std
        self.mean, self.std = mean, std

    def _allocate_near_data(self):
        save_npz = np.load(f"data/split-data/tl-cal-15/split-0/nearest_dataset.npz")
        source_data, train_target_data, valid_data = save_npz["source_input"], save_npz["train_target_input"], save_npz["valid_input"]
        source_label, train_target_label, valid_label = save_npz["source_label"], save_npz["train_target_label"], save_npz["valid_label"]
        if self.source_type == "east half":
            east_index = (source_data[:,2, 6] > 0)
            source_data, source_label = source_data[east_index], source_label[east_index]
        if self.source_type == "east3":
            source_xy = pd.read_csv("data/east3 xy.csv")[["cmaq_x", "cmaq_y"]]
            east_index = np.isin(source_data[:,[2,3], 6], source_xy).all(axis=1)
            source_data, source_label = source_data[east_index], source_label[east_index]
        if self.coord_pm:
            source_data, train_target_data, valid_data = source_data[:,[2,3,-1]], train_target_data[:,[2,3,-1]], valid_data[:,[2,3,-1]]
        self.input_shape = source_data.shape[1:]
        self.source_dt = {"input": source_data, "label":source_label}
        self.train_target_dt = {"input":train_target_data, "label":train_target_label}
        self.valid_dt = {"input":valid_data, "label":valid_label}
        # self.train_dt = _unite_train_data(self.source_dt, self.train_target_dt)

    def convert_loader(self, train_unite: bool):
        source_input = np.array(self.source_dt["input"])
        source_label = np.array(self.source_dt["label"])
        train_target_input = np.array(self.train_target_dt["input"])
        train_target_label = np.array(self.train_target_dt["label"])
        valid_input = np.array(self.valid_dt["input"])
        valid_label = np.array(self.valid_dt["label"])
        if train_unite:
            train_input = np.vstack([source_input, train_target_input])
            train_label = np.hstack([source_label, train_target_label])
            train_loader = _convert_loader(train_input, train_label, 128)
        else:
            source_loader = _convert_loader(source_input, source_label, 128)
            train_target_loader = _convert_loader(train_target_input, train_target_label, 128)
        valid_loader = _convert_loader(valid_input, valid_label, 128)
        if train_unite:
            return train_loader, valid_loader
        else:
            return source_loader, train_target_loader, valid_loader

class EncodePredCompose:
    def __init__(self, source_type:str, statistics: dict, normalize=False, coord_pm=False):
        self.source_type = source_type
        self.statistics = statistics
        self.normalize = normalize
        self.coord_pm = coord_pm

    def read_encoder_data(self, date: int):
        target_data = np.load(f"D:/target-encode/tl-cal-15/date{date}.npz")["dataset"]
        if self.coord_pm:
            target_data = target_data[:,[2,3,-1]]
        if self.normalize:
            target_data = (target_data - self.statistics["mean"]) / self.statistics["std"]
        return target_data

    def convert_loader(self, dataset: np.ndarray):
        data_loader = _convert_loader(dataset, np.zeros(len(dataset)), 64)
        return data_loader

class LcfTrainCompose:
    def __init__(self, input_dt: pd.DataFrame, label_dt: pd.Series, train_valid_data_id: dict, source_type:str, normalize=False):
        self.input_dt = input_dt
        self.label_dt = label_dt
        self.train_valid_data_id = train_valid_data_id
        self.source_type = source_type
        self._split_train_valid_cmaq()
        if normalize:
            self._normalize_train_valid()
        
    def _split_train_valid_cmaq(self):
        """
        Split the train data and validation data using the cmaq id's (self.train_valid_data_id).
        There are three types of data, source (train_out), train target (train_in), and validation (test).
        """
        self.source_dt, self.train_target_dt, self.valid_dt = {}, {}, {}
        set_index = self.train_valid_data_id
        source_index, train_target_index, valid_index = set_index['train_out_cluster'], set_index['train_in_cluster'], set_index['test_cluster']
        source_input, source_label = self.input_dt.loc[np.isin(self.input_dt["cmaq_id"], source_index)], self.label_dt[np.isin(self.input_dt["cmaq_id"], source_index)]
        train_target_input, train_target_label = self.input_dt.loc[np.isin(self.input_dt["cmaq_id"], train_target_index)], self.label_dt[np.isin(self.input_dt["cmaq_id"], train_target_index)]
        valid_input, valid_label = self.input_dt.loc[np.isin(self.input_dt["cmaq_id"], valid_index)], self.label_dt[np.isin(self.input_dt["cmaq_id"], valid_index)]
        source_input, train_target_input, valid_input = _drop_useless_col(source_input, train_target_input, valid_input)
        if self.source_type == "east half":
            east_index = (source_input["cmaq_x"] > 0)
            source_input, source_label = source_input[east_index], source_label[east_index]
        if self.source_type == "east3":
            source_xy = pd.read_csv("data/east3 xy.csv")[["cmaq_x", "cmaq_y"]]
            east_index = np.isin(source_input[["cmaq_x", "cmaq_y"]], source_xy).all(axis=1)
            source_input, source_label = source_input[east_index], source_label[east_index]
        self.source_dt = {"input":source_input, "label": source_label}
        self.train_target_dt = {"input":train_target_input, "label": train_target_label}
        self.valid_dt = {"input":valid_input, "label":valid_label}
        self.columns = source_input.columns
        
    def _normalize_train_valid(self):
        """
        Normalize the train (source and train target) and validation data.
        If standard deviation is 0, substitute to 1 to make the values 0 with just subtracting by mean.
        """
        source_input = self.source_dt["input"]
        train_target_input = self.train_target_dt["input"]
        valid_input = self.valid_dt["input"]
        train_input = np.vstack([source_input, train_target_input])
        mean, std = train_input.mean(axis=0), train_input.std(axis=0)
        self.source_dt["input"] = (source_input - mean) / std
        self.train_target_dt["input"] = (train_target_input - mean) / std
        self.valid_dt["input"] = (valid_input - mean) / std
        self.mean, self.std = mean, std

class LcfPredCompose:
    def __init__(self, source_type:str, statistics: dict, normalize=False, coord_pm=False):
        self.source_type = source_type
        self.statistics = statistics
        self.normalize = normalize
        self.coord_pm = coord_pm

    def combine_input_latent(self, input_data: pd.DataFrame, latent_data: np.ndarray):
        input_copy = input_data.copy()
        if self.normalize:
            input_copy = (input_copy - self.statistics["mean"]) / self.statistics["std"]
        input_copy["latent"] = latent_data
        return input_copy

class SimpleTrainCompose:
    def __init__(self, input_dt: pd.DataFrame, label_dt: pd.Series, train_valid_data_id: dict, source_type:str, normalize=False, weight_average_compute=False):
        self.input_dt = input_dt
        self.label_dt = label_dt
        self.train_valid_data_id = train_valid_data_id
        self.source_type = source_type
        self.weight_average_compute = weight_average_compute
        self._split_train_valid_cmaq()
        if weight_average_compute:
            self._weight_average()
        if normalize:
            self._normalize_train_valid()
        self.train_dt = _unite_train_data(self.source_dt, self.train_target_dt)

    def _split_train_valid_cmaq(self):
        """
        Split the train data and validation data using the cmaq id's (self.train_valid_data_id).
        There are three types of data, source (train_out), train target (train_in), and validation (test).
        """
        self.source_dt, self.train_target_dt, self.valid_dt = {}, {}, {}
        set_index = self.train_valid_data_id
        source_index, train_target_index, valid_index = set_index['train_out_cluster'], set_index['train_in_cluster'], set_index['test_cluster']
        source_input, source_label = self.input_dt.loc[np.isin(self.input_dt["cmaq_id"], source_index)], self.label_dt[np.isin(self.input_dt["cmaq_id"], source_index)]
        train_target_input, train_target_label = self.input_dt.loc[np.isin(self.input_dt["cmaq_id"], train_target_index)], self.label_dt[np.isin(self.input_dt["cmaq_id"], train_target_index)]
        valid_input, valid_label = self.input_dt.loc[np.isin(self.input_dt["cmaq_id"], valid_index)], self.label_dt[np.isin(self.input_dt["cmaq_id"], valid_index)]
        source_input, train_target_input, valid_input = _drop_useless_col(source_input, train_target_input, valid_input)
        if self.source_type == "east half":
            east_index = (source_input["cmaq_x"] > 0)
            source_input, source_label = source_input[east_index], source_label[east_index]
        if self.source_type == "east3":
            source_xy = pd.read_csv("data/east3 xy.csv")[["cmaq_x", "cmaq_y"]]
            east_index = np.isin(source_input[["cmaq_x", "cmaq_y"]], source_xy).all(axis=1)
            source_input, source_label = source_input[east_index], source_label[east_index]
        self.source_dt = {"input":source_input, "label": source_label}
        self.train_target_dt = {"input":train_target_input, "label": train_target_label}
        self.valid_dt = {"input":valid_input, "label":valid_label}
        self.columns = source_input.columns
        
    def _normalize_train_valid(self):
        """
        Normalize the train (source and train target) and validation data.
        If standard deviation is 0, substitute to 1 to make the values 0 with just subtracting by mean.
        """
        source_input = self.source_dt["input"]
        train_target_input = self.train_target_dt["input"]
        valid_input = self.valid_dt["input"]
        train_input = np.vstack([source_input, train_target_input])
        mean, std = train_input.mean(axis=0), train_input.std(axis=0)
        self.source_dt["input"] = (source_input - mean) / std
        self.train_target_dt["input"] = (train_target_input - mean) / std
        self.valid_dt["input"] = (valid_input - mean) / std
        self.mean, self.std = mean, std

    def _weight_average(self):
        """
        Compute the weighted average for train and validation data. (Need update)
        """
        train_dt = _unite_train_data(self.source_dt, self.train_target_dt)
        train_input = train_dt["input"]
        train_label = train_dt["label"]
        source_input = self.source_dt["input"]
        train_target_input = self.train_target_dt["input"]
        valid_input = self.valid_dt["input"]
        weight_average = WeightAverage(source_input, train_target_input, valid_input, train_input, train_label)
        source_input, train_target_input, valid_input = weight_average.weight_inputs
        self.source_dt["input"] = source_input
        self.train_target_dt["input"] = train_target_input
        self.valid_dt["input"] = valid_input

class SimplePredCompose:
    def __init__(self, source_type:str, statistics: dict, normalize=False, weight_average_compute=False):
        self.source_type = source_type
        self.statistics = statistics
        self.normalize = normalize
        self.weight_average_compute = weight_average_compute
        
    def set_weight_model(self, target_coord, train_data, train_label):
        self.pred_weight = PredWeightAverage(target_coord, train_data, train_label)

    def weight_average(self, input_data: pd.DataFrame):
        weighted_data = self.pred_weight.compute_date_wa(input_data)
        return weighted_data
    
    def compose_input(self, input_data: pd.DataFrame):
        composed_data = input_data.copy()
        if self.weight_average_compute:
            composed_data = self.weight_average(composed_data)
        if self.normalize:
            composed_data = (composed_data - self.statistics["mean"]) / self.statistics["std"]
        return composed_data
