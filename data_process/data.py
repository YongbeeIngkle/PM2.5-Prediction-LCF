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
    for split_id in source_data.keys():
        train_data[split_id] = {}
        source_input = source_data[split_id]["input"]
        train_target_input = train_target_data[split_id]["input"]
        source_label = source_data[split_id]["label"]
        train_target_label = train_target_data[split_id]["label"]
        if type(source_input) == pd.DataFrame:
            train_data[split_id]["input"] = pd.concat([source_input, train_target_input])
            train_data[split_id]["label"] = pd.concat([source_label, train_target_label])
        else:
            train_data[split_id]["input"] = np.vstack([source_input, train_target_input])
            train_data[split_id]["label"] = np.hstack([source_label, train_target_label])
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

class StationAllocate:
    def __init__(self, source_data: pd.DataFrame, train_target_data: pd.DataFrame, valid_data: pd.DataFrame, 
                 source_label: pd.DataFrame, train_target_label: pd.DataFrame, valid_label: pd.DataFrame, station_num: int):
        """
        Allocate nearest monitoring station dataset.

        source_data: source input data
        train_target_data: train target input data
        valid_data: validation input data
        source_label: source label data
        train_target_label: train target label
        valid_label: validation label
        station_num: number of allocating nearest monitoring data
        """
        self.source_data = source_data
        self.train_target_data = train_target_data
        self.valid_data = valid_data
        self.source_label = source_label
        self.train_target_label = train_target_label
        self.valid_label = valid_label
        self.station_num = station_num
        self.cmaq_cols = ["cmaq_x", "cmaq_y", "cmaq_id"]
        self._compute_distances()
        self.source_sort_stations = _sort_distance_stations(self.source_distance)
        self.train_target_sort_stations = _sort_distance_stations(self.train_target_distance)
        self.valid_sort_stations = _sort_distance_stations(self.valid_distance)
        self._allocate_all_data()

    def _compute_distances(self):
        """
        Compute distance matrix of source, train target, and validation data.
        """
        source_cmaq = pd.DataFrame(np.unique(self.source_data[self.cmaq_cols], axis=0), columns=self.cmaq_cols).set_index("cmaq_id")
        train_target_cmaq = pd.DataFrame(np.unique(self.train_target_data[self.cmaq_cols], axis=0), columns=self.cmaq_cols).set_index("cmaq_id")
        train_cmaq = pd.concat([source_cmaq, train_target_cmaq])
        valid_cmaq = pd.DataFrame(np.unique(self.valid_data[self.cmaq_cols], axis=0), columns=self.cmaq_cols).set_index("cmaq_id")
        self.source_distance = create_distance_matrix(source_cmaq, train_cmaq)
        self.train_target_distance = create_distance_matrix(train_target_cmaq, train_cmaq)
        self.valid_distance = create_distance_matrix(valid_cmaq, train_cmaq)

    def _date_allocate_data(self, data: pd.DataFrame, train_data: pd.DataFrame, sort_stations: pd.DataFrame, train_label: pd.DataFrame):
        """
        Allocate nearest monitoring station data of given data for one date.
        First, Define the PM2.5 of given data as 0 and train monitoring statoin data as true PM2.5.
        Second, allocate the existing station cmaq_id's on the date.
        Finally, allocate the train data of corresponding station cmaq_id's.

        data: input data for allocating
        train_data: train input data
        sort_stations: sorted stations of given data
        train_label: train label data
        """
        cmaq_id_data = data.set_index("cmaq_id")
        cmaq_id_data["PM25"] = 0
        cmaq_id_train = train_data.set_index("cmaq_id")
        cmaq_id_train["PM25"] = train_label
        date_stations = sort_stations.loc[cmaq_id_data.index]

        date_exist_stations = []
        for id in date_stations.index:
            row_station = date_stations.loc[id]
            row_exist_stations = row_station[np.isin(row_station, cmaq_id_train.index)].reset_index(drop=True)
            date_exist_stations.append(row_exist_stations)
        date_exist_stations = pd.concat(date_exist_stations, axis=1).T

        station_data = []
        for s in range(self.station_num):
            near_data = cmaq_id_train.loc[date_exist_stations[s]]
            station_data.append(near_data)
        station_data.insert(len(station_data)//2, cmaq_id_data)
        stack_station_data = np.stack(station_data, -1)
        return stack_station_data
        
    def _compute_date_set(self, date: int):
        """
        Compute (allocate) the nearest monitoring data-combined input dataset of source, train target, and validation dataset for given date.

        date: date to allocate
        """
        date_source_data = self.source_data.loc[self.source_data["day"]==date].copy()
        date_train_target_data = self.train_target_data.loc[self.train_target_data["day"]==date].copy()
        date_valid_data = self.valid_data.loc[self.valid_data["day"]==date].copy()
        date_source_label = self.source_label.loc[date_source_data.index]
        date_train_target_label = self.train_target_label.loc[date_train_target_data.index]
        date_valid_label = self.valid_label.loc[date_valid_data.index]
        date_train_data = pd.concat([date_source_data, date_train_target_data])
        date_train_label = pd.concat([date_source_label, date_train_target_label])
        date_train_label.index = date_train_data["cmaq_id"]
        date_source_dataset = self._date_allocate_data(date_source_data, date_train_data, self.source_sort_stations, date_train_label)
        date_train_target_dataset = self._date_allocate_data(date_train_target_data, date_train_data, self.train_target_sort_stations, date_train_label)
        date_valid_dataset = self._date_allocate_data(date_valid_data, date_train_data, self.valid_sort_stations, date_train_label)
        input_label_set = {
            "input": [date_source_dataset, date_train_target_dataset, date_valid_dataset],
            "label": [date_source_label, date_train_target_label, date_valid_label]
        }
        return input_label_set

    def _allocate_all_data(self):
        """
        Allocate the monitoring data-combined input dataset of source, train target, and validation data for all dates.
        """
        all_dates = np.unique(self.source_data["day"])
        all_source_data, all_train_target_data, all_valid_data = [], [], []
        all_source_label, all_train_target_label, all_valid_label = [], [], []
        for date_num, date in enumerate(all_dates):
            print(f"date {date_num}")
            date_input_label = self._compute_date_set(date)
            source_input, train_target_input, valid_input = date_input_label["input"]
            source_label, train_target_label, valid_label = date_input_label["label"]
            all_source_data.append(source_input)
            all_train_target_data.append(train_target_input)
            all_valid_data.append(valid_input)
            all_source_label.append(source_label)
            all_train_target_label.append(train_target_label)
            all_valid_label.append(valid_label)
        all_source_data = np.vstack(all_source_data)
        all_train_target_data = np.vstack(all_train_target_data)
        all_valid_data = np.vstack(all_valid_data)
        all_source_label = np.hstack(all_source_label)
        all_train_target_label = np.hstack(all_train_target_label)
        all_valid_label = np.hstack(all_valid_label)
        self.composed_inputs = (all_source_data, all_train_target_data, all_valid_data)
        self.composed_labels = (all_source_label, all_train_target_label, all_valid_label)

class PredStationAllocate:
    def __init__(self, source_data: pd.DataFrame, train_target_data: pd.DataFrame, target_coords: pd.DataFrame, 
                 source_label: pd.DataFrame, train_target_label: pd.DataFrame, station_num: int):
        """
        Allocate nearest monitoring station dataset.

        source_data: source input data
        train_target_data: train target input data
        valid_data: validation input data
        source_label: source label data
        train_target_label: train target label
        valid_label: validation label
        station_num: number of allocating nearest monitoring data
        """
        self.source_data = source_data
        self.train_target_data = train_target_data
        self.target_coords = target_coords
        self.source_label = source_label
        self.train_target_label = train_target_label
        self.station_num = station_num
        self.cmaq_cols = ["cmaq_x", "cmaq_y", "cmaq_id"]
        self._compute_distances()
        self.source_sort_stations = _sort_distance_stations(self.source_distance)
        self.train_target_sort_stations = _sort_distance_stations(self.train_target_distance)
        self.target_sort_stations = _sort_distance_stations(self.target_distance)

    def _compute_distances(self):
        """
        Compute distance matrix of source, train target, and target data.
        """
        source_cmaq = pd.DataFrame(np.unique(self.source_data[self.cmaq_cols], axis=0), columns=self.cmaq_cols).set_index("cmaq_id")
        train_target_cmaq = pd.DataFrame(np.unique(self.train_target_data[self.cmaq_cols], axis=0), columns=self.cmaq_cols).set_index("cmaq_id")
        train_cmaq = pd.concat([source_cmaq, train_target_cmaq])
        self.source_distance = create_distance_matrix(source_cmaq, train_cmaq)
        self.train_target_distance = create_distance_matrix(train_target_cmaq, train_cmaq)
        self.target_distance = create_distance_matrix(self.target_coords, train_cmaq)

    def _date_allocate_data(self, data: pd.DataFrame, train_data: pd.DataFrame, sort_stations: pd.DataFrame, train_label: pd.DataFrame):
        """
        Allocate nearest monitoring station data of given data for one date.
        First, Define the PM2.5 of given data as 0 and train monitoring statoin data as true PM2.5.
        Second, allocate the existing station cmaq_id's on the date.
        Finally, allocate the train data of corresponding station cmaq_id's.

        data: input data for allocating
        train_data: train input data
        sort_stations: sorted stations of given data
        train_label: train label data
        """
        cmaq_id_data = data.set_index("cmaq_id")
        cmaq_id_data["PM25"] = 0
        cmaq_id_train = train_data.set_index("cmaq_id")
        cmaq_id_train["PM25"] = train_label

        date_exist_stations = []
        for id in cmaq_id_data.index:
            row_station = sort_stations.loc[id]
            row_exist_stations = row_station[np.isin(row_station, cmaq_id_train.index)].reset_index(drop=True)
            date_exist_stations.append(row_exist_stations)
        date_exist_stations = pd.concat(date_exist_stations, axis=1).T

        station_data = []
        for s in range(self.station_num):
            near_data = cmaq_id_train.loc[date_exist_stations[s]]
            station_data.append(near_data)
        station_data.insert(len(station_data)//2, cmaq_id_data)
        stack_station_data = np.stack(station_data, -1)
        return stack_station_data
        
    def compute_date_set(self, date_target_data: pd.DataFrame):
        """
        Compute (allocate) the nearest monitoring data-combined input dataset of source, train target, and validation dataset for given date.

        date: date to allocate
        """
        date = date_target_data["day"].iloc[0]
        date_source_data = self.source_data.loc[self.source_data["day"]==date].copy()
        date_train_target_data = self.train_target_data.loc[self.train_target_data["day"]==date].copy()
        date_source_label = self.source_label.loc[date_source_data.index]
        date_train_target_label = self.train_target_label.loc[date_train_target_data.index]
        date_train_data = pd.concat([date_source_data, date_train_target_data])
        date_train_label = pd.concat([date_source_label, date_train_target_label])
        date_train_label.index = date_train_data["cmaq_id"]
        date_target_dataset = self._date_allocate_data(date_target_data, date_train_data, self.target_sort_stations, date_train_label)
        return date_target_dataset

class InputOutputSet(Dataset):
    def __init__(self, input_dt, output_dt):
        super().__init__()
        self.input_dt = input_dt
        self.output_dt = output_dt

    def __getitem__(self, i):
        return self.input_dt[i], self.output_dt[i]

    def __len__(self):
        return len(self.input_dt)

class DataCompose:
    def __init__(self, input_dt: pd.DataFrame, label_dt: pd.Series, train_valid_data_id: dict, source_type:str, normalize=False, pm_normalize=False, weight_average_compute=False, 
                 nearest_data_merge=False, coord_pm=False, save_merged_data=False, data_path=""):
        """
        Compose the dataset for train-test.

        input_dt: whole input dataset
        label_dt: whole label dataset
        train_valid_data_id: train-validation data id set
        normalize: normalize input data or not
        weight_average_compute: weighted average compute or not
        nearest_data_merge: nearest monitoring data compose or not
        save_merged_data: save the merged nearest data or not
        data_path: path to save merged data
        """
        self.input_dt = input_dt
        self.label_dt = label_dt
        self.train_valid_data_id = train_valid_data_id
        self.source_type = source_type
        self.pm_normalize = pm_normalize
        self.weight_average_compute = weight_average_compute
        self.nearest_data_merge = nearest_data_merge
        self.coord_pm = coord_pm
        self.data_path = data_path
        self.save_merged_data = save_merged_data
        self._split_train_valid_cmaq()
        if nearest_data_merge:
            self._allocate_near_data()
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
        self.input_shape = {}
        for split_id in self.train_valid_data_id.keys():
            print(f"cluster{split_id} composing...")
            set_index = self.train_valid_data_id[split_id]
            source_index, train_target_index, valid_index = set_index['train_out_cluster'], set_index['train_in_cluster'], set_index['test_cluster']
            source_input, source_label = self.input_dt.loc[np.isin(self.input_dt["cmaq_id"], source_index)], self.label_dt[np.isin(self.input_dt["cmaq_id"], source_index)]
            train_target_input, train_target_label = self.input_dt.loc[np.isin(self.input_dt["cmaq_id"], train_target_index)], self.label_dt[np.isin(self.input_dt["cmaq_id"], train_target_index)]
            valid_input, valid_label = self.input_dt.loc[np.isin(self.input_dt["cmaq_id"], valid_index)], self.label_dt[np.isin(self.input_dt["cmaq_id"], valid_index)]
            source_input, train_target_input, valid_input = _drop_useless_col(source_input, train_target_input, valid_input)
            if self.source_type == "east half":
                east_index = (source_input["cmaq_x"] > 0)
                source_input, source_label = source_input[east_index], source_label[east_index]
            if self.source_type == "west half":
                west_index = (source_input["cmaq_x"] < 0)
                source_input, source_label = source_input[west_index], source_label[west_index]
            if self.source_type == "east3":
                source_xy = pd.read_csv("data/east3 xy.csv")[["cmaq_x", "cmaq_y"]]
                east_index = np.isin(source_input[["cmaq_x", "cmaq_y"]], source_xy).all(axis=1)
                source_input, source_label = source_input[east_index], source_label[east_index]
            self.source_dt[split_id] = {"input":source_input, "label": source_label}
            self.train_target_dt[split_id] = {"input":train_target_input, "label": train_target_label}
            self.valid_dt[split_id] = {"input":valid_input, "label":valid_label}
            self.input_shape[split_id] = source_input.shape[1]

    def _weight_average(self):
        """
        Compute the weighted average for train and validation data. (Need update)
        """
        for split_id in self.train_valid_data_id.keys():
            train_dt = _unite_train_data(self.source_dt, self.train_target_dt)
            train_input = train_dt[split_id]["input"]
            train_label = train_dt[split_id]["label"]
            source_input = self.source_dt[split_id]["input"]
            train_target_input = self.train_target_dt[split_id]["input"]
            valid_input = self.valid_dt[split_id]["input"]
            weight_average = WeightAverage(source_input, train_target_input, valid_input, train_input, train_label)
            source_input, train_target_input, valid_input = weight_average.weight_inputs
            self.source_dt[split_id]["input"] = source_input
            self.train_target_dt[split_id]["input"] = train_target_input
            self.valid_dt[split_id]["input"] = valid_input
        
    def _normalize_train_valid(self):
        """
        Normalize the train (source and train target) and validation data.
        If standard deviation is 0, substitute to 1 to make the values 0 with just subtracting by mean.
        """
        for split_id in self.train_valid_data_id.keys():
            source_input = self.source_dt[split_id]["input"]
            train_target_input = self.train_target_dt[split_id]["input"]
            valid_input = self.valid_dt[split_id]["input"]
            train_input = np.vstack([source_input, train_target_input])
            mean, std = train_input.mean(axis=0), train_input.std(axis=0)
            std[std==0] = 1
            if not self.pm_normalize:
                if self.nearest_data_merge:
                    mean[-1,:] = 0
                    std[-1,:] = 1
                elif self.weight_average_compute:
                    mean[-1] = 0
                    std[-1] = 1
            self.source_dt[split_id]["input"] = (source_input - mean) / std
            self.train_target_dt[split_id]["input"] = (train_target_input - mean) / std
            self.valid_dt[split_id]["input"] = (valid_input - mean) / std
            self.mean, self.std = mean, std

    def _allocate_near_data(self):
        """
        Allocate the nearest monitoring station-combined input data.
        If first merging (self.save_merged_data=True), compose the dataset and save.
        Else, load the composed dataset.
        """
        self.input_shape = {}
        for split_id in self.train_valid_data_id.keys():
            print(f"cluster{split_id} composing...")
            if self.save_merged_data:
                source_input = self.source_dt[split_id]["input"]
                train_target_input = self.train_target_dt[split_id]["input"]
                valid_input = self.valid_dt[split_id]["input"]
                source_label = self.source_dt[split_id]["label"]
                train_target_label = self.train_target_dt[split_id]["label"]
                valid_label = self.valid_dt[split_id]["label"]
                station_allocate = StationAllocate(source_input, train_target_input, valid_input, source_label, train_target_label, valid_label, 12)
                source_data, train_target_data, valid_data = station_allocate.composed_inputs
                source_label, train_target_label, valid_label = station_allocate.composed_labels
                # np.savez(f"{self.data_path}split-{split_id}/nearest_dataset2.npz", source_input=source_data, train_target_input=train_target_data, valid_input=valid_data,
                #          source_label=source_label, train_target_label=train_target_label, valid_label=valid_label)
            else:
                save_npz = np.load(f"{self.data_path}split-{split_id}/nearest_dataset.npz")
                source_data, train_target_data, valid_data = save_npz["source_input"], save_npz["train_target_input"], save_npz["valid_input"]
                source_label, train_target_label, valid_label = save_npz["source_label"], save_npz["train_target_label"], save_npz["valid_label"]
                if self.source_type == "east half":
                    east_index = (source_data[:,2, 6] > 0)
                    source_data, source_label = source_data[east_index], source_label[east_index]
                if self.source_type == "west half":
                    west_index = (source_data[:,2, 6] < 0)
                    source_data, source_label = source_data[west_index], source_label[west_index]
                if self.source_type == "east3":
                    source_xy = pd.read_csv("data/east3 xy.csv")[["cmaq_x", "cmaq_y"]]
                    east_index = np.isin(source_data[:,[2,3], 6], source_xy).all(axis=1)
                    source_data, source_label = source_data[east_index], source_label[east_index]
                if self.coord_pm:
                    source_data, train_target_data, valid_data = source_data[:,[2,3,-1]], train_target_data[:,[2,3,-1]], valid_data[:,[2,3,-1]]
            self.source_dt[split_id]["input"] = source_data
            self.train_target_dt[split_id]["input"] = train_target_data
            self.valid_dt[split_id]["input"] = valid_data
            self.source_dt[split_id]["label"] = source_label
            self.train_target_dt[split_id]["label"] = train_target_label
            self.valid_dt[split_id]["label"] = valid_label
            self.input_shape[split_id] = source_data.shape[1:]
        self.train_dt = _unite_train_data(self.source_dt, self.train_target_dt)

    def regression_data_convert_loader(self, train_unite: bool):
        """
        Convert the regression data to DataLoader. If unite the train dataset, unite the source and target train data as train data.

        train_unite: unite the train data or not
        """
        if train_unite:
            train_dt, valid_dt = {}, {}
        else:
            source_dt, train_target_dt, valid_dt = {}, {}, {}
        for split_id in self.train_valid_data_id.keys():
            source_input = np.array(self.source_dt[split_id]["input"])
            source_label = np.array(self.source_dt[split_id]["label"])
            train_target_input = np.array(self.train_target_dt[split_id]["input"])
            train_target_label = np.array(self.train_target_dt[split_id]["label"])
            valid_input = np.array(self.valid_dt[split_id]["input"])
            valid_label = np.array(self.valid_dt[split_id]["label"])
            if train_unite:
                train_input = np.vstack([source_input, train_target_input])
                train_label = np.hstack([source_label, train_target_label])
                train_loader = _convert_loader(train_input, train_label, 128)
                train_dt[split_id] = train_loader
            else:
                source_loader = _convert_loader(source_input, source_label, 128)
                train_target_loader = _convert_loader(train_target_input, train_target_label, 128)
                source_dt[split_id] = source_loader
                train_target_dt[split_id] = train_target_loader
            valid_loader = _convert_loader(valid_input, valid_label, 128)
            valid_dt[split_id] = valid_loader
        if train_unite:
            return train_dt, valid_dt
        else:
            return source_dt, train_target_dt, valid_dt

    def autoencode_data_convert_loader(self, train_unite: bool):
        """
        Convert the autoencoder data to DataLoader. If unite the train dataset, unite the source and target train data as train data.

        train_unite: unite the train data or not
        """
        if train_unite:
            train_dt, valid_dt = {}, {}
        else:
            source_dt, train_target_dt, valid_dt = {}, {}, {}
        for split_id in self.train_valid_data_id.keys():
            source_input = np.array(self.source_dt[split_id]["input"])
            train_target_input = np.array(self.train_target_dt[split_id]["input"])
            valid_input = np.array(self.valid_dt[split_id]["input"])
            if train_unite:
                train_input = np.vstack([source_input, train_target_input])
                train_dt[split_id] = DataLoader(train_input, batch_size=128, shuffle=False, pin_memory=True)
            else:
                source_dt[split_id] = DataLoader(source_input, batch_size=128, shuffle=False, pin_memory=True)
                train_target_dt[split_id] = DataLoader(train_target_input, batch_size=128, shuffle=False, pin_memory=True)
            valid_dt[split_id] = DataLoader(valid_input, batch_size=128, shuffle=False, pin_memory=True)
        if train_unite:
            return train_dt, valid_dt
        else:
            return source_dt, train_target_dt, valid_dt

class PredCompose:
    def __init__(self, input_dt: pd.DataFrame, label_dt: pd.Series, target_coords: pd.DataFrame, train_valid_data_id: dict, source_type:str, normalize: bool, statistics: dict):
        """
        Compose the dataset for train-test.

        input_dt: whole input dataset
        label_dt: whole label dataset
        train_valid_data_id: train-validation data id set
        normalize: normalize input data or not
        weight_average_compute: weighted average compute or not
        nearest_data_merge: nearest monitoring data compose or not
        save_merged_data: save the merged nearest data or not
        data_path: path to save merged data
        """
        self.input_dt = input_dt
        self.label_dt = label_dt
        self.target_coords = target_coords
        self.train_valid_data_id = train_valid_data_id
        self.source_type = source_type
        self.normalize = normalize
        self.mean, self.std = statistics["mean"], statistics["std"]
        self._split_train_valid_cmaq()

    def _split_train_valid_cmaq(self):
        """
        Split the train data and validation data using the cmaq id's (self.train_valid_data_id).
        There are three types of data, source (train_out), train target (train_in), and validation (test).
        """
        self.input_shape = {}
        source_index, train_target_index = self.train_valid_data_id['train_out_cluster'], self.train_valid_data_id['train_in_cluster']
        source_input, source_label = self.input_dt.loc[np.isin(self.input_dt["cmaq_id"], source_index)], self.label_dt[np.isin(self.input_dt["cmaq_id"], source_index)]
        train_target_input, train_target_label = self.input_dt.loc[np.isin(self.input_dt["cmaq_id"], train_target_index)], self.label_dt[np.isin(self.input_dt["cmaq_id"], train_target_index)]
        source_input, train_target_input, _ = _drop_useless_col(source_input, train_target_input, train_target_input)
        if self.source_type == "east half":
            east_index = (source_input["cmaq_x"] > 0)
            source_input, source_label = source_input[east_index], source_label[east_index]
        if self.source_type == "east3":
            source_xy = pd.read_csv("data/east3 xy.csv")[["cmaq_x", "cmaq_y"]]
            east_index = np.isin(source_input[["cmaq_x", "cmaq_y"]], source_xy).all(axis=1)
            source_input, source_label = source_input[east_index], source_label[east_index]
        self.source_dt = {"input":source_input, "label": source_label}
        self.train_target_dt = {"input":train_target_input, "label": train_target_label}
        self.input_shape = source_input.shape[1]

    def normalize_input(self, target_input):
        variable_cols = self.source_dt["input"].columns
        return (target_input[variable_cols] - self.mean) / self.std

    def weight_average_data(self, target_input):
        """
        Compute the weighted average for train and validation data. (Need update)
        """
        train_input = pd.concat([self.source_dt["input"], self.train_target_dt["input"]])
        train_label = pd.concat([self.source_dt["label"], self.train_target_dt["label"]])
        weight_average = PredWeightAverage(self.target_coords, train_input, train_label)
        weight_target_input = weight_average.compute_date_wa(target_input[train_input.columns])
        if self.normalize:
            weight_target_input = (weight_target_input - self.mean) / self.std
        return weight_target_input
        
    def allocate_near_data(self, target_input: pd.DataFrame, coord_pm: bool, compose_save: bool, data_path: str):
        """
        Allocate the nearest monitoring station-combined input data.
        If first merging (self.save_merged_data=True), compose the dataset and save.
        Else, load the composed dataset.
        """
        self.input_shape = {}
        if compose_save:
            source_input = self.source_dt["input"]
            train_target_input = self.train_target_dt["input"]
            source_label = self.source_dt["label"]
            train_target_label = self.train_target_dt["label"]
            station_allocate = PredStationAllocate(source_input, train_target_input, self.target_coords, source_label, train_target_label, 12)
            target_data = station_allocate.compute_date_set(target_input)
            np.savez(f"{data_path}target_pred_input.npz", target_input=target_data)
        else:
            save_npz = np.load(f"{data_path}target_pred_input.npz")
            target_data = save_npz["target_input"]
        if coord_pm:
            target_data = target_data[:,[2,3,-1]]
        if self.normalize:
            target_data = (target_data - self.mean) / self.std
        return target_data

    def autoencode_data_convert_loader(self, target_data):
        """
        Convert the autoencoder data to DataLoader. If unite the train dataset, unite the source and target train data as train data.

        train_unite: unite the train data or not
        """
        target_loader  = _convert_loader(target_data, target_data, 32)
        return target_loader
