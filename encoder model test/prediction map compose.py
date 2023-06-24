import os
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix

tag_names = ['day', 'month', 'cmaq_x', 'cmaq_y', 'cmaq_id', 'rid', 'elev', 'forest_cover', 'pd', 'local', 'limi', 'high', 'is', 
    'nldas_pevapsfc','nldas_pressfc', 'nldas_cape', 'nldas_ugrd10m', 'nldas_vgrd10m', 'nldas_tmp2m', 'nldas_rh2m', 'nldas_dlwrfsfc', 
    'nldas_dswrfsfc', 'nldas_pcpsfc', 'nldas_fpcsfc', 'gc_aod', 'aod_value', 'emissi11_pm25', 'pm25_value']

def get_in_clusters(data_path: str, train_num: int):
    target_path = f"{data_path}tl-cal-{train_num}/"
    train_test_id = {}

    source_cmaq = np.load(f"{data_path}source_cmaq.npy")
    for set_id in range(10):
        target_cmaq = np.load(f"{target_path}split-{set_id}/target_cmaq.npz")
        train_cmaq = target_cmaq["train"]
        test_cmaq = target_cmaq["test"]
        train_test_id[set_id] = {
            "train_in_cluster":train_cmaq,
            "train_out_cluster":source_cmaq,
            "test_cluster":test_cmaq
        }
    return train_test_id

def cluster_train_valid_index(set_index: dict):
    train_target_index = list(set_index['train_in_cluster'])
    source_index = list(set_index['train_out_cluster'])
    valid_index = list(set_index['test_cluster'])
    return source_index, train_target_index, valid_index

def _drop_constant_col(train_dt: pd.DataFrame,):
    _std = train_dt.std(axis=0)
    train_dt_variable = train_dt.loc[:,_std>0]
    return train_dt_variable.columns

def _drop_na_col(train_dt: pd.DataFrame):
    train_drop_dt = train_dt.dropna(axis=1)
    return train_drop_dt.columns

def _drop_useless_col(source_data, train_target_data, valid_data):
    train_data = pd.concat([source_data, train_target_data])
    train_cols = _drop_na_col(train_data)
    train_cols = _drop_constant_col(train_data[train_cols])
    source_drop_const = source_data[train_cols]
    train_drop_const = train_target_data[train_cols]
    valid_drop_const = valid_data[train_cols]
    return source_drop_const, train_drop_const, valid_drop_const

def _sort_distance_stations(distance_data: pd.DataFrame):
        nearest_stations = pd.DataFrame(columns=range(distance_data.shape[1]), index=distance_data.index)
        for row in distance_data.index:
            nearest_stations.loc[row] = distance_data.columns[distance_data.loc[row].argsort()]
        return nearest_stations

def create_distance_matrix(dt1: pd.DataFrame, dt2: pd.DataFrame):
    all_distance = distance_matrix(dt1, dt2)
    all_distance[all_distance==0] = np.inf
    all_distance = pd.DataFrame(all_distance, index=dt1.index, columns=dt2.index)
    return all_distance

class StationAllocate:
    def __init__(self, source_data, train_target_data, valid_data, source_label, train_target_label, station_num, save_path):
        self.source_data = source_data
        self.train_target_data = train_target_data
        self.valid_data = valid_data
        self.source_label = source_label
        self.train_target_label = train_target_label
        self.station_num = station_num
        self.save_path = save_path
        self.cmaq_cols = ["cmaq_x", "cmaq_y", "cmaq_id"]
        self._compute_distances()
        self.source_sort_stations = _sort_distance_stations(self.source_distance)
        self.train_target_sort_stations = _sort_distance_stations(self.train_target_distance)
        self.valid_sort_stations = _sort_distance_stations(self.valid_distance)
        self._allocate_all_data()

    def _compute_distances(self):
        source_cmaq = pd.DataFrame(np.unique(self.source_data[self.cmaq_cols], axis=0), columns=self.cmaq_cols).set_index("cmaq_id")
        train_target_cmaq = pd.DataFrame(np.unique(self.train_target_data[self.cmaq_cols], axis=0), columns=self.cmaq_cols).set_index("cmaq_id")
        train_cmaq = pd.concat([source_cmaq, train_target_cmaq])
        valid_cmaq = pd.DataFrame(np.unique(self.valid_data[self.cmaq_cols], axis=0), columns=self.cmaq_cols).set_index("cmaq_id")
        self.source_distance = create_distance_matrix(source_cmaq, train_cmaq)
        self.train_target_distance = create_distance_matrix(train_target_cmaq, train_cmaq)
        self.valid_distance = create_distance_matrix(valid_cmaq, train_cmaq)

    def _date_allocate_data(self, data: pd.DataFrame, train_data: pd.DataFrame, sort_stations: pd.DataFrame, train_label):
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
        
    def _compute_date_set(self, date):
        date_source_data = self.source_data.loc[self.source_data["day"]==date].copy()
        date_train_target_data = self.train_target_data.loc[self.train_target_data["day"]==date].copy()
        date_valid_data = self.valid_data.loc[self.valid_data["day"]==date].copy()
        date_source_label = self.source_label.loc[date_source_data.index]
        date_train_target_label = self.train_target_label.loc[date_train_target_data.index]
        date_train_data = pd.concat([date_source_data, date_train_target_data])
        date_train_label = pd.concat([date_source_label, date_train_target_label])
        date_train_label.index = date_train_data["cmaq_id"]
        date_source_dataset = self._date_allocate_data(date_source_data, date_train_data, self.source_sort_stations, date_train_label)
        date_train_target_dataset = self._date_allocate_data(date_train_target_data, date_train_data, self.train_target_sort_stations, date_train_label)
        date_valid_dataset = self._date_allocate_data(date_valid_data, date_train_data, self.valid_sort_stations, date_train_label)
        input_label_set = {
            "input": [date_source_dataset, date_train_target_dataset, date_valid_dataset],
            "label": [date_source_label, date_train_target_label]
        }
        np.savez(f"{self.save_path}date{date}.npz", dataset=date_valid_dataset)
        return input_label_set

    def _allocate_all_data(self):
        all_dates = np.unique(self.valid_data["day"])
        all_source_data, all_train_target_data, all_valid_data = [], [], []
        all_source_label, all_train_target_label = [], []
        for date in all_dates:
            print(f"date {date}")
            date_input_label = self._compute_date_set(date)
            # source_input, train_target_input, valid_input = date_input_label["input"]
            # source_label, train_target_label = date_input_label["label"]
        #     all_source_data.append(source_input)
        #     all_train_target_data.append(train_target_input)
        #     all_valid_data.append(valid_input)
        #     all_source_label.append(source_label)
        #     all_train_target_label.append(train_target_label)
        # all_source_data = np.vstack(all_source_data)
        # all_train_target_data = np.vstack(all_train_target_data)
        # all_valid_data = np.vstack(all_valid_data)
        # all_source_label = np.hstack(all_source_label)
        # all_train_target_label = np.hstack(all_train_target_label)
        # self.composed_inputs = (all_source_data, all_train_target_data, all_valid_data)
        # self.composed_labels = (all_source_label, all_train_target_label)

class DataCompose:
    def __init__(self, input_dt: pd.DataFrame, label_dt: pd.Series, train_valid_data_id: dict, normalize=False, save_merged_data=False, data_path=""):
        self.input_dt = input_dt
        self.label_dt = label_dt
        self.train_valid_data_id = train_valid_data_id
        self.data_path = data_path
        self.save_merged_data = save_merged_data
        self._split_train_valid_cmaq()
        if normalize:
            self._normalize_train_valid()

    def _split_train_valid_cmaq(self):
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
            self.source_dt[split_id] = {"input":source_input, "label": source_label}
            self.train_target_dt[split_id] = {"input":train_target_input, "label": train_target_label}
            self.valid_dt[split_id] = {"input":valid_input, "label":valid_label}
            self.input_shape[split_id] = source_input.shape[1]
        
    def _normalize_train_valid(self):
        for split_id in self.train_valid_data_id.keys():
            source_input = self.source_dt[split_id]["input"]
            train_target_input = self.train_target_dt[split_id]["input"]
            valid_input = self.valid_dt[split_id]["input"]
            train_input = pd.concat([source_input, train_target_input])
            mean, std = train_input.mean(axis=0), train_input.std(axis=0)
            self.source_dt[split_id]["input"] = (source_input - mean) / std
            self.train_target_dt[split_id]["input"] = (train_target_input - mean) / std
            self.valid_dt[split_id]["input"] = (valid_input - mean) / std

    def allocate_near_data(self, target_data, save_path):
        self.input_shape = {}
        for split_id in self.train_valid_data_id.keys():
            print(f"cluster{split_id} composing...")
            source_input = self.source_dt[split_id]["input"]
            train_target_input = self.train_target_dt[split_id]["input"]
            valid_input = self.valid_dt[split_id]["input"]
            source_label = self.source_dt[split_id]["label"]
            train_target_label = self.train_target_dt[split_id]["label"]
            valid_label = self.valid_dt[split_id]["label"]
            if self.save_merged_data:
                station_allocate = StationAllocate(source_input, train_target_input, target_data[source_input.columns], source_label, train_target_label, 12, save_path)
                source_data, train_target_data, valid_data = station_allocate.composed_inputs
                source_label, train_target_label, valid_label = station_allocate.composed_labels
                # np.savez(f"{self.data_path}split-{split_id}/nearest_dataset.npz", source_input=source_data, train_target_input=train_target_data, valid_input=valid_data,
                #          source_label=source_label, train_target_label=train_target_label, valid_label=valid_label)

def _one_cluster_model(train_num, input_dt, label_dt, target_data):
    data_path = "data/split-data/"
    compose_path = f"D:/target-encode/tl-cal-{train_num}/"
    if not os.path.exists(compose_path):
        os.makedirs(compose_path)

    train_test_data_id = get_in_clusters(data_path, train_num)[0]
    data_compose = DataCompose(input_dt, label_dt, {0:train_test_data_id}, False, True, compose_path)
    data_compose.allocate_near_data(target_data, compose_path)
    
if __name__=='__main__':
    weight_average = False
    train_numbers = [15]

    monitoring_whole_data = pd.read_csv("data/us_monitoring.csv")[tag_names]
    input_dt = monitoring_whole_data.drop(columns=["pm25_value"])
    label_dt = monitoring_whole_data["pm25_value"]
    target_data = []
    for date in range(1,366):
        target_date_data = pd.read_csv(f"D:/split-by-day/us-2011-satellite-target4-day-{date}.csv")
        target_data.append(target_date_data)
    target_data = pd.concat(target_data)

    for train_num in train_numbers:
        _one_cluster_model(train_num, input_dt, label_dt, target_data)
