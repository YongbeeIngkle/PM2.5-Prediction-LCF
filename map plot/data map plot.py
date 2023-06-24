import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_process.data import tag_names
from data_process.spatial_validation import get_clusters, ClusterGrid, TargetGrid

def _get_input_label(input_dt: pd.DataFrame, label_dt: pd.DataFrame, train_test_data_id: dict):
    all_inputs, all_labels = [], []
    for cluster_id in train_test_data_id.keys():
        cluster_test_index = train_test_data_id[cluster_id]['test_cluster']
        all_labels.append(label_dt[cluster_test_index])
        all_inputs.append(input_dt.loc[cluster_test_index])
    all_labels = pd.concat(all_labels)
    all_inputs = pd.concat(all_inputs)
    return all_inputs, all_labels

def _source_target_split(coord_cluster, cluster_id):
    target_coords = coord_cluster.loc[coord_cluster["cluster id"] == cluster_id]
    source_coords = coord_cluster.loc[coord_cluster["cluster id"] != cluster_id]
    return {"source": source_coords, "target": target_coords}

class SingleAnalyzerClusterModel:
    def __init__(self, coor_cluster, whole_coord):
        self.target_grid = TargetGrid()
        self.coor_cluster = coor_cluster
        self.whole_coord = whole_coord
        self.target_cluster = 4
        self.east_cluster = [1, 2, 6]
        self.whole_clsuter = cluster_model.predict(whole_coord)

    def _split_data_coord(self):
        source_target_set = _source_target_split(self.coor_cluster, self.target_cluster)
        train_test_coord = self.target_grid.split_data_coord(source_target_set["target"], sp)
        return source_target_set["source"], train_test_coord

    def plot_whole_cluster(self, save, alpha):
        plt.figure(figsize=(13,8))
        for cluster_id in np.sort(np.unique(self.whole_clsuter)):
            cluster_coords = self.whole_coord.iloc[self.whole_clsuter==cluster_id]
            plt.scatter(cluster_coords['cmaq_x'], cluster_coords['cmaq_y'], s=3, alpha=alpha)
        if save:
            plt.show()
    
    def plot_target(self, train_num, source_type="whole"):
        marker_size = 17
        save_dir = f"figures/target cluster split/tl-cal-{train_num}/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        source_coord, target_coord_set = self._split_data_coord()
        if source_type == "east half":
            save_dir = f"{save_dir}east half "
            east_coord = source_coord[source_coord["cmaq_x"] > 0]
            west_coord = source_coord[source_coord["cmaq_x"] < 0]
        if source_type == "east3":
            save_dir = f"{save_dir}east3 "
            east_coord = source_coord[np.isin(source_coord["cluster id"], self.east_cluster)]
            east_coord.to_csv("data/east3 xy.csv", index=False)
        for set_id in target_coord_set.keys():
            set_data = target_coord_set[set_id]
            train_coord = set_data["train"]
            test_coord = set_data["test"]
            fig_savedir = f"{save_dir}split-{set_id}.png"
            self.plot_whole_cluster(False, 0.1)
            plt.scatter(train_coord["cmaq_x"], train_coord["cmaq_y"], s=marker_size, color="green", alpha=0.6, marker='o', label="train-target")
            plt.scatter(test_coord["cmaq_x"], test_coord["cmaq_y"], s=marker_size, color='red', alpha=0.6, marker='^', label="test")
            if "east" in source_type:
                plt.scatter(west_coord["cmaq_x"], west_coord["cmaq_y"], s=marker_size, color='blue', alpha=1, marker="+", label="source")
                plt.scatter(east_coord["cmaq_x"], east_coord["cmaq_y"], s=marker_size, color='purple', alpha=1, marker="x", label="east source")
            if source_type == "east half":
                plt.axvline(0, linestyle='-', color='r', alpha=0.4)
            plt.legend(bbox_to_anchor=(0.7, 1), loc="upper left", prop={'size': 12, 'weight':'bold'})
            plt.axis('off')
            plt.savefig(fig_savedir)
            plt.close()

if __name__=='__main__':
    sp = 15
    monitoring_whole_data = pd.read_csv("data/us_monitoring.csv")[tag_names]
    coord_whole_data = pd.read_csv("data/largeUS_coords_pred.csv", index_col=0)
    whole_coord = coord_whole_data.drop_duplicates().reset_index(drop=True)[['cmaq_x', 'cmaq_y']]

    input_dt = monitoring_whole_data.drop(columns=["pm25_value"])
    label_dt = monitoring_whole_data["pm25_value"]
    train_test_data_id, cluster_model = get_clusters(input_dt, label_dt)
    all_input, all_label = _get_input_label(input_dt, label_dt, train_test_data_id)

    single_grid = ClusterGrid("KMeans")
    _, coor_cluster = single_grid.cluster_grids(input_dt, label_dt)
    
    single_analyzer = SingleAnalyzerClusterModel(coor_cluster, whole_coord)
    # single_analyzer.plot_whole_cluster(True, 0.1)
    single_analyzer.plot_target(sp, "east half")
