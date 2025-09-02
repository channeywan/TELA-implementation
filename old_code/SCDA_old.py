from typing import List, Any, Tuple
from collections import deque
from data.loader import DiskDataLoader
from visualization.plotter import DiskPlotter
from data.processor import DiskDataProcessor
from config.settings import WarehouseConfig, DataConfig, ModelConfig, DirConfig
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import logging
import os
import pickle
import random
import linecache
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)


class SCDA:
    def __init__(self):
        self.loader = DiskDataLoader()
        self.processor = DiskDataProcessor()
        self.plotter = DiskPlotter()

    def write_result(self, utilize_trace, violation_count_per_warehouse, max_violation_duration, all_violation_durations, average_dimension_load_imb, average_warehouse_load_imb):
        """写入结果"""
        DirConfig.ensure_dirs()
        result_dir = os.path.join(DirConfig.SCDA_DIR, f"SCDA_result.txt")
        with open(result_dir, "w") as f:
            f.write("----------------------Config------------------------\n")
            f.write(f"warehouse_number:{WarehouseConfig.WAREHOUSE_NUMBER}\n")
            f.write(f"episodes:{ModelConfig.EPISODES}\n")
            f.write(
                f"evaluate_time_number:{DataConfig.EVALUATE_TIME_NUMBER}\n")
            f.write(
                f"violation_queue_length:{DataConfig.VIOLATION_QUEUE_LENGTH}\n")
            f.write(
                f"min_violation_time_window:{DataConfig.MIN_VIOLATION_TIME_WINDOW}\n")
            f.write(f"cluster_index_list:{DataConfig.CLUSTER_INDEX_LIST}\n")
            f.write(
                f"reservation_rate_for_monitor:{ModelConfig.RESERVATION_RATE_FOR_MONITOR}\n")
            f.write(f"episode_remain:{ModelConfig.EPISODE_REMAIN}\n")
            f.write("---------------------------------------------------\n")
            f.write("----------------------utilize_trace------------------------\n")
            for timestamp in range(len(utilize_trace)):
                f.write(",".join(map(str, utilize_trace[timestamp]))+"\n")
            f.write("---------------------------------------------------\n")
            f.write(
                "----------------------violation_count_per_warehouse------------------------\n")
            f.write(",".join(map(str, violation_count_per_warehouse))+"\n")
            f.write("---------------------------------------------------\n")
            f.write(
                "----------------------max_violation_duration------------------------\n")
            f.write(",".join(map(str, max_violation_duration))+"\n")
            f.write("---------------------------------------------------\n")
            f.write(
                "----------------------all_violation_durations------------------------\n")
            for episode in range(len(all_violation_durations)):
                for warehouse in range(WarehouseConfig.WAREHOUSE_NUMBER):
                    f.write(f"episode{episode+1},warehouse{warehouse},"+",".join(
                        map(str, all_violation_durations[episode][warehouse]))+"\n")
            f.write("---------------------------------------------------\n")
            f.write(
                "----------------------average_dimension_load_imb------------------------\n")
            f.write(",".join(map(str, average_dimension_load_imb))+"\n")
            f.write("---------------------------------------------------\n")
            f.write(
                "----------------------average_warehouse_load_imb------------------------\n")
            f.write(",".join(map(str, average_warehouse_load_imb))+"\n")
            f.write("---------------------------------------------------\n")
            logger.info(f"结果已写入 {result_dir}")

    def run(self) -> None:
        utilize_trace_episodes = np.zeros((DataConfig.EVALUATE_TIME_NUMBER, 3))
        violation_count_per_warehouse_episodes = np.zeros(
            WarehouseConfig.WAREHOUSE_NUMBER)
        max_violation_duration_episodes = np.zeros(
            WarehouseConfig.WAREHOUSE_NUMBER)
        episode_all_violation_durations = []
        average_dimension_load_imb_episodes = np.zeros(
            WarehouseConfig.WAREHOUSE_NUMBER)
        average_warehouse_load_imb_episodes = np.zeros(3)
        for episode in range(ModelConfig.EPISODES):
            logger.info(f"Episode {episode+1}/{ModelConfig.EPISODES}")
            items = self.loader.load_items(
                type="both", cluster_index_list=DataConfig.CLUSTER_INDEX_LIST_PREDICT, purpose="train")
            items_predict = self.predict(items)
            warehouses_trace = self.place_item(items_predict)
            utilize_trace, violation_count_per_warehouse, max_violation_duration, all_violation_durations, average_dimension_load_imb, average_warehouse_load_imb = self.evaluate_warehouses(
                warehouses_trace)
            utilize_trace = np.array(utilize_trace)
            violation_count_per_warehouse = np.array(
                violation_count_per_warehouse)
            max_violation_duration = np.array(max_violation_duration)
            average_dimension_load_imb = np.array(average_dimension_load_imb)
            average_warehouse_load_imb = np.array(average_warehouse_load_imb)
            utilize_trace_episodes += utilize_trace
            violation_count_per_warehouse_episodes += violation_count_per_warehouse
            max_violation_duration_episodes += max_violation_duration
            episode_all_violation_durations.append(all_violation_durations)
            average_dimension_load_imb_episodes += average_dimension_load_imb
            average_warehouse_load_imb_episodes += average_warehouse_load_imb
        utilize_trace_episodes = utilize_trace_episodes/ModelConfig.EPISODES
        violation_count_per_warehouse_episodes = violation_count_per_warehouse_episodes / \
            ModelConfig.EPISODES
        max_violation_duration_episodes = max_violation_duration_episodes/ModelConfig.EPISODES
        average_dimension_load_imb_episodes = average_dimension_load_imb_episodes / \
            ModelConfig.EPISODES
        average_warehouse_load_imb_episodes = average_warehouse_load_imb_episodes / \
            ModelConfig.EPISODES
        self.write_result(utilize_trace_episodes, violation_count_per_warehouse_episodes, max_violation_duration_episodes,
                          episode_all_violation_durations, average_dimension_load_imb_episodes, average_warehouse_load_imb_episodes)

    def train_model(self):
        items = self.loader.load_items(
            type="both", cluster_index_list=DataConfig.CLUSTER_INDEX_LIST_TRAIN, purpose="train")
        items = pd.DataFrame(items, columns=["disk_ID", "disk_capacity", "disk_if_local", "disk_attr", "disk_type", "disk_if_VIP",
                             "disk_pay", "vm_cpu", "vm_mem", "avg_rbw", "avg_wbw", "peak_rbw", "peak_wbw", "timestamp_num", "burst_label", "cluster_index"])
        model_cluster, labels_cluster = self.train_Kmeans(
            items)
        model_classify = self.train_DecisionTree(items, labels_cluster)
        model_cluster_dir = os.path.join(
            DirConfig.MODEL_DIR, "model_cluster.pkl")
        model_classify_dir = os.path.join(
            DirConfig.MODEL_DIR, "model_classify.pkl")
        with open(model_cluster_dir, "wb") as f:
            pickle.dump(model_cluster, f)
        with open(model_classify_dir, "wb") as f:
            pickle.dump(model_classify, f)

    def train_Kmeans(self, items: pd.DataFrame):
        """
        训练Kmeans模型
        (average read BW, average write BW, peak read BW, peak write BW)->label
        input:
            items: 物品列表
        output:
            model_cluster: Kmeans模型
            labels_cluster: 物品的聚类标签
        """
        train_data = items[["avg_rbw", "avg_wbw", "peak_rbw", "peak_wbw"]]
        cluster_K = ModelConfig.CLUSTER_K
        scaler = MinMaxScaler()
        train_data_scaled = scaler.fit_transform(train_data)
        model_cluster = KMeans(n_clusters=cluster_K)
        model_cluster.fit(train_data_scaled)
        cluster_centers = model_cluster.cluster_centers_
        labels_cluster = model_cluster.labels_
        unique_labels, counts = np.unique(labels_cluster, return_counts=True)
        the_silhouette_score = silhouette_score(
            train_data_scaled, labels_cluster)
        logger.info(f"Silhouette score: {the_silhouette_score}\n")
        return model_cluster, labels_cluster

    def train_DecisionTree(self, items: pd.DataFrame, labels_cluster: List[int]):
        """
        训练决策树模型
        (disk_capacity, disk_if_local, disk_type, vm_cpu, vm_mem)->label
        input:
            items: 物品列表
            labels_cluster: 物品的聚类标签
        output:
            best_clf: 最优的决策树模型
        """
        params_grid = {
            'max_depth': [3, 4, 5, 6],
            'min_samples_split': [2, 3, 4, 5],
            'min_samples_leaf': [1, 2, 3, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        train_data = items[["disk_capacity",
                            "disk_if_local", "disk_type", "vm_cpu", "vm_mem"]]
        train_data_encoded = pd.get_dummies(
            train_data, columns=["disk_type"], drop_first=True)
        model_classify = GridSearchCV(estimator=DecisionTreeClassifier(
            criterion="gini"), param_grid=params_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
        model_classify.fit(train_data_encoded, labels_cluster)
        logger.info(f"DecisionTree model: {model_classify.best_params_}\n")
        best_clf = model_classify.best_estimator_
        logger.info(f"准确率: {model_classify.best_score_}\n")
        return best_clf

    def predict(self, items: List[List[Any]]):
        """
        预测物品的聚类标签
        input:
            items: 物品列表
        output:
            labels_classify: 物品的聚类标签
        """
        model_cluster_dir = os.path.join(
            DirConfig.MODEL_DIR, "model_cluster.pkl")
        model_classify_dir = os.path.join(
            DirConfig.MODEL_DIR, "model_classify.pkl")
        if not os.path.exists(model_cluster_dir) or not os.path.exists(model_classify_dir):
            self.train_model()
        with open(model_cluster_dir, "rb") as f:
            model_cluster = pickle.load(f)
        with open(model_classify_dir, "rb") as f:
            model_classify = pickle.load(f)
        items = pd.DataFrame(items, columns=["disk_ID", "disk_capacity", "disk_if_local", "disk_attr", "disk_type", "disk_if_VIP",
                             "disk_pay", "vm_cpu", "vm_mem", "avg_rbw", "avg_wbw", "peak_rbw", "peak_wbw", "timestamp_num", "burst_label", "cluster_index"])
        data_predict = items[["disk_capacity",
                              "disk_if_local", "disk_type", "vm_cpu", "vm_mem"]]
        data_predict_encoded = pd.get_dummies(
            data_predict, columns=["disk_type"], drop_first=True)
        pre_labels_classify = model_classify.predict(data_predict_encoded)
        cluster_centers = model_cluster.cluster_centers_
        center_coords = cluster_centers[pre_labels_classify]
        df_centers = pd.DataFrame(center_coords[:, :2], columns=[
                                  "pre_avg_rbw", "pre_avg_wbw"])
        df_selected_cols = items[[
            "disk_ID", "disk_capacity", "timestamp_num", "avg_rbw", "avg_wbw", "cluster_index"]]
        items_predict = pd.concat([df_selected_cols, df_centers], axis=1)
        items_predict.to_csv(os.path.join(
            DirConfig.SCDA_DIR, "items_predict.csv"))

        return items_predict

    def _random_choice(self, items: pd.DataFrame) -> pd.Series:
        """随机选择一个物品"""
        return items.sample()

    def place_item(self, items: pd.DataFrame) -> List[List[List[int]]]:
        """
        进行放置操作
        input:
            items: 物品列表
        output:
            warehouses_trace: 每个时间戳的仓库利用率[timestamp][warehouse][dimension]
        """
        items_placed = 0
        warehouses_trace = [[[0 for _ in range(3)] for _ in range(
            WarehouseConfig.WAREHOUSE_NUMBER)] for _ in range(DataConfig.EVALUATE_TIME_NUMBER)]
        warehouses_resource_allocated = [[0, 0, 0]
                                         for _ in range(WarehouseConfig.WAREHOUSE_NUMBER)]
        warehouses_cannot_use_by_monitor = [
            0 for _ in range(WarehouseConfig.WAREHOUSE_NUMBER)]
        violation_time_queues = [deque(maxlen=DataConfig.VIOLATION_QUEUE_LENGTH)
                                 for _ in range(WarehouseConfig.WAREHOUSE_NUMBER)]

        # 创建items的副本以支持不重复选择（可选）
        available_items = items.copy()

        # while len(items)-items_placed > ModelConfig.EPISODE_REMAIN:  # 疑问:为什么只测试这么多的数据？
        while items_placed < ModelConfig.EPISODE_REMAIN and not available_items.empty:
            # 使用随机选择（可能重复）
            item = self._random_choice(items)

            # 如果要使用不重复选择
            # item = self._random_choice(available_items)
            # available_items.remove(item)

            items_placed += 1
            selected_warehouse = self.select_warehouse(
                item, warehouses_resource_allocated, warehouses_cannot_use_by_monitor)
            if selected_warehouse == -1:
                if 0 not in warehouses_cannot_use_by_monitor:
                    break
                else:
                    cluster_index = item["cluster_index"].iloc[0]
                    disk_ID = item["disk_ID"].iloc[0]
                    logger.warning(
                        f"No warehouse can place cluster{cluster_index} item {disk_ID} at No.{items_placed}.")
                    continue
            # 更新仓库状态
            self.update_warehouse_state(selected_warehouse, item,
                                        warehouses_resource_allocated, warehouses_trace, items_placed)
            self.monitor_warning_violation(
                warehouses_trace, warehouses_cannot_use_by_monitor, violation_time_queues, items_placed)
            if items_placed % 500 == 0:
                logger.info(f"Placed {items_placed} items.")
        logger.info(
            f"Placed {items_placed} items.length of items: {len(items)},episode_remain: {ModelConfig.EPISODE_REMAIN},warehouse_number: {WarehouseConfig.WAREHOUSE_NUMBER}")
        return warehouses_trace

    def select_warehouse(self, item: pd.DataFrame, warehouses_resource_allocated: List[List[int]], warehouses_cannot_use_by_monitor: List[int]) -> int:
        selected_warehouse = -1
        min_manhatten_distance = float('inf')
        disk_capacity = item["disk_capacity"].iloc[0]
        disk_pre_read_BW = item["pre_avg_rbw"].iloc[0]
        disk_pre_write_BW = item["pre_avg_wbw"].iloc[0]
        for warehouse in range(WarehouseConfig.WAREHOUSE_NUMBER):
            if warehouses_cannot_use_by_monitor[warehouse] == 1:
                continue
            if warehouses_resource_allocated[warehouse][0]+disk_capacity > WarehouseConfig.WAREHOUSE_MAX[0][warehouse]*ModelConfig.RESERVATION_RATE_FOR_MONITOR or warehouses_resource_allocated[warehouse][1]+disk_pre_read_BW > WarehouseConfig.WAREHOUSE_MAX[1][warehouse]*ModelConfig.RESERVATION_RATE_FOR_MONITOR or warehouses_resource_allocated[warehouse][2]+disk_pre_write_BW > WarehouseConfig.WAREHOUSE_MAX[2][warehouse]*ModelConfig.RESERVATION_RATE_FOR_MONITOR:
                continue
            current_capacity_utilization = (warehouses_resource_allocated[
                warehouse][0]+disk_capacity) / WarehouseConfig.WAREHOUSE_MAX[0][warehouse]
            current_read_bandwidth_utilization = (warehouses_resource_allocated[
                warehouse][1]+disk_pre_read_BW) / WarehouseConfig.WAREHOUSE_MAX[1][warehouse]
            current_write_bandwidth_utilization = (warehouses_resource_allocated[
                warehouse][2]+disk_pre_write_BW) / WarehouseConfig.WAREHOUSE_MAX[2][warehouse]
            current_utilization_center = (
                current_capacity_utilization+current_read_bandwidth_utilization+current_write_bandwidth_utilization)/3
            current_manhatten_distance = abs(current_capacity_utilization-current_utilization_center)+abs(
                current_read_bandwidth_utilization-current_utilization_center)+abs(current_write_bandwidth_utilization-current_utilization_center)
            if current_manhatten_distance < min_manhatten_distance:
                min_manhatten_distance = current_manhatten_distance
                selected_warehouse = warehouse
        return selected_warehouse

    def evaluate_warehouses(self, warehouses_trace: List[List[List[int]]]) -> Tuple[List[List[float]], List[int], List[int], List[List[int]], List[float], List[float]]:
        """
        评估仓库
        utilize_trace:每个时间戳的平均仓库利用率[timestamp][dimension]
        is_warehouse_violated:每个仓库当前时刻是否违规[warehouse]
        violation_count_per_warehouse:每个仓库累计违规次数[warehouse]
        current_violation_duration:每个仓库当前连续违规时长[warehouse]
        max_violation_duration:每个仓库历史最大连续违规时长[warehouse]
        all_violation_durations:每个仓库所有连续违规时长的列表[warehouse][n]
        """
        utilize_trace = []
        violation_count_per_warehouse = [
            0 for _ in range(WarehouseConfig.WAREHOUSE_NUMBER)]
        current_violation_duration = [
            0 for _ in range(WarehouseConfig.WAREHOUSE_NUMBER)]
        max_violation_duration = [0 for _ in range(
            WarehouseConfig.WAREHOUSE_NUMBER)]
        all_violation_durations = [[]
                                   for _ in range(WarehouseConfig.WAREHOUSE_NUMBER)]
        average_dimension_load_imb = np.zeros(WarehouseConfig.WAREHOUSE_NUMBER)
        average_warehouse_load_imb = np.zeros(3)

        for time_idx in range(DataConfig.EVALUATE_TIME_NUMBER):
            # 计算利用率
            warehouses_trace_now = warehouses_trace[time_idx]
            warehouses_trace_now = np.array(warehouses_trace_now)
            warehouses_max_array = np.array(WarehouseConfig.WAREHOUSE_MAX).T
            warehouses_utilize_now = warehouses_trace_now/warehouses_max_array
            utilize_now = np.mean(warehouses_utilize_now, axis=0)
            utilize_trace.append(utilize_now)

            # 计算SLA违反
            is_warehouse_violated = [0 for _ in range(
                WarehouseConfig.WAREHOUSE_NUMBER)]
            for warehouse in range(WarehouseConfig.WAREHOUSE_NUMBER):
                for dim in range(3):
                    if warehouses_utilize_now[warehouse][dim] >= 1:
                        is_warehouse_violated[warehouse] = 1
                        break

            for warehouse in range(WarehouseConfig.WAREHOUSE_NUMBER):
                if is_warehouse_violated[warehouse] == 1:
                    violation_count_per_warehouse[warehouse] += 1
                    current_violation_duration[warehouse] += 1

                else:
                    if current_violation_duration[warehouse] != 0:
                        all_violation_durations[warehouse].append(
                            current_violation_duration[warehouse])
                        if current_violation_duration[warehouse] > max_violation_duration[warehouse]:
                            max_violation_duration[warehouse] = current_violation_duration[warehouse]
                        current_violation_duration[warehouse] = 0

            # 计算负载不均衡度
            warehouse_utilize_var = np.var(warehouses_utilize_now, axis=0)
            warehouse_utilize_std = np.std(warehouses_utilize_now, axis=0)
            warehouse_utilize_mean = np.mean(warehouses_utilize_now, axis=0)
            warehouse_load_imb = np.zeros_like(warehouse_utilize_std)
            for dim_idx in range(len(warehouse_utilize_mean)):
                if warehouse_utilize_mean[dim_idx] == 0:
                    warehouse_load_imb[dim_idx] = 0
                    logger.warning(f"除数为0，资源维度{dim_idx}的负载不均衡度设置为0")
                else:
                    warehouse_load_imb[dim_idx] = warehouse_utilize_std[dim_idx] / \
                        warehouse_utilize_mean[dim_idx]

            dimension_utilize_var = np.var(warehouses_utilize_now, axis=1)
            dimension_utilize_std = np.std(warehouses_utilize_now, axis=1)
            dimension_utilize_mean = np.mean(warehouses_utilize_now, axis=1)
            dimension_load_imb = np.zeros_like(dimension_utilize_std)
            for warehouse_idx in range(len(dimension_utilize_mean)):
                if dimension_utilize_mean[warehouse_idx] == 0:
                    dimension_load_imb[warehouse_idx] = 0
                    logger.warning(f"除数为0，仓库{warehouse_idx}中资源的负载不均衡度设置为0")
                else:
                    dimension_load_imb[warehouse_idx] = dimension_utilize_std[warehouse_idx] / \
                        dimension_utilize_mean[warehouse_idx]

            # 累加负载不均衡度（移到内层循环外）
            average_dimension_load_imb += dimension_load_imb
            average_warehouse_load_imb += warehouse_load_imb

        # 计算平均值
        average_dimension_load_imb /= DataConfig.EVALUATE_TIME_NUMBER
        average_warehouse_load_imb /= DataConfig.EVALUATE_TIME_NUMBER

        return utilize_trace, violation_count_per_warehouse, max_violation_duration, all_violation_durations, average_dimension_load_imb, average_warehouse_load_imb

    def update_warehouse_state(self, selected_warehouse: int, item: pd.DataFrame, warehouses_resource_allocated: List[List[int]], warehouses_trace: List[List[List[int]]], current_time: int) -> None:
        """
        更新仓库状态
        由于进行云盘放置模拟时是每个时间戳放置一个云盘，所以当前时间戳等于已经放置的云盘数量
        input:
            selected_warehouse: 选择的仓库编号
            item: 要放置的云盘
            warehouses_cannot_use_by_monitor: 不能使用的仓库列表
            warehouses_resource_allocated: 目前仓库已经分配的资源
            warehouses_trace: 全生命周期内每个时间戳的仓库资源使用情况
            current_time: 当前时间戳，由于进行云盘放置模拟时是每个时间戳放置一个云盘，所以当前时间戳等于已经放置的云盘数量
        """
        warehouses_resource_allocated[selected_warehouse][0] += item["disk_capacity"].iloc[0]
        warehouses_resource_allocated[selected_warehouse][1] += item["pre_avg_rbw"].iloc[0]
        warehouses_resource_allocated[selected_warehouse][2] += item["pre_avg_wbw"].iloc[0]
        cluster_index = item["cluster_index"].iloc[0]
        disk_ID = item["disk_ID"].iloc[0]
        trace_dir = os.path.join(DirConfig.TRACE_ROOT,
                                 f"20_136090{cluster_index}", f"{disk_ID}")
        trace_lines = linecache.getlines(trace_dir)
        for time in range(current_time, DataConfig.EVALUATE_TIME_NUMBER):
            fields = trace_lines[time].strip().split(",")
            # IOPS = int(fields[1])+int(fields[3])
            # bandwidth = int(fields[2])+int(fields[4])
            read_bandwidth = int(fields[2])
            write_bandwidth = int(fields[4])
            capacity = int(item["disk_capacity"].iloc[0])
            warehouses_trace[time][selected_warehouse][0] += capacity
            warehouses_trace[time][selected_warehouse][1] += read_bandwidth
            warehouses_trace[time][selected_warehouse][2] += write_bandwidth

    def monitor_warning_violation(self, warehouses_trace: List[List[List[int]]], warehouses_cannot_use_by_monitor: List[int], warehouse_violation_time_queues: List[deque], current_time: int) -> None:
        """
        监控SLA违反,如果一个仓库在短时间内发生多次资源利用率超过阈值，则认为该仓库发生SLA违反，后续放置不再考虑该仓库
        """
        for warehouse in range(WarehouseConfig.WAREHOUSE_NUMBER):
            if warehouses_cannot_use_by_monitor[warehouse] == 1:
                continue
            if (warehouses_trace[current_time][warehouse][0] > WarehouseConfig.WAREHOUSE_MAX[0][warehouse] * ModelConfig.RESERVATION_RATE_FOR_MONITOR or
                warehouses_trace[current_time][warehouse][1] > WarehouseConfig.WAREHOUSE_MAX[1][warehouse] * ModelConfig.RESERVATION_RATE_FOR_MONITOR or
                    warehouses_trace[current_time][warehouse][2] > WarehouseConfig.WAREHOUSE_MAX[2][warehouse] * ModelConfig.RESERVATION_RATE_FOR_MONITOR):
                if len(warehouse_violation_time_queues[warehouse]) == DataConfig.VIOLATION_QUEUE_LENGTH:
                    oldest_violation_time = warehouse_violation_time_queues[warehouse].popleft(
                    )
                    if current_time - oldest_violation_time <= DataConfig.MIN_VIOLATION_TIME_WINDOW:
                        warehouses_cannot_use_by_monitor[warehouse] = 1
                warehouse_violation_time_queues[warehouse].append(current_time)
