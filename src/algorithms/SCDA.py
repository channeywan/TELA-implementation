from typing import List, Any, Tuple
import numpy as np
import pandas as pd
import os
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score
import logging

from .base_algorithm import BaseAlgorithm
from config.settings import WarehouseConfig, DataConfig, ModelConfig, DirConfig

logger = logging.getLogger(__name__)


class SCDA(BaseAlgorithm):
    """SCDA算法实现"""

    def __init__(self):
        super().__init__("SCDA")

    def load_and_preprocess_items(self):
        """加载和预处理数据"""
        items = self.loader.load_items(
            type="both", cluster_index_list=DataConfig.CLUSTER_INDEX_LIST_PREDICT, purpose="train")
        return self.predict(items)

    def train_model(self):
        """训练模型"""
        items = self.loader.load_items(
            type="both", cluster_index_list=DataConfig.CLUSTER_INDEX_LIST_TRAIN, purpose="train")
        items = pd.DataFrame(items, columns=["disk_ID", "disk_capacity", "disk_if_local", "disk_attr",
                                             "disk_type", "disk_if_VIP", "disk_pay", "vm_cpu", "vm_mem",
                                             "avg_rbw", "avg_wbw", "peak_rbw", "peak_wbw", "timestamp_num",
                                             "burst_label", "cluster_index"])

        model_cluster, labels_cluster = self.train_Kmeans(items)
        model_classify = self.train_DecisionTree(items, labels_cluster)

        # 保存模型
        os.makedirs(os.path.join(DirConfig.MODEL_DIR, "SCDA"), exist_ok=True)
        model_cluster_dir = os.path.join(
            DirConfig.MODEL_DIR, "SCDA", "model_cluster.pkl")
        model_classify_dir = os.path.join(
            DirConfig.MODEL_DIR, "SCDA", "model_classify.pkl")
        scaler_dir = os.path.join(
            DirConfig.MODEL_DIR, "SCDA", "scaler.pkl")

        with open(model_cluster_dir, "wb") as f:
            pickle.dump(model_cluster, f)
        with open(model_classify_dir, "wb") as f:
            pickle.dump(model_classify, f)
        with open(scaler_dir, "wb") as f:
            pickle.dump(self.scaler, f)

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
        cluster_K = ModelConfig.SCDA_CLUSTER_K

        # 保存scaler以便预测时使用
        self.scaler = MinMaxScaler()
        train_data_scaled = self.scaler.fit_transform(train_data)

        model_cluster = KMeans(n_clusters=cluster_K)
        model_cluster.fit(train_data_scaled)
        labels_cluster = model_cluster.labels_
        cluster_centers = model_cluster.cluster_centers_
        unique_labels, counts = np.unique(labels_cluster, return_counts=True)
        the_silhouette_score = silhouette_score(
            train_data_scaled, labels_cluster)
        logger.info(f"Silhouette score: {the_silhouette_score}")

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
        train_data_encoded = self.encode_item(train_data)

        model_classify = GridSearchCV(
            estimator=DecisionTreeClassifier(criterion="gini"),
            param_grid=params_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=0
        )
        model_classify.fit(train_data_encoded, labels_cluster)

        logger.info(f"DecisionTree model: {model_classify.best_params_}")
        logger.info(f"准确率: {model_classify.best_score_}")

        return model_classify.best_estimator_

    def predict(self, items: pd.DataFrame):
        """
        预测物品的聚类标签
        input:
            items: 物品列表
        output:
            labels_classify: 物品的聚类标签
        """
        model_cluster, model_classify = self.load_model()
        data_predict = items[["disk_capacity",
                              "disk_if_local", "disk_type", "vm_cpu", "vm_mem"]]

        # 确保预测数据有相同的列
        data_predict_encoded = self.encode_item(data_predict)

        pre_labels_classify = model_classify.predict(data_predict_encoded)
        cluster_centers = model_cluster.cluster_centers_
        # 反缩放聚类中心到原始尺度
        cluster_centers_original = self.scaler.inverse_transform(
            cluster_centers)
        center_coords = cluster_centers_original[pre_labels_classify]
        df_centers = pd.DataFrame(center_coords[:, :2], columns=[
                                  "pre_rbw", "pre_wbw"])
        df_selected_cols = items[["disk_ID", "disk_capacity",
                                  "timestamp_num", "avg_rbw", "avg_wbw", "cluster_index"]]
        items_predict = pd.concat([df_selected_cols, df_centers], axis=1)

        items_predict.to_csv(os.path.join(
            DirConfig.SCDA_DIR, "items_predict.csv"))
        return items_predict

    def select_warehouse(self, item: pd.Series, warehouses_resource_allocated: np.ndarray,
                         warehouses_cannot_use_by_monitor: List[int], _: Any = None) -> int:
        """选择仓库（基于曼哈顿距离的负载均衡策略）"""
        selected_warehouse = -1
        min_manhatten_distance = float('inf')

        disk_capacity = item["disk_capacity"]
        disk_pre_read_BW = item["pre_rbw"]
        disk_pre_write_BW = item["pre_wbw"]

        for warehouse in range(WarehouseConfig.WAREHOUSE_NUMBER):
            if warehouses_cannot_use_by_monitor[warehouse] == 1:
                continue

            # 检查资源容量约束
            if (warehouses_resource_allocated[warehouse][0] + disk_capacity > WarehouseConfig.WAREHOUSE_MAX[0][warehouse] * ModelConfig.RESERVATION_RATE_FOR_MONITOR or
                warehouses_resource_allocated[warehouse][1] + disk_pre_read_BW > WarehouseConfig.WAREHOUSE_MAX[1][warehouse] * ModelConfig.RESERVATION_RATE_FOR_MONITOR or
                    warehouses_resource_allocated[warehouse][2] + disk_pre_write_BW > WarehouseConfig.WAREHOUSE_MAX[2][warehouse] * ModelConfig.RESERVATION_RATE_FOR_MONITOR):
                continue

            # 计算放置后的利用率
            current_capacity_utilization = ((warehouses_resource_allocated[warehouse][0] + disk_capacity) /
                                            WarehouseConfig.WAREHOUSE_MAX[0][warehouse])
            current_read_bandwidth_utilization = ((warehouses_resource_allocated[warehouse][1] + disk_pre_read_BW) /
                                                  WarehouseConfig.WAREHOUSE_MAX[1][warehouse])
            current_write_bandwidth_utilization = ((warehouses_resource_allocated[warehouse][2] + disk_pre_write_BW) /
                                                   WarehouseConfig.WAREHOUSE_MAX[2][warehouse])

            # 计算利用率中心点和曼哈顿距离
            current_utilization_center = ((current_capacity_utilization + current_read_bandwidth_utilization +
                                           current_write_bandwidth_utilization) / 3)
            current_manhatten_distance = (abs(current_capacity_utilization - current_utilization_center) +
                                          abs(current_read_bandwidth_utilization - current_utilization_center) +
                                          abs(current_write_bandwidth_utilization - current_utilization_center))

            if current_manhatten_distance < min_manhatten_distance:
                min_manhatten_distance = current_manhatten_distance
                selected_warehouse = warehouse

        return selected_warehouse

    def load_model(self) -> Tuple[KMeans, DecisionTreeClassifier]:
        model_cluster_dir = os.path.join(
            DirConfig.MODEL_DIR, "SCDA", "model_cluster.pkl")
        model_classify_dir = os.path.join(
            DirConfig.MODEL_DIR, "SCDA", "model_classify.pkl")
        scaler_dir = os.path.join(
            DirConfig.MODEL_DIR, "SCDA", "scaler.pkl")
        if not os.path.exists(model_cluster_dir) or not os.path.exists(model_classify_dir) or not os.path.exists(scaler_dir):
            self.train_model()

        with open(model_cluster_dir, "rb") as f:
            model_cluster = pickle.load(f)
        with open(model_classify_dir, "rb") as f:
            model_classify = pickle.load(f)
        with open(scaler_dir, "rb") as f:
            self.scaler = pickle.load(f)
        return model_cluster, model_classify

    def additional_state_update(self, selected_warehouse: int, item: pd.Series, warehouses_resource_allocated: np.ndarray, warehouses_trace: np.ndarray, current_time: int, additional_state: Any):
        warehouses_resource_allocated[selected_warehouse][1] += item["pre_rbw"]
        warehouses_resource_allocated[selected_warehouse][2] += item["pre_wbw"]
        return
