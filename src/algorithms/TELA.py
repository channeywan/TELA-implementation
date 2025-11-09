import logging
from typing import List, Any, Tuple
import numpy as np
import pandas as pd
import os
import pickle
import random
from pyearth import Earth
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from .base_algorithm import BaseAlgorithm
from .ODA import ODA
from .SCDA import SCDA
from config.settings import DataConfig, WarehouseConfig, ModelConfig, DirConfig
import joblib
from tqdm import tqdm
logger = logging.getLogger(__name__)


class TELA(BaseAlgorithm):
    """TELA算法实现 - 基于时间序列负载均衡的算法"""

    def __init__(self):
        super().__init__("TELA")
        self.warehouse_burst_type_counts = np.zeros(
            (WarehouseConfig.WAREHOUSE_NUMBER, ModelConfig.TELA_CLUSTER_K), dtype=int)
        self.train_disks_trace = {}

    def load_and_preprocess_items(self):
        """加载和预处理数据"""
        # 加载预测数据
        items = self.loader.load_items(
            type="both",
            cluster_index_list=DataConfig.CLUSTER_INDEX_LIST_PREDICT,
            purpose="train"
        )
        for cluster_index in DataConfig.CLUSTER_INDEX_LIST_PREDICT:
            trace_dir = os.path.join(
                DirConfig.CLUSTER_TRACE_DB_ROOT, f"cluster_{cluster_index}_trace.pkl")
            self.disks_trace[cluster_index] = joblib.load(trace_dir)
        items_encoded = self.encode_item(items)
        # 预测磁盘类型和负载
        return self.predict_disk_types_and_loads(items_encoded)

    def train_models(self):
        """训练TELA所需的所有模型"""
        logger.info("开始训练TELA模型")

        # 加载训练数据
        items_train = self.loader.load_items(
            type="both",
            cluster_index_list=DataConfig.CLUSTER_INDEX_LIST_TRAIN,
            purpose="train"
        )
        for cluster_index in DataConfig.CLUSTER_INDEX_LIST_TRAIN:
            trace_dir = os.path.join(
                DirConfig.CLUSTER_TRACE_DB_ROOT, f"cluster_{cluster_index}_trace.pkl")
            self.train_disks_trace[cluster_index] = joblib.load(trace_dir)
        items_train_encoded = self.encode_item(items_train)

        # 第一阶段：训练稳定型/突发型分类模型
        model_classify_type = self.train_type_classifier(
            items_train_encoded)

        # 第二阶段：训练细分聚类和分类模型
        cluster_stable, cluster_burst, classify_stable, classify_burst, stable_scaler, burst_scaler = self.train_cluster_classifier(
            items_train_encoded)

        items_burst = items_train_encoded[items_train_encoded["burst_label"] == 1].copy(
        )
        items_burst["cluster_label"] = cluster_burst.labels_
        # 训练负载拟合模型
        warehouse_counts_snapshots, warehouse_peak_load_snap, warehouse_avg_load_snap = self.simulate_warehouse_load(
            items_burst)
        peak_bandwidth_model, avg_bandwidth_model = self.train_warehouse_load_model(
            warehouse_counts_snapshots, warehouse_peak_load_snap, warehouse_avg_load_snap)

        # 保存所有模型
        self.save_models(model_classify_type, cluster_stable, cluster_burst,
                         classify_stable, classify_burst,  stable_scaler, burst_scaler, peak_bandwidth_model, avg_bandwidth_model)

        logger.info("TELA模型训练完成")

    def train_type_classifier(self, items: pd.DataFrame):
        """训练第一阶段分类器：区分稳定型和突发型磁盘"""
        logger.info("开始训练第一阶段分类器：区分稳定型和突发型磁盘")
        params_grid = {
            'max_depth': [3, 4, 5, 6, 8],
            'min_samples_split': [2, 3, 4, 5],
            'min_samples_leaf': [1, 2, 3, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        train_features = items[["disk_capacity", "disk_type", "disk_if_VIP",
                                "vm_cpu", "vm_mem"]]

        model_classify_type = GridSearchCV(
            estimator=DecisionTreeClassifier(criterion="gini"),
            param_grid=params_grid, cv=5, scoring='accuracy', n_jobs=1, verbose=0
        )
        model_classify_type.fit(train_features, items["burst_label"])

        # 计算准确率
        best_model = model_classify_type.best_estimator_
        logger.info(f"区分磁盘负载类型分类器:{model_classify_type.best_params_}")
        logger.info(f"准确率:{model_classify_type.best_score_}")

        return best_model

    def train_cluster_classifier(self, items: pd.DataFrame):
        """训练第二阶段模型：每个类别内部的聚类和分类"""
        logger.info("开始训练第二阶段模型：每个类别内部的聚类和分类")
        cluster_K = ModelConfig.TELA_CLUSTER_K
        params_grid = {
            'max_depth': [3, 4, 5, 6, 8],
            'min_samples_split': [2, 3, 4, 5],
            'min_samples_leaf': [1, 2, 3, 4],
            'max_features': ['sqrt', 'log2', None]
        }

        # 稳定型和突发型聚类
        items_stable = items[items["burst_label"] == 0]
        items_burst = items[items["burst_label"] == 1]
        logger.info(f"稳定型磁盘数量：{len(items_stable)}")
        logger.info(f"突发型磁盘数量：{len(items_burst)}")
        stable_features = items_stable[["avg_bandwidth"]]
        burst_features = items_burst[["peak_bandwidth"]]
        # 对训练数据进行缩放
        stable_scaler = MinMaxScaler()
        burst_scaler = MinMaxScaler()
        stable_features_scaled = stable_scaler.fit_transform(stable_features)
        burst_features_scaled = burst_scaler.fit_transform(burst_features)
        cluster_stable = KMeans(n_clusters=cluster_K)
        cluster_stable.fit(stable_features_scaled)
        score_stable = silhouette_score(
            stable_features_scaled, cluster_stable.labels_)
        logger.info(f"稳定型聚类轮廓系数:{score_stable}")
        cluster_burst = KMeans(n_clusters=cluster_K)
        cluster_burst.fit(burst_features_scaled)
        score_burst = silhouette_score(
            burst_features_scaled, cluster_burst.labels_)
        logger.info(f"突发型聚类轮廓系数:{score_burst}")
        # self.plotter.plot_kMeans_cluster(
        #     burst_features, stable_features, cluster_burst.labels_, cluster_stable.labels_)
        model_classify_stable = GridSearchCV(
            estimator=DecisionTreeClassifier(criterion="gini"),
            param_grid=params_grid, cv=5, scoring='accuracy', n_jobs=1, verbose=0
        )
        model_classify_burst = GridSearchCV(
            estimator=DecisionTreeClassifier(criterion="gini"),
            param_grid=params_grid, cv=5, scoring='accuracy', n_jobs=1, verbose=0
        )
        model_classify_stable.fit(items_stable[[
                                  "disk_capacity", "disk_type", "disk_if_VIP", "vm_cpu", "vm_mem"]], cluster_stable.labels_)
        model_classify_burst.fit(items_burst[[
                                 "disk_capacity", "disk_type", "disk_if_VIP", "vm_cpu", "vm_mem"]], cluster_burst.labels_)
        best_classify_stable = model_classify_stable.best_estimator_
        best_classify_burst = model_classify_burst.best_estimator_
        logger.info(f"稳定型分类器:{model_classify_stable.best_params_}")
        logger.info(f"准确率:{model_classify_stable.best_score_}")
        logger.info(f"突发型分类器:{model_classify_burst.best_params_}")
        logger.info(f"准确率:{model_classify_burst.best_score_}")

        # sorted_centers = sorted(enumerate(cluster_burst.cluster_centers_),
        #                         key=lambda x: x[1][0])
        # self.burst_map = [x[0] for x in sorted_centers][::-1]

        return cluster_stable, cluster_burst, best_classify_stable, best_classify_burst, stable_scaler, burst_scaler

    def train_warehouse_load_model(self, warehouse_counts_snapshots, warehouse_peak_load_snap, warehouse_avg_load_snap):
        """训练负载拟合模型"""
        logger.info("开始训练负载拟合模型")
        X_train = warehouse_counts_snapshots
        # peak_RBW_model = pwlf.PiecewiseLinFit(X_train, simulate_load[:, 0])
        # peak_WBW_model = pwlf.PiecewiseLinFit(X_train, simulate_load[:, 1])
        # n_segments_range = range(2, 6)
        # result_RBW = np.zeros(len(n_segments_range))
        # result_WBW = np.zeros(len(n_segments_range))
        # for i, n_segments in enumerate(n_segments_range):
        #     peak_RBW_model.fitfast(n_segments=n_segments)
        #     result_RBW[i] = peak_RBW_model.bic
        #     peak_WBW_model.fitfast(n_segments=n_segments)
        #     result_WBW[i] = peak_WBW_model.bic
        # best_n_segments_RBW = n_segments_range[np.argmin(result_RBW)]
        # best_n_segments_WBW = n_segments_range[np.argmin(result_WBW)]
        # logger.info(f"RBW负载拟合模型最佳段数: {best_n_segments_RBW}")
        # logger.info(f"WBW负载拟合模型最佳段数: {best_n_segments_WBW}")
        # peak_RBW_model = pwlf.PiecewiseLinFit(X_train, simulate_load[:, 0])
        # peak_RBW_model.fit(n_segments=best_n_segments_RBW)
        # peak_WBW_model = pwlf.PiecewiseLinFit(X_train, simulate_load[:, 1])
        # peak_WBW_model.fit(n_segments=best_n_segments_WBW)
        # logger.info(f"RBW负载拟合模型 BIC分数: {peak_RBW_model.bic:.4f}")
        # logger.info(f"WBW负载拟合模型 BIC分数: {peak_WBW_model.bic:.4f}")
        peak_bandwidth_model = Earth(max_degree=1)
        avg_bandwidth_model = Earth(max_degree=1)
        peak_bandwidth_model.fit(X_train, warehouse_peak_load_snap)
        avg_bandwidth_model.fit(X_train, warehouse_avg_load_snap)
        return peak_bandwidth_model, avg_bandwidth_model

    def simulate_warehouse_load(self, items: pd.DataFrame):
        """
        simulate_load_trace: 模拟负载的时序数据
        placed_warehouses: 磁盘放置的仓库
        warehouse_load_snap:新增每个磁盘时，磁盘对应仓库的负载峰值和均值指标[len(items),(RBW_max,WBW_max,RBW_avg,WBW_avg)]
        """
        logger.info("使用oda模拟仓库放置")
        oda = ODA()
        oda.disks_trace = self.train_disks_trace
        placed_warehouses, avaliable_items = oda.place_item(items)
        placed_items = avaliable_items.iloc[np.where(
            placed_warehouses != -1)[0]]
        placed_warehouses = placed_warehouses[
            placed_warehouses != -1]
        simulate_load_trace = np.zeros(
            (WarehouseConfig.WAREHOUSE_NUMBER, DataConfig.EVALUATE_TIME_NUMBER))
        warehouse_category_counts = np.zeros(
            (WarehouseConfig.WAREHOUSE_NUMBER, ModelConfig.TELA_CLUSTER_K), dtype=int)
        warehouse_counts_snapshots = np.zeros(
            (len(placed_items), ModelConfig.TELA_CLUSTER_K), dtype=int)
        warehouse_peak_load_snap = np.zeros(len(placed_items))
        warehouse_avg_load_snap = np.zeros(len(placed_items))
        for i, item in tqdm(placed_items.reset_index(drop=True).iterrows(), total=len(placed_items), desc="模拟仓库突发负载"):
            warehouse_idx = placed_warehouses[i]
            disk_trace_bandwidth = self.train_disks_trace[item["cluster_index"]
                                                          ][item["disk_ID"]]
            first_day_line = self._iterate_first_day(
                disk_trace_bandwidth["timestamp"])
            circle_trace = self._get_circular_trace(
                disk_trace_bandwidth, item["disk_capacity"], first_day_line, DataConfig.EVALUATE_TIME_NUMBER)

            simulate_load_trace[warehouse_idx, :] += circle_trace[:, 1]
            warehouse_peak_load_snap[i] = np.max(
                simulate_load_trace[warehouse_idx, :])
            warehouse_avg_load_snap[i] = np.mean(
                simulate_load_trace[warehouse_idx, :])
            warehouse_category_counts[warehouse_idx,
                                      item["cluster_label"]] += 1
            warehouse_counts_snapshots[i] = warehouse_category_counts[warehouse_idx, :]
        placed_items.to_csv(os.path.join(
            DirConfig.TELA_DIR, "train_items.csv"))
        return warehouse_counts_snapshots, warehouse_peak_load_snap, warehouse_avg_load_snap

    def predict_disk_types_and_loads(self, items: pd.DataFrame):
        """预测磁盘类型和负载"""
        # 加载训练好的模型
        type_classifier, cluster_stable, cluster_burst, classify_stable, classify_burst, stable_scaler, burst_scaler, self.peak_bandwidth_model, self.avg_bandwidth_model = self.load_item_predict_models()

        # 决策树分类磁盘类型，区分稳定型和突发型
        logger.info("开始决策树分类磁盘类型，区分稳定型和突发型")
        type_classify_features = items[["disk_capacity", "disk_type", "disk_if_VIP",
                                        "vm_cpu", "vm_mem"]]
        type_classify_labels = type_classifier.predict(type_classify_features)
        items["pre_burst_label"] = type_classify_labels
        logger.info(
            f"稳定突发区分决策树预测准确率: {np.mean(type_classify_labels == items['burst_label']):.4f}")
        # 突发型和稳定型磁盘分别分类到对应聚类中的某一簇
        logger.info("开始突发型和稳定型磁盘分别分类到对应聚类中的某一簇")
        burst_items = items[items["pre_burst_label"]
                            == 1].copy().reset_index(drop=True)
        stable_items = items[items["pre_burst_label"]
                             == 0].copy().reset_index(drop=True)
        burst_items_belong_to_cluster = classify_burst.predict(burst_items[[
            "disk_capacity", "disk_type", "disk_if_VIP", "vm_cpu", "vm_mem"]])
        stable_items_belong_to_cluster = classify_stable.predict(stable_items[[
            "disk_capacity", "disk_type", "disk_if_VIP", "vm_cpu", "vm_mem"]])
        # 将磁盘特征与所属簇序号拼接
        burst_items["belong_to_cluster"] = burst_items_belong_to_cluster
        stable_items["belong_to_cluster"] = stable_items_belong_to_cluster
        # 反缩放聚类中心到原有尺度
        burst_items_belong_to_cluster_centers = burst_scaler.inverse_transform(
            cluster_burst.cluster_centers_[burst_items_belong_to_cluster])
        stable_items_belong_to_cluster_centers = stable_scaler.inverse_transform(
            cluster_stable.cluster_centers_[stable_items_belong_to_cluster])
        # 将聚类中心与磁盘特征拼接
        # 对于稳定型磁盘来说，pre_rbw和pre_wbw是平均负载，对于突发型磁盘来说，pre_rbw和pre_wbw是预测的峰值负载
        burst_items["pre_bandwidth"] = burst_items_belong_to_cluster_centers
        stable_items["pre_bandwidth"] = stable_items_belong_to_cluster_centers

        items_with_cluster_centers = pd.concat(
            [burst_items, stable_items], axis=0, ignore_index=True)
        items_with_cluster_centers.to_csv(os.path.join(
            DirConfig.TELA_DIR, "predict_items.csv"))
        logger.info("预测磁盘类型和负载完成")
        return items_with_cluster_centers

    def select_warehouse(self, item: pd.Series) -> int:
        """TELA的仓库选择策略：基于时间序列预测的负载均衡"""
        selected_warehouse = -1

        if item["pre_burst_label"] == 0:
            min_manhatten_distance = float('inf')
            disk_capacity = item["disk_capacity"]
            disk_pre_bandwidth = item["pre_bandwidth"]
            for warehouse in range(WarehouseConfig.WAREHOUSE_NUMBER):
                if self.warehouses_cannot_use_by_monitor[warehouse] == 1:
                    continue

                # 检查资源容量约束
                if np.any(self.warehouses_resource_allocated[warehouse] + np.array([disk_capacity, disk_pre_bandwidth]) > WarehouseConfig.WAREHOUSE_MAX[warehouse] * ModelConfig.RESERVATION_RATE_FOR_MONITOR):
                    continue

                # 计算放置后的利用率
                current_capacity_utilization = ((self.warehouses_resource_allocated[warehouse][0] + disk_capacity) /
                                                WarehouseConfig.WAREHOUSE_MAX[warehouse, 0])
                current_bandwidth_utilization = ((self.warehouses_resource_allocated[warehouse][1] + disk_pre_bandwidth) /
                                                 WarehouseConfig.WAREHOUSE_MAX[warehouse, 1])

                # 计算利用率中心点和曼哈顿距离
                current_utilization_center = (
                    (current_capacity_utilization + current_bandwidth_utilization) / 2)
                current_manhatten_distance = (abs(current_capacity_utilization - current_utilization_center) +
                                              abs(current_bandwidth_utilization - current_utilization_center))

                if current_manhatten_distance < min_manhatten_distance:
                    min_manhatten_distance = current_manhatten_distance
                    selected_warehouse = warehouse
        else:
            min_peak_resource_usage = np.inf
            for warehouse in range(WarehouseConfig.WAREHOUSE_NUMBER):
                if self.warehouses_cannot_use_by_monitor[warehouse] == 1:
                    continue
                if self.warehouses_resource_allocated[warehouse][0] + item["disk_capacity"] > WarehouseConfig.WAREHOUSE_MAX[warehouse, 0] * ModelConfig.RESERVATION_RATE_FOR_MONITOR:
                    continue
                train_X = self.warehouse_burst_type_counts[warehouse]+np.array(
                    [1 if i == item["belong_to_cluster"] else 0 for i in range(ModelConfig.TELA_CLUSTER_K)])
                train_X = np.array([train_X])
                predict_warehouse_peak_bandwidth = self.peak_bandwidth_model.predict(
                    train_X)
                if predict_warehouse_peak_bandwidth+self.warehouses_resource_allocated[warehouse][1] > WarehouseConfig.WAREHOUSE_MAX[warehouse, 1] * ModelConfig.PEAK_PREDICTION_TOLERANCE_FACTOR:
                    continue
                peak_resource_usage = predict_warehouse_peak_bandwidth / \
                    WarehouseConfig.WAREHOUSE_MAX[warehouse, 1]
                if peak_resource_usage < min_peak_resource_usage:
                    min_peak_resource_usage = peak_resource_usage
                    selected_warehouse = warehouse
        return selected_warehouse

    def additional_state_update(self, selected_warehouse: int, item: pd.Series):
        if item["pre_burst_label"] == 0:
            self.warehouses_resource_allocated[selected_warehouse][1] += item["pre_bandwidth"]
        else:
            self.warehouse_burst_type_counts[selected_warehouse][item["belong_to_cluster"]] += 1
        return

    def save_models(self, type_classifier, cluster_stable, cluster_burst, classify_stable, classify_burst, stable_scaler, burst_scaler, peak_bandwidth_model, avg_bandwidth_model):
        """保存所有训练好的模型"""
        os.makedirs(os.path.join(DirConfig.MODEL_DIR, "TELA"), exist_ok=True)

        # 保存第一阶段分类器
        with open(os.path.join(DirConfig.MODEL_DIR, "TELA", "type_classifier.pkl"), "wb") as f:
            pickle.dump(type_classifier, f)

        # 保存第二阶段模型
        with open(os.path.join(DirConfig.MODEL_DIR, "TELA", "cluster_stable.pkl"), "wb") as f:
            pickle.dump(cluster_stable, f)
        with open(os.path.join(DirConfig.MODEL_DIR, "TELA", "cluster_burst.pkl"), "wb") as f:
            pickle.dump(cluster_burst, f)
        with open(os.path.join(DirConfig.MODEL_DIR, "TELA", "classify_stable.pkl"), "wb") as f:
            pickle.dump(classify_stable, f)
        with open(os.path.join(DirConfig.MODEL_DIR, "TELA", "classify_burst.pkl"), "wb") as f:
            pickle.dump(classify_burst, f)
        with open(os.path.join(DirConfig.MODEL_DIR, "TELA", "stable_scaler.pkl"), "wb") as f:
            pickle.dump(stable_scaler, f)
        with open(os.path.join(DirConfig.MODEL_DIR, "TELA", "burst_scaler.pkl"), "wb") as f:
            pickle.dump(burst_scaler, f)
        # 保存拟合模型
        with open(os.path.join(DirConfig.MODEL_DIR, "TELA", "peak_bandwidth_model.pkl"), "wb") as f:
            pickle.dump(peak_bandwidth_model, f)
        with open(os.path.join(DirConfig.MODEL_DIR, "TELA", "avg_bandwidth_model.pkl"), "wb") as f:
            pickle.dump(avg_bandwidth_model, f)

        logger.info("TELA模型已保存")

    def load_item_predict_models(self):
        """
        加载训练好的模型
        type_classifier:对磁盘进行分类，区分稳定型和突发型
        cluster_stable:稳定型磁盘聚类
        cluster_burst:突发型磁盘聚类
        classify_stable:稳定型磁盘分类到聚类中的某一簇
        classify_burst:突发型磁盘分类到聚类中的某一簇
        warehouse_burst_load_model:负载拟合模型
        stable_scaler:稳定型磁盘缩放器，用于将磁盘负载缩放到0-1之间
        burst_scaler:突发型磁盘缩放器，用于将磁盘负载缩放到0-1之间
        """

        try:
            # 加载第一阶段分类器
            with open(os.path.join(DirConfig.MODEL_DIR, "TELA", "type_classifier.pkl"), "rb") as f:
                type_classifier = pickle.load(f)

            # 加载第二阶段模型
            with open(os.path.join(DirConfig.MODEL_DIR, "TELA", "cluster_stable.pkl"), "rb") as f:
                cluster_stable = pickle.load(f)
            with open(os.path.join(DirConfig.MODEL_DIR, "TELA", "cluster_burst.pkl"), "rb") as f:
                cluster_burst = pickle.load(f)
            with open(os.path.join(DirConfig.MODEL_DIR, "TELA", "classify_stable.pkl"), "rb") as f:
                classify_stable = pickle.load(f)
            with open(os.path.join(DirConfig.MODEL_DIR, "TELA", "classify_burst.pkl"), "rb") as f:
                classify_burst = pickle.load(f)
            with open(os.path.join(DirConfig.MODEL_DIR, "TELA", "stable_scaler.pkl"), "rb") as f:
                stable_scaler = pickle.load(f)
            with open(os.path.join(DirConfig.MODEL_DIR, "TELA", "burst_scaler.pkl"), "rb") as f:
                burst_scaler = pickle.load(f)
            with open(os.path.join(DirConfig.MODEL_DIR, "TELA", "peak_bandwidth_model.pkl"), "rb") as f:
                peak_bandwidth_model = pickle.load(f)
            with open(os.path.join(DirConfig.MODEL_DIR, "TELA", "avg_bandwidth_model.pkl"), "rb") as f:
                avg_bandwidth_model = pickle.load(f)
            return type_classifier, cluster_stable, cluster_burst, classify_stable, classify_burst, stable_scaler, burst_scaler, peak_bandwidth_model, avg_bandwidth_model

        except FileNotFoundError:
            self.train_models()
            return self.load_item_predict_models()
        except Exception as e:
            logger.error(f"TELA模型加载失败: {e}")
            return None
