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

logger = logging.getLogger(__name__)


class TELA(BaseAlgorithm):
    """TELA算法实现 - 基于时间序列负载均衡的算法"""

    def __init__(self):
        super().__init__("TELA")
        self.peak_RBW_model = None
        self.peak_WBW_model = None

    def load_and_preprocess_items(self):
        """加载和预处理数据"""
        # 加载预测数据
        items = self.loader.load_items(
            type="both",
            cluster_index_list=DataConfig.CLUSTER_INDEX_LIST_PREDICT,
            purpose="train"
        )
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
        items_train_encoded = self.encode_item(items_train)

        # 第一阶段：训练稳定型/突发型分类模型
        model_classify_type = self.train_type_classifier(
            items_train_encoded)

        # 第二阶段：训练细分聚类和分类模型
        cluster_stable, cluster_burst, classify_stable, classify_burst, stable_scaler, burst_scaler = self.train_cluster_classifier(
            items_train_encoded)

        # 训练负载拟合模型
        simulate_load, placed_warehouses = self.simulate_warehouse_load(
            items_train_encoded[items_train_encoded["burst_label"] == 1])
        self.peak_RBW_model, self.peak_WBW_model = self.train_warehouse_load_model(
            simulate_load, placed_warehouses, cluster_burst.labels_)

        # 保存所有模型
        self.save_models(model_classify_type, cluster_stable, cluster_burst,
                         classify_stable, classify_burst,  stable_scaler, burst_scaler)

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
        stable_features = items_stable[["avg_rbw", "avg_wbw"]]
        burst_features = items_burst[["peak_rbw", "peak_wbw"]]
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

        model_classify_stable = GridSearchCV(
            estimator=DecisionTreeClassifier(criterion="gini"),
            param_grid=params_grid, cv=5, scoring='accuracy', n_jobs=1, verbose=0
        )
        model_classify_burst = GridSearchCV(
            estimator=DecisionTreeClassifier(criterion="gini"),
            param_grid=params_grid, cv=5, scoring='accuracy', n_jobs=1, verbose=0
        )
        model_classify_stable.fit(items_stable[[
                                  "disk_capacity", "disk_type", "disk_if_VIP", "vm_cpu", "vm_mem"]], items_stable["burst_label"])
        model_classify_burst.fit(items_burst[[
                                 "disk_capacity", "disk_type", "disk_if_VIP", "vm_cpu", "vm_mem"]], items_burst["burst_label"])
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

    def train_warehouse_load_model(self, simulate_load: np.ndarray, placed_warehouses: np.ndarray, cluster_labels: np.ndarray):
        """训练负载拟合模型"""
        disk_number = len(simulate_load)
        logger.info("开始训练负载拟合模型")
        cluster_disk_count = np.zeros(
            (disk_number, WarehouseConfig.WAREHOUSE_NUMBER, ModelConfig.TELA_CLUSTER_K), dtype=int)

        for i in range(disk_number):
            cluster_disk_count[i, placed_warehouses[i], cluster_labels[i]] += 1
        X_train = cluster_disk_count[np.arange(
            disk_number), placed_warehouses, :]
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
        peak_RBW_model = Earth(max_degree=1)
        peak_WBW_model = Earth(max_degree=1)
        peak_RBW_model.fit(X_train, simulate_load[:, 0])
        peak_WBW_model.fit(X_train, simulate_load[:, 1])
        return peak_RBW_model, peak_WBW_model

    def simulate_warehouse_load(self, items: pd.DataFrame):
        """
        simulate_load_trace: 模拟负载的时序数据
        placed_warehouses: 磁盘放置的仓库
        warehouse_load:增加每个磁盘仓库的负载峰值和均值指标[len(items),(RBW_max,WBW_max,RBW_avg,WBW_avg)]
        """
        oda = ODA()
        _, placed_warehouses = oda.place_item(items)
        simulate_load_trace = np.zeros(
            (WarehouseConfig.WAREHOUSE_NUMBER, 2, DataConfig.EVALUATE_TIME_NUMBER))
        warehouse_load = np.zeros((len(items), 4))

        for i, item in items.iterrows():
            # 跳过未成功放置的磁盘
            if i >= len(placed_warehouses) or placed_warehouses[i] == -1:
                continue

            warehouse_idx = placed_warehouses[i]
            cluster_index = item["cluster_index"]
            disk_ID = item["disk_ID"]
            trace_dir = os.path.join(DirConfig.TRACE_ROOT,
                                     f"20_136090{cluster_index}", f"{disk_ID}")
            disk_trace_bandwidth = np.loadtxt(
                trace_dir, delimiter=',', usecols=(2, 4), dtype=int)
            read_bandwidth_trace = disk_trace_bandwidth[:
                                                        DataConfig.EVALUATE_TIME_NUMBER, 0]
            write_bandwidth_trace = disk_trace_bandwidth[:
                                                         DataConfig.EVALUATE_TIME_NUMBER, 1]
            simulate_load_trace[warehouse_idx,
                                0, :] += read_bandwidth_trace
            simulate_load_trace[warehouse_idx,
                                1, :] += write_bandwidth_trace
            warehouse_load[i, 0] = np.max(
                simulate_load_trace[warehouse_idx, 0, :])
            warehouse_load[i, 1] = np.max(
                simulate_load_trace[warehouse_idx, 1, :])
            warehouse_load[i, 2] = np.mean(
                simulate_load_trace[warehouse_idx, 0, :])
            warehouse_load[i, 3] = np.mean(
                simulate_load_trace[warehouse_idx, 1, :])
        items.to_csv(os.path.join(
            DirConfig.TELA_DIR, "train_items.csv"))
        return warehouse_load, placed_warehouses

    def predict_disk_types_and_loads(self, items: pd.DataFrame):
        """预测磁盘类型和负载"""
        # 加载训练好的模型
        type_classifier, cluster_stable, cluster_burst, classify_stable, classify_burst, stable_scaler, burst_scaler = self.load_item_predict_models()

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
        burst_items_belong_to_cluster_centers_df = pd.DataFrame(
            burst_items_belong_to_cluster_centers, columns=["pre_rbw", "pre_wbw"])
        stable_items_belong_to_cluster_centers_df = pd.DataFrame(
            stable_items_belong_to_cluster_centers, columns=["pre_rbw", "pre_wbw"])
        stable_items.to_csv(os.path.join(
            DirConfig.TELA_DIR, "stable_items.csv"))
        stable_items_belong_to_cluster_centers_df.to_csv(os.path.join(
            DirConfig.TELA_DIR, "stable_items_belong_to_cluster_centers_df.csv"))
        burst_items_with_cluster_centers = pd.concat(
            [burst_items, burst_items_belong_to_cluster_centers_df], axis=1)
        stable_items_with_cluster_centers = pd.concat(
            [stable_items, stable_items_belong_to_cluster_centers_df], axis=1)
        items_with_cluster_centers = pd.concat(
            [burst_items_with_cluster_centers, stable_items_with_cluster_centers], axis=0, ignore_index=True)
        items_with_cluster_centers.to_csv(os.path.join(
            DirConfig.TELA_DIR, "predict_items.csv"))
        logger.info("预测磁盘类型和负载完成")
        return items_with_cluster_centers

    def select_warehouse(self, item: pd.Series, warehouses_resource_allocated: List[List[int]],
                         warehouses_cannot_use_by_monitor: np.ndarray, warehouse_burst_type_counts: np.ndarray) -> int:
        """TELA的仓库选择策略：基于时间序列预测的负载均衡"""
        selected_warehouse = -1
        scda = SCDA()
        if item["pre_burst_label"] == 0:
            selected_warehouse = scda.select_warehouse(
                item, warehouses_resource_allocated, warehouses_cannot_use_by_monitor)
        else:
            min_peak_resource_usage = np.inf
            for warehouse in range(WarehouseConfig.WAREHOUSE_NUMBER):
                if warehouses_cannot_use_by_monitor[warehouse] == 1:
                    continue
                if warehouses_resource_allocated[warehouse][0] + item["disk_capacity"] > WarehouseConfig.WAREHOUSE_MAX[0][warehouse] * ModelConfig.RESERVATION_RATE_FOR_MONITOR:
                    continue
                train_X = warehouse_burst_type_counts[warehouse]+np.array(
                    [1 if i == item["belong_to_cluster"] else 0 for i in range(ModelConfig.TELA_CLUSTER_K)])
                train_X = np.array([train_X])
                predict_warehouse_peak_RBW = self.peak_RBW_model.predict(
                    train_X)
                predict_warehouse_peak_WBW = self.peak_WBW_model.predict(
                    train_X)
                if predict_warehouse_peak_RBW+warehouses_resource_allocated[warehouse][1] > WarehouseConfig.WAREHOUSE_MAX[1][warehouse] * ModelConfig.PEAK_PREDICTION_TOLERANCE_FACTOR or predict_warehouse_peak_WBW+warehouses_resource_allocated[warehouse][2] > WarehouseConfig.WAREHOUSE_MAX[2][warehouse] * ModelConfig.PEAK_PREDICTION_TOLERANCE_FACTOR:
                    continue
                peak_resource_usage = predict_warehouse_peak_RBW / \
                    WarehouseConfig.WAREHOUSE_MAX[1][warehouse] + \
                    predict_warehouse_peak_WBW / \
                    WarehouseConfig.WAREHOUSE_MAX[2][warehouse]
                if peak_resource_usage < min_peak_resource_usage:
                    min_peak_resource_usage = peak_resource_usage
                    selected_warehouse = warehouse
        return selected_warehouse

    def additional_state_update(self, selected_warehouse: int, item: pd.Series, warehouses_resource_allocated: np.ndarray, warehouses_trace: np.ndarray, current_time: int, warehouse_burst_type_counts: np.ndarray):
        if item["pre_burst_label"] == 0:
            warehouses_resource_allocated[selected_warehouse][1] += item["pre_rbw"]
            warehouses_resource_allocated[selected_warehouse][2] += item["pre_wbw"]
        else:
            warehouse_burst_type_counts[selected_warehouse][item["belong_to_cluster"]] += 1
        return

    def initialize_additional_state(self):
        warehouse_burst_type_counts = np.zeros(
            (WarehouseConfig.WAREHOUSE_NUMBER, ModelConfig.TELA_CLUSTER_K), dtype=int)
        return warehouse_burst_type_counts

    def save_models(self, type_classifier, cluster_stable, cluster_burst, classify_stable, classify_burst, stable_scaler, burst_scaler):
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
        with open(os.path.join(DirConfig.MODEL_DIR, "TELA", "peak_RBW_model.pkl"), "wb") as f:
            pickle.dump(self.peak_RBW_model, f)
        with open(os.path.join(DirConfig.MODEL_DIR, "TELA", "peak_WBW_model.pkl"), "wb") as f:
            pickle.dump(self.peak_WBW_model, f)

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
            with open(os.path.join(DirConfig.MODEL_DIR, "TELA", "peak_RBW_model.pkl"), "rb") as f:
                self.peak_RBW_model = pickle.load(f)
            with open(os.path.join(DirConfig.MODEL_DIR, "TELA", "peak_WBW_model.pkl"), "rb") as f:
                self.peak_WBW_model = pickle.load(f)
            logger.info("TELA模型加载成功")
            return type_classifier, cluster_stable, cluster_burst, classify_stable, classify_burst, stable_scaler, burst_scaler

        except FileNotFoundError:
            logger.warning("TELA模型文件未找到,开始训练")
            self.train_models()
            return self.load_item_predict_models()
        except Exception as e:
            logger.error(f"TELA模型加载失败: {e}")
            return None
