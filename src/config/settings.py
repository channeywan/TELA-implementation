import os
import numpy as np


class DirConfig:

    # 项目根目录
    PROJECT_ROOT = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))

    # 数据目录
    TRACE_ROOT = "/data/Tencent_CVD/Shanghai"
    CLUSTER_INFO_ROOT = "/home/wcl/cluster_info"

    # 输出目录
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
    VISUALIZATION_TRACE_DIR = os.path.join(OUTPUT_DIR, "visualization_trace")
    PEAK_DISTRIBUTION_DIR = os.path.join(OUTPUT_DIR, "peak_distribution")
    STATISTIC_DIR = os.path.join(OUTPUT_DIR, "statistic")
    PLACEMENT_DIR = os.path.join(OUTPUT_DIR, "placement")
    ODA_DIR = os.path.join(PLACEMENT_DIR, "ODA")
    SCDA_DIR = os.path.join(PLACEMENT_DIR, "SCDA")
    TELA_DIR = os.path.join(PLACEMENT_DIR, "Tela")
    # 模型保存目录
    MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

    # 确保目录存在
    @classmethod
    def ensure_dirs(cls):
        """确保必要的目录存在"""
        for dir_path in [cls.OUTPUT_DIR, cls.VISUALIZATION_TRACE_DIR, cls.PEAK_DISTRIBUTION_DIR, cls.ODA_DIR, cls.SCDA_DIR, cls.TELA_DIR, cls.MODEL_DIR]:
            os.makedirs(dir_path, exist_ok=True)


class DataConfig:

    # 选择分析的集群列表
    CLUSTER_INDEX_LIST_ODA = [0, 2, 4]
    CLUSTER_INDEX_LIST_TRAIN = [0, 2, 4]
    CLUSTER_INDEX_LIST_PREDICT = [1, 3]

    # 评价时间数量
    EVALUATE_TIME_NUMBER = 8640

    # 云盘放置数量
    DISK_NUMBER = 8000

    # 云盘最小生命周期（时间戳）
    MIN_TIMESTAMP_NUM = 8640

    # 时序间隔
    TIME_INTERVAL = 1

    # 时间窗口长度(小时)for coach
    TIME_WINDOW_LENGTH = 4

    # 检测的时间窗口大小for tela
    VIOLATION_WINDOW_SIZE = 1000

    # 一个时间窗口内发生违反SLA的次数，超过即满载
    MAX_VIOLATION_OCCURRENCE = 10

    # 监视器为防止违反SLA的预留比例
    RESERVATION_RATE_FOR_MONITOR = 0.8


class ModelConfig:
    """模型相关配置"""

    # 阈值设置
    IO_LINE = 100
    BANDWIDTH_LINE = 50
    READ_BANDWIDTH_LINE = 9
    WRITE_BANDWIDTH_LINE = 30

    # 聚类参数
    SCDA_CLUSTER_K = 3
    TELA_CLUSTER_K = 5

    # 预留比例
    RESERVATION_RATE = 0.95

    EPISODES = 1

    # 监控预留率（用于SLA违反检测）
    RESERVATION_RATE_FOR_MONITOR = 0.8
    PEAK_PREDICTION_TOLERANCE_FACTOR = 1.3


class WarehouseConfig:
    # 仓库最大容量 (9个仓库配置)(MB)
    MAX_CAPACITY = [25000, 25000, 25000, 40000,
                    40000, 10000, 10000, 30000, 20000]

    # 仓库最大read bandwidth
    MAX_READ_BANDWIDTH = [30000, 30000, 30000, 13000,
                          13000, 13000, 47000, 47000, 47000]

    # 仓库最大write bandwidth
    MAX_WRITE_BANDWIDTH = [70000, 70000, 70000, 130000,
                           130000, 130000, 20000, 20000, 20000]

    # 仓库最大值矩阵
    WAREHOUSE_MAX = np.array(
        [MAX_CAPACITY, MAX_READ_BANDWIDTH, MAX_WRITE_BANDWIDTH])

    # 仓库数量
    WAREHOUSE_NUMBER = 9
    CLUSTER_NUMBER = 10


class TestConfig:
    TEST_DIR = os.path.join(DirConfig.PLACEMENT_DIR, "test")
