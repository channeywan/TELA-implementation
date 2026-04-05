import os
import numpy as np


class DirConfig:

    # 项目根目录
    PROJECT_ROOT = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))

    # 数据目录
    TRACE_ROOT = "/data/Tencent_CVD/workload/0521"
    CLUSTER_INFO_ROOT = "/data/tidal_info/cluster_info"
    CLUSTER_INFO_BUSINESS_TYPE_ROOT = "/data/tidal_info/history_file/cluster_info_business_type"
    CLUSTER_TRACE_DB_ROOT = "/data/tidal_info/cluster_trace_db"
    # 输出目录
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
    VISUALIZATION_TRACE_DIR = os.path.join(OUTPUT_DIR, "visualization_trace")
    PEAK_DISTRIBUTION_DIR = os.path.join(OUTPUT_DIR, "peak_distribution")
    STATISTIC_DIR = os.path.join(OUTPUT_DIR, "statistic")
    PLACEMENT_DIR = os.path.join(OUTPUT_DIR, "placement")
    ODA_DIR = os.path.join(PLACEMENT_DIR, "ODA")
    SCDA_DIR = os.path.join(PLACEMENT_DIR, "SCDA")
    TELA_DIR = os.path.join(PLACEMENT_DIR, "Tela")
    Oracle_DIR = os.path.join(PLACEMENT_DIR, "Oracle")
    TIDAL_DIR = os.path.join(PLACEMENT_DIR, "TIDAL")
    RoundRobin_DIR = os.path.join(PLACEMENT_DIR, "RoundRobin")
    MODEL_DISTILL_DIR = os.path.join(PROJECT_ROOT, "model_distill")
    BUSINESS_TYPE_DIR = os.path.join(PROJECT_ROOT, "business_type")
    # 模型保存目录
    MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
    TEMP_DIR = os.path.join(OUTPUT_DIR, "temp")
    TRASH_DIR = os.path.join(TEMP_DIR, "trash")
    INTERMEDIATE_DIR = os.path.join(TEMP_DIR, "intermediate")
    MOTIVATION_DIR = os.path.join(OUTPUT_DIR, "motivation")
    EVALUATION_DIR = os.path.join(OUTPUT_DIR, "evaluation")
    # 确保目录存在

    @classmethod
    def ensure_dirs(cls):
        """确保必要的目录存在"""
        for dir_path in [cls.OUTPUT_DIR, cls.VISUALIZATION_TRACE_DIR, cls.PEAK_DISTRIBUTION_DIR, cls.ODA_DIR, cls.SCDA_DIR, cls.TELA_DIR, cls.Oracle_DIR, cls.MODEL_DIR]:
            os.makedirs(dir_path, exist_ok=True)


class DataConfig:
    # 选择分析的集群列表
    # 400, 401, 405, 406, 407, 408, 413, 414, 415, 436, 437, 452, 456, 458, 476, 477, 478, 486, 487, 511, 517, 518, 519,
    # 520, 521, 529, 532, 538, 546, 547, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 576, 577, 578, 588,
    # 589, 602, 604, 605, 606, 609, 610, 650, 652, 653, 656, 662, 674, 676, 677, 685, 686, 697, 698, 700
    CLUSTER_INDEX_LIST_ODA = [407, 408, 415, 436, 437, 456, 458,
                              511, 517, 518, 520, 538, 546, 560, 564, 565, 566, 567, 577, 588]
    CLUSTER_INDEX_LIST_TRAIN = [400, 401, 405, 406,
                                413, 414, 452, 476, 477, 478, 486, 487, 519]
    CLUSTER_INDEX_LIST_PREDICT = [407, 408, 415, 436, 437, 456, 458,
                                  511, 517, 518, 520, 538, 546, 560, 564, 565, 566, 567, 577, 588]
    CLUSTER_INDEX_LIST_ORACLE = [407, 408, 415, 436, 437, 456, 458,
                                 511, 517, 518, 520, 538, 546, 560, 564, 565, 566, 567, 577, 588]
    REQUEST_BUSINESS_TYPE_CLUSTER_INDEX_LIST = [400, 401, 405, 406, 413, 414, 452, 476, 477, 478, 486, 487, 519, 521, 529, 532, 547, 561, 562, 563, 568, 569, 570, 571, 576,
                                                578, 589, 602, 604, 605, 606, 609, 610, 650, 652, 653, 656, 662, 674, 676, 677, 685, 686, 697, 698, 700, 704, 709, 727, 735, 736, 742, 748, 752, 753, 754, 756, 757, 771]
    SELECT_CLUSTER_INDEX_LIST = [407, 408, 415, 436, 437, 456, 458,
                                 511, 517, 518, 520, 538, 546, 560, 564, 565, 566, 567, 577, 588]
    CLUSTER_DIR_LIST = [400, 401, 405, 406, 407, 408, 413, 414, 415, 436, 437, 452, 456, 458, 476, 477, 478, 486, 487, 511, 517, 518, 519,
                        520, 521, 529, 532, 538, 546, 547, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 576, 577, 578, 588,
                        589, 602, 604, 605, 606, 609, 610, 650, 652, 653, 656, 662, 674, 676, 677, 685, 686, 697, 698, 700, 704, 709, 727, 735, 736,
                        742, 748, 752, 753, 754, 756, 757, 771]

    # 评价时间数量
    EVALUATE_TIME_NUMBER = 2016
    MAX_BANDWIDTH_THRESHOLD = 1e7
    # 云盘放置数量
    DISK_NUMBER = 6048

    # 云盘最小生命周期（时间戳）
    MIN_TIMESTAMP_NUM = 2016

    # 时序间隔
    TIME_INTERVAL = 1

    # 时间窗口长度(小时)for coach
    TIME_WINDOW_LENGTH = 4

    # 检测的时间窗口大小for tela
    VIOLATION_WINDOW_SIZE = 4500

    # 一个时间窗口内发生违反SLA的次数，超过即满载
    MAX_VIOLATION_OCCURRENCE = 2000
    MAX_OVERLOAD_LIFETIME_OCCURRENCE = 150
    RESERVATION_RATE_FOR_MONITOR = 0.8
    WINDOWS_LENGTH_IN_ONE_DAY = [
        "5min",  "30min", "1h","2h" ,"3h", "4h", "6h", "8h"]
    TRACE_VECTOR_INTERVAL = 6
    BUSINESS_TYPE_LIST = ["game-service", "office-system", "gov-public-service", "corp-website-portal", "ecommerce-retail", "local-service-delivery", "media-video-streaming", "media-news-portal", "finance-payment", "data-collection-delivery", "ai-machine-learning", "dev-test-env", "education-learning", "community-social-forum",
                          "compute-simulation", "personal-use", "iot-saas-platform", "logistics-mobility", "travel-hospitality", "infra-node", "infra-coordination", "infra-database", "infra-message-queue", "infra-cloud-function", "infra-jumpbox", "infra-cache", "infra-logging-monitoring", "generic-autoscaling", "generic-unknown"]


class ModelConfig:
    """模型相关配置"""

    # 阈值设置
    IO_LINE = 100
    BANDWIDTH_LINE = 25

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
    # capacity_mean = 7323193/6
    # bandwidth_mean = 4181941923/6
    capacity_mean = 1383882/6
    bandwidth_mean = 192201738/6
    # CAPACITY_RATIO = np.array([1.1, 1.1, 1, 1, 1, 1])
    # BANDWIDTH_RATIO = np.array([0.8, 1, 1.2, 1, 1.2, 1.4])

    CAPACITY_RATIO = np.array([1.3, 1.3, 1.3, 1.3, 1.3, 1.3])
    BANDWIDTH_RATIO = np.array([1.2, 1.2, 1.2, 1.2, 1.2, 1.2])
    # MAX_CAPACITY = np.array(
    #     [1.1, 1.1, 1, 1, 1, 1])*capacity_mean
    # MAX_BANDWIDTH = np.array(
    #     [1, 1, 1.2, 1.2, 1.4, 1.4])*bandwidth_mean

    # 仓库最大值矩阵
    WAREHOUSE_MAX = np.array(
        [CAPACITY_RATIO*capacity_mean, BANDWIDTH_RATIO*bandwidth_mean]).T

    # 仓库数量
    WAREHOUSE_NUMBER = 6
    CLUSTER_NUMBER = 10


class TestConfig:
    TEST_DIR = os.path.join(DirConfig.PLACEMENT_DIR, "test")
