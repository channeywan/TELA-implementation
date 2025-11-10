import os
import logging
import numpy as np
import csv
from typing import List, Dict, Any, Optional
import pandas as pd

from config.settings import DirConfig, DataConfig, WarehouseConfig, ModelConfig

logger = logging.getLogger(__name__)


class DiskDataLoader:
    """磁盘数据加载器"""

    def __init__(self):
        pass

    def load_items(self, cluster_index_list: List[int] = range(WarehouseConfig.WAREHOUSE_NUMBER), type: str = "both", purpose: str = "train") -> pd.DataFrame:
        ORIGINAL_COLS = ["disk_ID", "disk_capacity", "disk_if_local", "disk_attr", "disk_type", "disk_if_VIP",
                         "disk_pay", "vm_cpu", "vm_mem", "avg_bandwidth", "peak_bandwidth", "timestamp_num", "burst_label", "bandwidth_mul", "bandwidth_zero_ratio"]
        PREDICT_COLS = ["disk_ID", "disk_capacity", "disk_if_local", "disk_attr", "disk_type", "disk_if_VIP",
                        "disk_pay", "vm_cpu", "vm_mem"]
        logger.info(f"正在加载{type}磁盘数据,用途为{purpose}")
        all_cluster_items = []
        for cluster_index in cluster_index_list:
            item_dir = os.path.join(
                DirConfig.CLUSTER_INFO_ROOT,
                f"cluster{cluster_index}"
            )
            if not os.path.exists(item_dir):
                logger.warning(f"数据文件不存在: {item_dir}")
                continue
            try:
                items = pd.read_csv(item_dir, header=None,
                                    names=ORIGINAL_COLS, encoding="utf-8")
                items["cluster_index"] = cluster_index
                cluster_items = self._parse_disk(
                    items, type)
                if cluster_items is not None and not cluster_items.empty:
                    all_cluster_items.append(cluster_items)
            except (IOError, OSError) as e:
                logger.error(f"读取文件 {item_dir} 出错: {e}")
                continue
            logger.info(f"集群 {cluster_index} 加载了 {len(cluster_items)} 个磁盘")
        all_items = pd.concat(all_cluster_items, ignore_index=True)
        if purpose == "predict":
            all_items = all_items[PREDICT_COLS]
        return all_items

    def _parse_disk(self, items: pd.DataFrame, type: str = "both") -> Optional[pd.DataFrame]:
        """
        解析磁盘项目数据
        """
        type_map = {'disk_capacity': int, 'disk_if_local': int, 'disk_if_VIP': int,
                    'vm_cpu': float, 'vm_mem': float, 'avg_bandwidth': float,
                    'peak_bandwidth': float,
                    'timestamp_num': int, 'burst_label': int,
                    'cluster_index': int, 'bandwidth_mul': float, 'bandwidth_zero_ratio': float}

        items = items.astype(type_map)
        max_bandwidth = np.min(WarehouseConfig.MAX_BANDWIDTH) * \
            ModelConfig.RESERVATION_RATE_FOR_MONITOR
        mask_cpu_mem = (items['vm_cpu'] > 0) & (items['vm_mem'] > 0)
        mask_timestamp = items['timestamp_num'] >= DataConfig.MIN_TIMESTAMP_NUM
        mask_avg_bw = (items['avg_bandwidth'] > 0)
        mask_bandwidth_zero_ratio = (items['bandwidth_zero_ratio'] < 0.4)
        validation_mask = mask_cpu_mem & mask_timestamp & mask_avg_bw & mask_bandwidth_zero_ratio
        if type == "burst":
            type_mask = (items['burst_label'] == 1)
        elif type == "stable":
            type_mask = (items['burst_label'] == 0)
        elif type == "both":
            type_mask = pd.Series(True, index=items.index)
        else:
            raise ValueError(f"Invalid type: {type}")
        items = items[validation_mask & type_mask]
        return items
