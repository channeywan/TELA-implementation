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
        '''
        train:
        [0:1]:disk_ID,disk_capacity
        [2:7]:disk_if_local, disk_attr, disk_type, disk_if_VIP, disk_pay,
        [7:13]:vm_cpu, vm_mem, avg_rbw, avg_wbw, peak_rbw, peak_wbw,timestamp_num, burst_label
        [15]:cluster_index

        predict:
        [0:1]:disk_ID,disk_capacity
        [2:7]:disk_if_local, disk_attr, disk_type, disk_if_VIP, disk_pay,
        [7:10]:vm_cpu, vm_mem,cluster_index
        '''
        ORIGINAL_COLS = ["disk_ID", "disk_capacity", "disk_if_local", "disk_attr", "disk_type", "disk_if_VIP",
                         "disk_pay", "vm_cpu", "vm_mem", "avg_rbw", "avg_wbw", "peak_rbw", "peak_wbw", "timestamp_num", "burst_label"]
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
                    items, cluster_index, purpose, type)
                if cluster_items is not None and not cluster_items.empty:
                    all_cluster_items.append(cluster_items)
            except (IOError, OSError) as e:
                logger.error(f"读取文件 {item_dir} 出错: {e}")
                continue
            logger.info(f"集群 {cluster_index} 加载了 {len(cluster_items)} 个突发性磁盘")
        all_items = pd.concat(all_cluster_items, ignore_index=True)
        if purpose == "predict":
            all_items = all_items[PREDICT_COLS]
        return all_items

    def _parse_disk(self, items: pd.DataFrame, cluster_index: int, purpose: str = "train", type: str = "both") -> Optional[pd.DataFrame]:
        """
        解析磁盘项目数据
        """
        type_map = {'disk_capacity': int, 'disk_if_local': int, 'disk_if_VIP': int,
                    'vm_cpu': float, 'vm_mem': float, 'avg_rbw': float,
                    'avg_wbw': float, 'peak_rbw': float, 'peak_wbw': float,
                    'timestamp_num': int, 'burst_label': int,
                    'cluster_index': int}

        items = items.astype(type_map)
        max_read_bandwidth = WarehouseConfig.MAX_READ_BANDWIDTH[cluster_index] * \
            ModelConfig.RESERVATION_RATE_FOR_MONITOR
        max_write_bandwidth = WarehouseConfig.MAX_WRITE_BANDWIDTH[cluster_index] * \
            ModelConfig.RESERVATION_RATE_FOR_MONITOR
        mask_cpu_mem = (items['vm_cpu'] > 0) & (items['vm_mem'] > 0)
        mask_timestamp = items['timestamp_num'] >= DataConfig.MIN_TIMESTAMP_NUM
        mask_bandwidth = (items['peak_rbw'] <= max_read_bandwidth) & (
            items['peak_wbw'] <= max_write_bandwidth)
        mask_avg_bw = (items['avg_rbw'] >= 3) & (items['avg_wbw'] >= 3)
        validation_mask = mask_cpu_mem & mask_timestamp & mask_bandwidth & mask_avg_bw
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
