import os
import logging
import numpy as np
from typing import List, Dict, Any, Optional
import pandas as pd

from config.settings import DirConfig, DataConfig, WarehouseConfig, ModelConfig

logger = logging.getLogger(__name__)


class DiskDataLoader:
    """磁盘数据加载器"""

    def __init__(self):
        pass

    def load_items(self, cluster_index_list: List[int] = WarehouseConfig.CLUSTER_DIR_LIST, type: str = "both") -> pd.DataFrame:
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
                logger.info(f"加载集群 {cluster_index} 的磁盘数据")
                items = pd.read_csv(item_dir,  encoding="utf-8")
                if len(items) == 0:
                    logger.warning(f"cluster {cluster_index} 没有数据")
                    continue
                items["cluster_index"] = cluster_index
                cluster_items = self._parse_disk(
                    items, type)
                if cluster_items is not None and not cluster_items.empty:
                    all_cluster_items.append(cluster_items)
            except (IOError, OSError) as e:
                logger.error(f"读取文件 {item_dir} 出错: {e}")
                continue
            logger.info(
                f"集群 {cluster_index} 加载了 {len(cluster_items) if cluster_items is not None else 0} 个磁盘")
        # 如果没有加载到任何数据，返回空 DataFrame
        if len(all_cluster_items) == 0:
            return pd.DataFrame()
        all_items = pd.concat(all_cluster_items, ignore_index=True)
        return all_items

    def _parse_disk(self, items: pd.DataFrame, type: str = "both") -> Optional[pd.DataFrame]:
        """
        解析磁盘项目数据
        """
        type_map = {'disk_capacity': int, 'disk_if_VIP': int,
                    'vm_cpu': float, 'vm_memory': float, 'avg_bandwidth': float,
                    'peak_bandwidth': float, 'burst_label': int,
                    'cluster_index': int, 'bandwidth_mul': float, 'bandwidth_zero_ratio': float}

        items = items.astype(type_map)
        max_bandwidth = np.min(WarehouseConfig.MAX_BANDWIDTH) * \
            ModelConfig.RESERVATION_RATE_FOR_MONITOR
        mask_cpu_mem = (items['vm_cpu'] > 0) & (items['vm_memory'] > 0)
        mask_avg_bw = (items['avg_bandwidth'] > 0)
        mask_bandwidth_zero_ratio = (items['bandwidth_zero_ratio'] < 0.4)
        validation_mask = mask_cpu_mem & mask_avg_bw & mask_bandwidth_zero_ratio
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
