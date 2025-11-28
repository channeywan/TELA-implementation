from visualization.plotter import TelaPlotter
from sklearn.metrics import root_mean_squared_error as rmse, r2_score
import os
import pandas as pd
import numpy as np
import sys
import logging
from pathlib import Path
from config.settings import DirConfig, DataConfig, WarehouseConfig, TestConfig
from data.loader import DiskDataLoader
from itertools import permutations
from sklearn.model_selection import train_test_split
sys.path.insert(0, str(Path(__file__).parent))
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
if __name__ == "__main__":
    leave_disk_info = pd.read_csv(os.path.join(
        DirConfig.BUSINESS_TYPE_DIR, "leave_disk_info_business_type.csv"))
    leave_disk_info = leave_disk_info[["description", "business_type"]]
    remain_disk_info = pd.read_csv(os.path.join(
        DirConfig.BUSINESS_TYPE_DIR, "remain_disks_valid_business_type.csv"))
    remain_disk_info = remain_disk_info[["description", "business_type"]]
    remain_disk_workload = pd.concat([leave_disk_info, remain_disk_info])
    print(remain_disk_workload["business_type"].value_counts())
