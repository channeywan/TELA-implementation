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
    loader = DiskDataLoader()
    items = loader.load_items(DataConfig.CLUSTER_DIR_LIST)
    items_history = items[["disk_ID", "cluster_index", "recent_history_bandwidth_memory",
                           "recent_history_bandwidth_cpu", "recent_history_bandwidth_app", "recent_history_bandwidth_cpu_memory"]]
    df = pd.read_csv(os.path.join(DirConfig.CLUSTER_INFO_ROOT,
                     "selected_items_with_business_type.csv"))
    df = pd.merge(df, items_history, on=[
                  "disk_ID", "cluster_index"], how="left")
    df.to_csv(os.path.join(DirConfig.CLUSTER_INFO_ROOT,
              "selected_items_with_business_type.csv"), index=False)
