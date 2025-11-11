import logging
from data.loader import DiskDataLoader
from config.settings import DirConfig, DataConfig, ModelConfig, TestConfig, WarehouseConfig
from visualization.plotter import TelaPlotter
from algorithms.TELA import TELA
import sys
import os
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from datetime import datetime
import matplotlib.pyplot as plt
# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# vm_type:['S5' 'S6' nan 'SA3' 'C6' 'SA2' 'TS5' 'MA2' 'IT5' 'MA3' 'GN7' 'GN10X' 'D3' 'C5' 'S4' 'M5' 'M6' 'GN10Xp' 'GN7vw' 'S5se' 'BC1']
# disk_uuid, vm_alias, vm_cpu, vm_mem, vm_type, is_vip, ins_type(cvm,eks),project_name,buss_name,
# disk_alias,disk_size,disk_type(root,data,tfs,tssd),volume_type(cbsssd,cbsBSSD,cbsPremiun,cbsHSSD)
if __name__ == "__main__":
    work_desc = os.path.join(DirConfig.TRACE_ROOT,
                             "153_10077752", "describe.csv")
    disks_info = pd.read_csv(work_desc, sep=',', index_col=0)
    print(disks_info["disk_status"].unique())
