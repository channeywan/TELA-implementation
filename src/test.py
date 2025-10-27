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


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # 确保目录存在
    placed_warehouses = np.loadtxt(os.path.join(
        DirConfig.SCDA_DIR, 'placed_warehouses'), delimiter=',', dtype=int)
