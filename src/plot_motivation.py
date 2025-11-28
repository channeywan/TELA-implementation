from data.utils import PeakValleyWindowsDistribution
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
from data.peak_hour_analyze import peak_hour_frequency
from visualization.plotter import MotivationPlotter
sys.path.insert(0, str(Path(__file__).parent))
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
if __name__ == "__main__":
    MotivationPlotter().plot_figure_2a(save_dir=DirConfig.MOTIVATION_DIR)
    MotivationPlotter().plot_figure_2b(save_dir=DirConfig.MOTIVATION_DIR)
