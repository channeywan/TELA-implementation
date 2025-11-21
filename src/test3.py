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
    plotter = TelaPlotter()
    plotter.plot_scatter([(0.09564457627678793+0.053437550831631285)/2, (0.12504237340863641+0.19014722964133898)/2, (0.0915686407016325+0.1726856698957383)/2, (0.03487939092014046+0.09669256904934487)/2],
                         [0.04960462225684755, 0.04807727739462747, 0.04841741902497335, 0.046515012581257946], save_dir=os.path.join(DirConfig.TEMPLE_DIR, "trash"), title="space-time-imbalance", xlabel="space imbalance", ylabel="time imbalance")
