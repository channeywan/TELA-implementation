from visualization.plotter import TelaPlotter,EvaluationPlotter
from sklearn.metrics import root_mean_squared_error as rmse, r2_score
import os
import pandas as pd
import numpy as np
import sys
import logging
from pathlib import Path
from config.settings import DirConfig, DataConfig, WarehouseConfig, TestConfig
from data.loader import DiskDataLoader
from algorithms.TIDAL import TIDAL
from algorithms.TELA import TELA
from algorithms.SCDA import SCDA
from algorithms.ODA import ODA
from algorithms.Oracle import Oracle
from algorithms.round_robin import RoundRobin
from sklearn.model_selection import train_test_split
from scipy.stats import gaussian_kde
from scipy.optimize import brentq
from typing import List
import pickle
sys.path.insert(0, str(Path(__file__).parent.parent))
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
def softpercentile(data: List[float], percentile: float) -> float:
        kde=gaussian_kde(data)
        threshold=brentq(lambda x: kde.integrate_box_1d(-np.inf, x) - percentile, 0, 2016)
        return threshold
def run_all_algorithms(train_items, test_items):
    algorithm_name_list = ["ODA", "TELA",
                           "SCDA", "Oracle", "TIDAL", "RoundRobin"]
    for algorithm_name in algorithm_name_list:
        analyzer = getattr(sys.modules[__name__], algorithm_name)()
        analyzer.place_items_number_list = place_items_number_list
        analyzer.train_items = train_items
        analyzer.test_items = test_items
        analyzer.run()
def diff_tau(train_items, test_items):
    tau_list = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    for tau in tau_list:
        tidal_analyzer = TIDAL()
        tidal_analyzer.unknown_threshold = tau
        tidal_analyzer.train_items = train_items
        tidal_analyzer.test_items = test_items
        tidal_analyzer.run()
def diff_K(train_items, test_items):
    K_list = [3,6,12,24]
    for K in K_list:
        tidal_analyzer = TIDAL()
        tidal_analyzer.trace_vector_interval = K
        tidal_analyzer.warehouse_trace_vector = np.zeros(
            (int(24/K), WarehouseConfig.WAREHOUSE_NUMBER))
        tidal_analyzer.train_items = train_items
        tidal_analyzer.test_items = test_items
        tidal_analyzer.run()
def diff_M(train_items, test_items):
    M_list = [2, 3, 4, 5, 6]
    for M in M_list:
        tidal_analyzer = TIDAL()
        tidal_analyzer.min_bandwidth_warehouse_num = M
        tidal_analyzer.train_items = train_items
        tidal_analyzer.test_items = test_items
        tidal_analyzer.run()
def diff_target(train_items, test_items):
    target_list = ["min_delta_util_var", "min_util_var", "min_load_var", "min_delta_load_var", "min_peak_load_before_placement","min_peak_load_after_placement"]
    for target in target_list:
        tidal_analyzer = TIDAL()
        tidal_analyzer.target = target
        tidal_analyzer.train_items = train_items
        tidal_analyzer.test_items = test_items
        tidal_analyzer.run()
def find_best_random_state_with_noise_ratio(train_items, test_items):
    otf=[]
    Spatial_Imbalance=[]
    Temporal_Imbalance=[]
    Mean_Duration=[]
    for random_state in range(50):
        epoch_otf=[]
        epoch_Spatial_Imbalance=[]
        epoch_Temporal_Imbalance=[]
        epoch_Mean_Duration=[]
        noise_ratio_list = [0,0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        for noise_ratio in noise_ratio_list:
            tidal_analyzer = TIDAL()
            tidal_analyzer.random_state = random_state
            tidal_analyzer.noise_ratio = noise_ratio
            tidal_analyzer.train_items = train_items
            tidal_analyzer.test_items = test_items
            tidal_analyzer.run()
            epoch_otf.append(tidal_analyzer.overload_percentage)
            epoch_Spatial_Imbalance.append(tidal_analyzer.space_imbalance)
            epoch_Temporal_Imbalance.append(tidal_analyzer.time_imbalance)
            epoch_Mean_Duration.append(np.mean(tidal_analyzer.overload_duration))
        otf.append(epoch_otf)
        Spatial_Imbalance.append(epoch_Spatial_Imbalance)
        Temporal_Imbalance.append(epoch_Temporal_Imbalance)
        Mean_Duration.append(epoch_Mean_Duration)
    otf=np.array(otf)
    Spatial_Imbalance=np.array(Spatial_Imbalance)
    Temporal_Imbalance=np.array(Temporal_Imbalance)
    Mean_Duration=np.array(Mean_Duration)
    mean_otf=np.mean(otf,axis=0)
    mean_Spatial_Imbalance=np.mean(Spatial_Imbalance,axis=0)
    mean_Temporal_Imbalance=np.mean(Temporal_Imbalance,axis=0)
    mean_Mean_Duration=np.mean(Mean_Duration,axis=0)
    EvaluationPlotter().plot_figure_11_test(save_dir=DirConfig.EVALUATION_DIR,otf=mean_otf,time_imbalance=mean_Temporal_Imbalance,space_imbalance=mean_Spatial_Imbalance,mean_duration=mean_Mean_Duration)
    result={}
    result["otf"]=otf
    result["Spatial_Imbalance"]=Spatial_Imbalance
    result["Temporal_Imbalance"]=Temporal_Imbalance
    result["Mean_Duration"]=Mean_Duration
    result["random_state"]=range(50)
    with open("diff_random_state_noise_ratio_result.pkl","wb") as f:
        pickle.dump(result,f)
def find_suitable_config_on_tela(train_items, test_items):
    for violation_window_size in [500,600,700,800,900,1000,1200,1400,1600,1800,2000,2200,2400,2600,2800,3000,3200,3400,3600,3800,4000,4200,4400,4600,4800,5000,5200,5400,5600,5800,6000]:
        for max_violation_occurrence in range(10,violation_window_size-10,10):
            tela_analyzer = TELA()
            tela_analyzer.train_items = train_items
            tela_analyzer.test_items = test_items
            tela_analyzer.violation_window_size = violation_window_size
            tela_analyzer.max_violation_occurrence = max_violation_occurrence
            tela_analyzer.place_items_number_list = place_items_number_list
            tela_analyzer.run()
            EvaluationPlotter().plot_figure_2_test(save_dir=os.path.join(DirConfig.EVALUATION_DIR,"testfigure2"),tela_overload_duration=tela_analyzer.overload_duration,violation_count=tela_analyzer.violation_count,violation_window_size=violation_window_size,max_violation_occurrence=max_violation_occurrence)

if __name__ == "__main__":
    place_items_number_list = [5200, 5400, 5500,
                               5600, 5700, 5800, 5900, 6000, 6048]
    selected_items = DiskDataLoader().load_selected_items()
    train_items, test_items = train_test_split(
        selected_items, test_size=0.3, random_state=42)


   
    