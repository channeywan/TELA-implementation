import sys
from pathlib import Path
import logging
import pandas as pd
import numpy as np
from config.settings import DirConfig, DataConfig, WarehouseConfig, TestConfig, ModelConfig
from data.loader import DiskDataLoader
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
sys.path.insert(0, str(Path(__file__).parent))
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    cat_type_list = ["disk_if_VIP", "disk_type",
                     "volume_type", "business_type"]
    features = ["disk_capacity", "vm_cpu",
                "vm_memory", "recent_history_bandwidth_memory"]
    loader = DiskDataLoader()
    items = loader.load_items(DataConfig.CLUSTER_DIR_LIST)
    train_items, test_items = train_test_split(
        items, test_size=0.2, random_state=42, shuffle=True)
    encoder = OrdinalEncoder(
        handle_unknown='use_encoded_value', unknown_value=-1)
    X_train = train_items[features + cat_type_list]
    X_train[cat_type_list] = encoder.fit_transform(X_train[cat_type_list])
    X_test = test_items[features + cat_type_list]
    X_test[cat_type_list] = encoder.transform(X_test[cat_type_list])
    y_train = train_items["avg_bandwidth"]
    y_test = test_items["avg_bandwidth"]
    thresholds = np.percentile(y_train, [0, 50, 80, 95, 100])
    bins = [-np.inf, thresholds[1], thresholds[2], thresholds[3], np.inf]
    labels = [0, 1, 2, 3]

    y_train_class = pd.cut(
        y_train, bins=bins, labels=labels, include_lowest=True)
    y_test_class = pd.cut(
        y_test, bins=bins, labels=labels, include_lowest=True)

    print(
        f"训练集分类分布:\n{y_train_class.value_counts(normalize=True).sort_index()}")

    class_mapping_df = pd.DataFrame({'y': y_train, 'class': y_train_class})
    class_means = class_mapping_df.groupby('class')['y'].mean().to_dict()

    print(f"\n类别到数值的映射 (均值): {class_means}")
    clf = RandomForestClassifier(
        random_state=42
    )

    clf.fit(X_train, y_train_class)
    y_pred_class = clf.predict(X_test).flatten()
    f1_score = f1_score(y_test_class, y_pred_class, average='weighted')
    accuracy = accuracy_score(y_test_class, y_pred_class)
    print("分类指标:\n")
    print(f"F1 Score: {f1_score:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    y_pred_value = np.array([class_means[c] for c in y_pred_class])
    mae = mean_absolute_error(y_test, y_pred_value)
    r2 = r2_score(y_test, y_pred_value)
    rmse = root_mean_squared_error(y_test, y_pred_value)
    mean = y_test.mean()
    print("回归指标:\n")
    print(f"MAE (平均绝对误差): {mae:.4f}")
    print(f"R2 (决定系数): {r2:.4f}")
    print(f"RMSE (均方根误差): {rmse:.4f}")
    print(f"平均值: {mean:.4f}")
    print(f"如果分类准确率100%\n")
    y_cluster_pred_value = np.array([class_means[c] for c in y_test_class])
    mae = mean_absolute_error(y_test, y_cluster_pred_value)
    r2 = r2_score(y_test, y_cluster_pred_value)
    rmse = root_mean_squared_error(y_test, y_cluster_pred_value)
    mean = y_test.mean()
    print("分类回归指标:\n")
    print(f"MAE (平均绝对误差): {mae:.4f}")
    print(f"R2 (决定系数): {r2:.4f}")
    print(f"RMSE (均方根误差): {rmse:.4f}")
    print(f"平均值: {mean:.4f}")
