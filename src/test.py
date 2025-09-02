import logging
from data.loader import DiskDataLoader
from config.settings import DirConfig, DataConfig, ModelConfig
from algorithms.SCDA import SCDA
import sys
import os
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_scda_cluster_centers():
    """测试SCDA聚类模型的簇中心"""
    print("=" * 60)
    print("SCDA聚类模型簇中心检查")
    print("=" * 60)

    # 1. 检查是否存在训练好的模型
    model_cluster_dir = os.path.join(
        DirConfig.MODEL_DIR, "SCDA", "model_cluster.pkl")
    model_classify_dir = os.path.join(
        DirConfig.MODEL_DIR, "SCDA", "model_classify.pkl")
    scaler_dir = os.path.join(DirConfig.MODEL_DIR, "SCDA", "scaler.pkl")

    print(f"模型文件路径:")
    print(f"  聚类模型: {model_cluster_dir}")
    print(f"  分类模型: {model_classify_dir}")
    print(f"  缩放器: {scaler_dir}")
    print()

    # 2. 创建SCDA实例并训练/加载模型
    scda = SCDA()

    if not os.path.exists(model_cluster_dir):
        print("模型不存在，开始训练...")
        scda.train_model()
        print("训练完成!")
    else:
        print("发现已存在的模型文件")

    # 3. 加载模型和数据
    try:
        model_cluster, model_classify = scda.load_model()
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 4. 检查聚类模型的基本信息
    print(f"\n聚类模型信息:")
    print(f"  聚类数量 K = {model_cluster.n_clusters}")
    print(f"  簇中心形状: {model_cluster.cluster_centers_.shape}")
    print(f"  特征维度: {model_cluster.cluster_centers_.shape[1]}")

    # 5. 详细检查每个簇中心（缩放后的值）
    print(f"\n📊 缩放后的簇中心坐标:")
    cluster_centers_scaled = model_cluster.cluster_centers_
    for i, center in enumerate(cluster_centers_scaled):
        print(
            f"  簇 {i}: [{center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f}, {center[3]:.6f}]")

    # 6. 检查簇中心是否完全相同
    print(f"\n🔍 簇中心相似性检查:")
    centers_are_identical = True
    for i in range(len(cluster_centers_scaled)):
        for j in range(i+1, len(cluster_centers_scaled)):
            diff = np.abs(
                cluster_centers_scaled[i] - cluster_centers_scaled[j])
            max_diff = np.max(diff)
            print(f"  簇{i} vs 簇{j}: 最大差异 = {max_diff:.8f}")
            if max_diff > 1e-10:  # 考虑浮点数精度
                centers_are_identical = False

    if centers_are_identical:
        print("  ⚠️  警告: 所有簇中心完全相同!")
    else:
        print("  ✅ 簇中心存在差异")

    # 7. 反缩放到原始尺度并检查
    if hasattr(scda, 'scaler') and scda.scaler is not None:
        print(f"\n📊 原始尺度的簇中心坐标:")
        cluster_centers_original = scda.scaler.inverse_transform(
            cluster_centers_scaled)

        for i, center in enumerate(cluster_centers_original):
            print(
                f"  簇 {i}: [avg_rbw={center[0]:.2f}, avg_wbw={center[1]:.2f}, peak_rbw={center[2]:.2f}, peak_wbw={center[3]:.2f}]")

        # 检查原始尺度下的差异
        print(f"\n🔍 原始尺度簇中心相似性检查:")
        original_centers_identical = True
        for i in range(len(cluster_centers_original)):
            for j in range(i+1, len(cluster_centers_original)):
                diff = np.abs(
                    cluster_centers_original[i] - cluster_centers_original[j])
                max_diff = np.max(diff)
                print(f"  簇{i} vs 簇{j}: 最大差异 = {max_diff:.4f}")
                if max_diff > 0.01:  # 原始尺度下的合理阈值
                    original_centers_identical = False

        if original_centers_identical:
            print("  ⚠️  警告: 原始尺度下所有簇中心也几乎相同!")
        else:
            print("  ✅ 原始尺度下簇中心存在合理差异")
    else:
        print("  ❌ 缩放器不存在，无法反缩放")

    # 8. 加载训练数据进行进一步分析
    print(f"\n📈 训练数据分析:")
    try:
        loader = DiskDataLoader()
        items_train = loader.load_items(
            type="both",
            cluster_index_list=DataConfig.CLUSTER_INDEX_LIST_TRAIN,
            purpose="train"
        )

        if len(items_train) > 0:
            # 转换为DataFrame进行分析
            items_df = pd.DataFrame(items_train, columns=[
                "disk_ID", "disk_capacity", "disk_if_local", "disk_attr",
                "disk_type", "disk_if_VIP", "disk_pay", "vm_cpu", "vm_mem",
                "avg_rbw", "avg_wbw", "peak_rbw", "peak_wbw", "timestamp_num",
                "burst_label", "cluster_index"
            ])

            # 分析聚类特征的分布
            cluster_features = ["avg_rbw", "avg_wbw", "peak_rbw", "peak_wbw"]
            print(f"  训练数据样本数: {len(items_df)}")
            print(f"  聚类特征统计:")

            for feature in cluster_features:
                values = items_df[feature].astype(float)
                print(f"    {feature}: min={values.min():.2f}, max={values.max():.2f}, "
                      f"mean={values.mean():.2f}, std={values.std():.2f}")

            # 检查数据是否有足够的变异性
            feature_data = items_df[cluster_features].astype(float)
            print(f"\n  数据变异性检查:")
            print(f"    特征间相关性:")
            correlation_matrix = feature_data.corr()
            print(correlation_matrix.round(3))

            # 检查是否所有数据点都相同
            unique_combinations = feature_data.drop_duplicates()
            print(
                f"    唯一的特征组合数: {len(unique_combinations)} / {len(feature_data)}")

            if len(unique_combinations) < ModelConfig.SCDA_CLUSTER_K:
                print(
                    f"  ⚠️  警告: 唯一特征组合数({len(unique_combinations)}) < 聚类数({ModelConfig.SCDA_CLUSTER_K})")
                print("     这可能导致聚类中心重叠!")

        else:
            print("  ❌ 训练数据为空")

    except Exception as e:
        print(f"  ❌ 训练数据加载失败: {e}")

    # 9. 提供诊断建议
    print(f"\n💡 诊断建议:")
    if centers_are_identical:
        print("  1. 检查训练数据是否有足够的变异性")
        print("  2. 考虑减少聚类数量K")
        print("  3. 检查特征预处理是否正确")
        print("  4. 尝试使用不同的聚类算法参数")
        print("  5. 检查数据是否存在异常值或数据质量问题")
    else:
        print("  ✅ 聚类模型看起来正常")

    print("=" * 60)


def test_prediction_pipeline():
    """测试完整的预测流程"""
    print("\n" + "=" * 60)
    print("SCDA预测流程测试")
    print("=" * 60)

    try:
        scda = SCDA()

        # 加载预测数据
        loader = DiskDataLoader()
        items_predict = loader.load_items(
            type="both",
            cluster_index_list=DataConfig.CLUSTER_INDEX_LIST_PREDICT,
            purpose="train"
        )

        if len(items_predict) > 0:
            items_df = pd.DataFrame(items_predict, columns=[
                "disk_ID", "disk_capacity", "disk_if_local", "disk_attr",
                "disk_type", "disk_if_VIP", "disk_pay", "vm_cpu", "vm_mem",
                "avg_rbw", "avg_wbw", "peak_rbw", "peak_wbw", "timestamp_num",
                "burst_label", "cluster_index"
            ])

            print(f"预测数据样本数: {len(items_df)}")

            # 选择前5个样本进行测试
            test_samples = items_df.head(5)

            print(f"\n测试样本的真实值:")
            for idx, (_, row) in enumerate(test_samples.iterrows()):
                print(
                    f"  样本{idx}: avg_rbw={row['avg_rbw']:.2f}, avg_wbw={row['avg_wbw']:.2f}")

            # 进行预测
            predictions = scda.predict(test_samples)

            print(f"\n预测结果:")
            for idx, (_, row) in enumerate(predictions.head(5).iterrows()):
                print(
                    f"  样本{idx}: pre_avg_rbw={row['pre_avg_rbw']:.6f}, pre_avg_wbw={row['pre_avg_wbw']:.6f}")

            # 计算预测误差
            print(f"\n预测误差分析:")
            for idx in range(min(5, len(predictions))):
                true_rbw = float(test_samples.iloc[idx]['avg_rbw'])
                true_wbw = float(test_samples.iloc[idx]['avg_wbw'])
                pred_rbw = float(predictions.iloc[idx]['pre_avg_rbw'])
                pred_wbw = float(predictions.iloc[idx]['pre_avg_wbw'])

                error_rbw = abs(true_rbw - pred_rbw)
                error_wbw = abs(true_wbw - pred_wbw)
                relative_error_rbw = error_rbw / max(true_rbw, 1e-6) * 100
                relative_error_wbw = error_wbw / max(true_wbw, 1e-6) * 100

                print(f"  样本{idx}:")
                print(
                    f"    RBW误差: {error_rbw:.4f} (相对误差: {relative_error_rbw:.1f}%)")
                print(
                    f"    WBW误差: {error_wbw:.4f} (相对误差: {relative_error_wbw:.1f}%)")

        else:
            print("❌ 预测数据为空")

    except Exception as e:
        print(f"❌ 预测流程测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 确保目录存在
    DirConfig.ensure_dirs()

    # 运行测试
    test_scda_cluster_centers()
    test_prediction_pipeline()
