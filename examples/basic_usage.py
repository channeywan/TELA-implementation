#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基本使用示例
演示如何使用重构后的磁盘性能周期检测代码
"""

from visualization.plotter import DiskPlotter
from data.processor import DiskDataProcessor
from data.loader import DiskDataLoader
from config.settings import Config
import sys
import os
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def example_basic_analysis():
    """基本分析示例"""
    print("=== 基本磁盘性能分析示例 ===")

    # 初始化组件
    loader = DiskDataLoader()
    processor = DiskDataProcessor()
    plotter = DiskPlotter()

    # 1. 加载突发性磁盘数据
    print("1. 正在加载突发性磁盘数据...")
    burst_items = loader.load_burst_items()

    if burst_items:
        total_disks = sum(len(cluster) for cluster in burst_items)
        print(f"   成功加载 {total_disks} 个突发性磁盘")

        # 2. 计算峰值时间
        print("2. 正在计算峰值时间...")
        peak_time = processor.calculate_peak_time(burst_items)
        print("   峰值时间计算完成")

        # 3. 计算方差系数
        print("3. 正在计算方差系数...")
        rbw_scores, wbw_scores = processor.calculate_all_variance_coefficients()
        print(f"   方差计算完成: RBW={len(rbw_scores)}, WBW={len(wbw_scores)}")

        # 4. 绘制方差密度图
        print("4. 正在绘制方差密度图...")
        plotter.plot_variance_density(rbw_scores, wbw_scores, "示例：磁盘方差分析")
        print("   方差密度图绘制完成")

    else:
        print("   未找到突发性磁盘数据")


def example_single_disk_analysis():
    """单个磁盘分析示例"""
    print("\n=== 单个磁盘分析示例 ===")

    plotter = DiskPlotter()

    # 绘制特定磁盘的追踪图 (需要根据实际数据调整参数)
    cluster_index = 0
    disk_id = "example_disk_id"  # 替换为实际的磁盘ID

    print(f"正在绘制磁盘 {disk_id} (集群{cluster_index}) 的追踪图...")

    try:
        plotter.plot_disk_trace(cluster_index, disk_id, "week")
        print("   磁盘追踪图绘制完成")
    except Exception as e:
        print(f"   绘制失败: {e}")


def example_configuration():
    """配置示例"""
    print("\n=== 配置信息示例 ===")

    config = Config()

    print(f"项目根目录: {config.PROJECT_ROOT}")
    print(f"数据根目录: {config.TRACE_ROOT}")
    print(f"输出目录: {config.OUTPUT_DIR}")

    # 确保目录存在
    config.ensure_dirs()
    print("已确保所有必要目录存在")


if __name__ == '__main__':
    try:
        # 运行配置示例
        example_configuration()

        # 运行基本分析示例
        example_basic_analysis()

        # 运行单个磁盘分析示例
        example_single_disk_analysis()

        print("\n=== 所有示例运行完成 ===")

    except Exception as e:
        print(f"运行示例时发生错误: {e}")
        import traceback
        traceback.print_exc()
