from visualization.plotter import DiskPlotter, CoachPlotter
from data.processor import DiskDataProcessor, CoachProcessor
from data.loader import DiskDataLoader
from cluster_info_init.cluster_info_initializer import ClusterInfoInitializer
from config.settings import DirConfig, DataConfig, WarehouseConfig
from algorithms.ODA import ODA
from algorithms.SCDA import SCDA
from algorithms.TELA import TELA
import logging
import argparse
import sys
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(filename)s - %(lineno)s - %(message)s',
    handlers=[
        logging.FileHandler('disk_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class DiskCycleDetector:
    """磁盘周期检测器主类"""

    def __init__(self):
        self.loader = DiskDataLoader()
        self.processor = DiskDataProcessor()
        self.plotter = DiskPlotter()

        # 确保必要的目录存在
        DirConfig.ensure_dirs()

    def run_burst_analysis(self):
        """运行突发性分析"""
        logger.info("开始突发性磁盘分析")

        # 1. 加载突发性磁盘数据
        burst_items = self.loader.load_items(type="burst")

        if not burst_items:
            logger.error("未加载到任何突发性磁盘数据")
            return

        total_disks = len(burst_items)
        logger.info(f"共加载 {total_disks} 个突发性磁盘")

        # 2. 计算峰值时间
        peak_time = self.processor.calculate_peak_time(burst_items)
        logger.info("峰值时间计算完成")

        # 3. 计算方差系数
        rbw_scores, wbw_scores = self.processor.calculate_all_variance_coefficients()
        logger.info(f"计算方差系数完成: RBW={len(rbw_scores)}, WBW={len(wbw_scores)}")

        # 4. 绘制方差密度图
        self.plotter.plot_variance_density(rbw_scores, wbw_scores)

        # 5. 绘制峰值分布图（采样）
        self.plotter.plot_peak_distribution(sample_interval=500)

        logger.info("突发性磁盘分析完成")

    def run_stable_analysis(self):
        """运行稳定性分析"""
        logger.info("开始稳定性磁盘分析")

        # 加载稳定性磁盘数据
        stable_items = self.loader.load_items(type="stable")

        if not stable_items:
            logger.error("未加载到任何稳定性磁盘数据")
            return

        total_disks = sum(len(cluster) for cluster in stable_items)
        logger.info(f"共加载 {total_disks} 个稳定性磁盘")

        logger.info("稳定性磁盘分析完成")

    def plot_disk_traces(self, cluster_index: int, disk_id: str, trace_type: str = "week"):
        """绘制特定磁盘的追踪图"""
        logger.info(f"绘制磁盘 {disk_id} (集群{cluster_index}) 的追踪图")
        self.plotter.plot_disk_trace(cluster_index, disk_id, trace_type)

    def run_full_analysis(self):
        """运行完整分析"""
        logger.info("开始完整的磁盘周期检测分析")

        try:
            # 运行突发性分析
            self.run_burst_analysis()

            # 运行稳定性分析
            self.run_stable_analysis()

            logger.info("完整分析完成")

        except Exception as e:
            logger.error(f"分析过程中发生错误: {e}")
            raise


class CoachCycleDetector:
    def __init__(self):
        self.warehouse_num = WarehouseConfig.WAREHOUSE_NUMBER
        self.time_window_length = DataConfig.TIME_WINDOW_LENGTH
        self.loader = DiskDataLoader()
        self.processor = CoachProcessor()
        self.plotter = CoachPlotter()

    def run_coach_analysis(self) -> None:
        logger.info("开始coach分析")
        items = self.loader.load_items(type="burst",
                                       cluster_index_list=DataConfig.CLUSTER_INDEX_LIST)
        if not items:
            logger.error("未加载到任何突发性磁盘数据")
            return
        rbw_peak, rbw_valley, wbw_peak, wbw_valley = self.processor.statistic_peak_valley_window_of_week(
            items, self.time_window_length)
        logger.info("coach分析完成")
        self.plotter.plot_peak_valley_windows_distribution(
            rbw_peak, rbw_valley, wbw_peak, wbw_valley)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='云盘放置策略分析')
    subparsers = parser.add_subparsers(dest='mode', required=True, help='运行模式')

    # init 子命令
    init_parser = subparsers.add_parser('init', help='初始化仓库数据')

    # statistic 子命令
    statistic_parser = subparsers.add_parser('statistic', help='统计分析')
    statistic_parser.add_argument(
        '--method', choices=['coach', 'burst', 'stable'], required=True, help='分析方法')

    # plot 子命令
    plot_parser = subparsers.add_parser('plot', help='绘图')
    plot_parser.add_argument(
        '--method', choices=['trace', 'cluster'], required=True, help='绘图方法')
    plot_parser.add_argument(
        '--cluster-index', type=int, help='集群索引 (trace模式需要)')
    plot_parser.add_argument('--disk-id', type=str, help='磁盘ID (trace模式需要)')
    plot_parser.add_argument(
        '--trace-time', choices=['week', 'all'], type=str, help='追踪时间 (trace模式需要)')

    # placement 子命令
    placement_parser = subparsers.add_parser('placement', help='放置策略分析')
    placement_parser.add_argument(
        '--algorithm', choices=['odp', 'scda', 'tela'], required=True, help='放置方法')

    args = parser.parse_args()

    detector = DiskCycleDetector()
    coach_detector = CoachCycleDetector()

    if args.mode == 'statistic':
        if args.method == 'burst':
            detector.run_burst_analysis()
        elif args.method == 'stable':
            detector.run_stable_analysis()
        elif args.method == 'coach':
            coach_detector.run_coach_analysis()
        else:
            logger.error(f"未知的statistic分析方法: {args.method}")
            sys.exit(1)
    elif args.mode == 'plot':
        if args.method == 'trace':
            if args.cluster_index is None or args.disk_id is None or args.trace_time is None:
                logger.error(
                    "plot-trace模式需要 --cluster-index, --disk-id, --trace-time 参数")
                sys.exit(1)
            if args.trace_time == 'week':
                detector.plot_disk_traces(
                    args.cluster_index, args.disk_id, args.trace_time)
            elif args.trace_time == 'all':
                detector.plot_disk_traces(
                    args.cluster_index, args.disk_id, args.trace_time)
            else:
                logger.error(f"未知的trace-time: {args.trace_time}")
                sys.exit(1)
        elif args.method == 'cluster':
            logger.info("plot-cluster模式暂未实现，仅作占位")
        else:
            logger.error(f"未知的plot绘图方法: {args.method}")
            sys.exit(1)
    elif args.mode == 'placement':
        if args.algorithm == 'odp':
            oda_analyzer = ODA()
            logger.info("运行ODA放置策略分析")
            oda_analyzer.run()
        elif args.algorithm == 'scda':
            logger.info("运行SCDA放置策略分析）")
            scda_analyzer = SCDA()
            scda_analyzer.run()
        elif args.algorithm == 'tela':
            logger.info("运行TELA放置策略分析（请补充实现）")
            tela_analyzer = TELA()
            tela_analyzer.run()
        else:
            logger.error(f"未知的placement放置方法: {args.method}")
            sys.exit(1)
    elif args.mode == 'init':
        initializer = ClusterInfoInitializer()
        initializer.init_cluster_info()
    else:
        logger.error(f"未知的运行模式: {args.mode}")
        sys.exit(1)


if __name__ == '__main__':
    main()
