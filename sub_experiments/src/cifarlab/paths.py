"""项目目录约定：所有产物都落到这里定义的目录，脚本里不再出现硬编码路径。

    from cifarlab import paths
    paths.CKPT / 'googlenet_best.pth'
"""
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]           # 仓库根（submission/）
ROOT = REPO.parent                                   # 数据目录所在处
DATA = ROOT / 'CIFAR-10'                    # 原始 pickle（data_batch_1..5 / test_batch）
DATA100 = ROOT / 'CIFAR-100' / 'cifar-100-python'  # CIFAR-100（train / test / meta）
RESULTS = REPO / 'results'
CKPT = RESULTS / 'checkpoints'              # 主模型权重
ABLATIONS = CKPT / 'ablations'              # 17 组对照的权重
FIGURES = RESULTS / 'figures'               # 曲线 / 混淆矩阵 / 汇总图
LOGS = RESULTS / 'logs'                     # 训练日志
METRICS = RESULTS / 'metrics'               # 每组实验的 json 指标
TB = RESULTS / 'tb'                         # TensorBoard 事件文件

for _dir in (CKPT, ABLATIONS, FIGURES, LOGS, METRICS, TB):
    _dir.mkdir(parents=True, exist_ok=True)
