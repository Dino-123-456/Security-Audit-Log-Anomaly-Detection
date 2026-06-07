# utils/metrics.py
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from typing import Optional, Dict
import time


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: Optional[np.ndarray] = None,
    higher_is_more_anomalous: bool = True
) -> Dict[str, float]:
    """
    统一评估函数，确保IF与LSTM-AE使用完全相同的指标计算逻辑。

    Args:
        y_true: 真实标签 (0=正常, 1=异常)
        y_pred: 二分类预测结果
        y_scores: 连续异常分数
        higher_is_more_anomalous: 分数方向标记。
            True  = 分数越大越异常 (LSTM-AE重构误差, 或已取负的IF分数)
            False = 分数越小越异常 (IF原始decision_function)
            ⚠️ 调用方必须显式传入此参数，防止默认值导致的静默错误
    Returns:
        包含precision, recall, f1, auc的字典
    """
    metrics = {
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_scores is not None:
        # 检查测试集是否同时包含正负样本
        if len(np.unique(y_true)) < 2:
            metrics['auc'] = float('nan')
        else:
            # 🔑 关键修复：统一为"越大越异常"后再计算AUC
            scores_for_auc = y_scores if higher_is_more_anomalous else -y_scores
            try:
                metrics['auc'] = float(roc_auc_score(y_true, scores_for_auc))
            except ValueError:
                metrics['auc'] = float('nan')
    else:
        metrics['auc'] = float('nan')

    return metrics


def measure_inference_time(predict_fn, X: np.ndarray, n_runs: int = 10) -> float:
    """测量推理耗时（秒），去除首次运行的冷启动开销"""
    # 预热
    predict_fn(X[:min(10, len(X))])
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        predict_fn(X)
        times.append(time.perf_counter() - start)
    return float(np.median(times))