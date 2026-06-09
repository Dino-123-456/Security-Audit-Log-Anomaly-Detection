# experiments/train_lstm_v5.py
"""
训练脚本：专用于 v5 LSTM Autoregressive 模型。
"""
import argparse
import yaml
import numpy as np
import torch
import json
from datetime import datetime
from pathlib import Path
from models.lstm_v5_detector import LSTMAutoregressiveDetector
from utils.logger import get_logger
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


# 复用 v6 的 Top-K 计算函数
def calculate_topk_precision(y_true, y_scores, k_values=[100, 500, 1000]):
    """计算 Top-K 精确率"""
    topk_precisions = {}
    sorted_indices = np.argsort(y_scores)[::-1]
    for k in k_values:
        if k > len(y_true):
            topk_precisions[f'p_at_{k}'] = None
            continue
        top_k_indices = sorted_indices[:k]
        precision_at_k = np.mean(y_true[top_k_indices])
        topk_precisions[f'p_at_{k}'] = float(precision_at_k)
    return topk_precisions


def main(config_path: str):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) # 这里原本就是对的

    log_dir = Path(config['evaluation'].get('log_dir', 'outputs/logs'))
    logger = get_logger('train_lstm_v5', log_dir=str(log_dir))
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger.info(f"🚀 开始训练 v5 LSTM Autoregressive | 加载配置: {config_path}")

    # === 1. 加载数据 ===
    from data.data_loader import BGLDataLoader
    loader = BGLDataLoader(config)
    data = loader.load()
    X_train, X_val, X_test = data['X_train'], data['X_val'], data['X_test']
    
    # 【关键逻辑】恢复使用 y (Any 标签)
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']
    
    vocab_size = int(data['vocab_size'])
    logger.info(f"数据加载完成 | Vocab Size: {vocab_size} "
                f"| Train: {X_train.shape} | Test: {X_test.shape}")

    # === 2. 初始化模型 ===
    detector = LSTMAutoregressiveDetector(config, vocab_size=vocab_size)

    # === 3. 训练 ===
    logger.info("开始模型训练 (LSTM Autoregressive)...")
    history = detector.fit(X_train, y_train, X_val, y_val)

    # === 4. 评估 ===
    logger.info("开始测试集评估...")
    
    # 获取异常分数
    test_scores = detector.score(X_test)
    
    # 计算阈值 (如果未自动计算)
    if not hasattr(detector, 'threshold') or detector.threshold is None:
        val_scores = detector.score(X_val)
        detector.threshold = np.percentile(val_scores, 95)
        
    y_pred = (test_scores > detector.threshold).astype(int)

    # 计算指标
    auc = roc_auc_score(y_test, test_scores)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # 新增 Top-K
    topk_metrics = calculate_topk_precision(y_test, test_scores, k_values=[100, 500, 1000])

    metrics = {
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        **topk_metrics
    }
    logger.info(f"测试集评估结果: {metrics}")

    # === 5. 保存产出 ===
    output_dir = Path(config['evaluation']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "v5_lstm_autoregressive_metrics.npz"
    np.savez(metrics_path, **metrics)
    logger.info(f"指标已保存: {metrics_path}")

    model_path = output_dir / "v5_lstm_autoregressive_model.pt"
    torch.save(detector.model.state_dict(), model_path)
    logger.info(f"模型已保存: {model_path}")

    summary_path = log_dir / f"{run_id}_v5_training_summary.json"
    summary = {
        'run_id': run_id,
        'config_path': config_path,
        'metrics': metrics,
        'threshold': float(detector.threshold),
        'model_type': 'v5_lstm_autoregressive'
    }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"训练摘要已保存: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train v5 LSTM Autoregressive Anomaly Detector")
    parser.add_argument("--config", type=str, default="configs/experiment_config.yaml")
    args = parser.parse_args()
    main(args.config)