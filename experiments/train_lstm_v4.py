# experiments/train_lstm_v4.py
"""
训练脚本：专用于 v4 LSTM Autoencoder (Masked Prediction) 模型。
基于双向 LSTM 重构误差进行异常检测。
"""
import argparse
import yaml
import numpy as np
import torch
import json
from datetime import datetime
from pathlib import Path
from models.lstm_ae_detector import LSTMAEDetector
from utils.logger import get_logger
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


def calculate_topk_precision(y_true, y_scores, k_values=[100, 500, 1000]):
    """
    计算 Top-K 精确率。
    :param y_true: 真实标签数组
    :param y_scores: 异常分数数组（分数越高越异常）
    :param k_values: K 值列表
    :return: dict of p@k
    """
    topk_precisions = {}
    sorted_indices = np.argsort(y_scores)[::-1] # 降序排列索引
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
        # 修复点1: safe_grad -> safe_load
        config = yaml.safe_load(f) 

    log_dir = Path(config['evaluation'].get('log_dir', 'outputs/logs'))
    logger = get_logger('train_lstm_v4', log_dir=str(log_dir))
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger.info(f"🚀 开始训练 v4 LSTM Autoencoder | 加载配置: {config_path}")

    # === 1. 加载数据 ===
    from data.data_loader import BGLDataLoader
    loader = BGLDataLoader(config)
    data = loader.load()
    X_train, X_val, X_test = data['X_train'], data['X_val'], data['X_test']
    
    # 【关键逻辑】v4 使用 Masked 标签
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']
    
    vocab_size = int(data['vocab_size'])
    logger.info(f"数据加载完成 | Vocab Size: {vocab_size} "
                f"| Train: {X_train.shape} | Test: {X_test.shape}")

    # === 2. 初始化模型 ===
    detector = LSTMAEDetector(config, vocab_size=vocab_size)

    # === 3. 训练 ===
    logger.info("开始模型训练 (LSTM Autoencoder - Masked Prediction)...")
    history = detector.fit(X_train, y_train, X_val, y_val)

    # === 4. 评估 ===
    logger.info("开始测试集评估...")
    
    # 获取异常分数 (越高越异常)
    test_scores = detector.score(X_test) 
    
    # 计算阈值和二分类预测
    # 假设 detector.calibrate_threshold 可以根据 y_val 和 X_val 分数计算阈值
    # 如果没有该方法，可能需要根据 validation 分数手动计算
    if not hasattr(detector, 'threshold') or detector.threshold is None:
        # 简单示例：使用验证集分数的 95% 分位数作为阈值
        val_scores = detector.score(X_val)
        detector.threshold = np.percentile(val_scores, 95)
        
    y_pred = (test_scores > detector.threshold).astype(int)

    # 计算传统指标
    auc = roc_auc_score(y_test, test_scores)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # 计算 Top-K 精确率 (修复点2: 新增指标)
    topk_metrics = calculate_topk_precision(y_test, test_scores, k_values=[100, 500, 1000])

    # 合并所有指标
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

    # 1. 保存指标
    metrics_path = output_dir / "v4_lstm_ae_metrics.npz"
    np.savez(metrics_path, **metrics)
    logger.info(f"指标已保存: {metrics_path}")

    # 2. 保存模型权重
    model_path = output_dir / "v4_lstm_ae_model.pt"
    torch.save(detector.model.state_dict(), model_path)
    logger.info(f"模型已保存: {model_path}")

    # 3. 保存训练摘要 (JSON)
    summary_path = log_dir / f"{run_id}_v4_training_summary.json"
    summary = {
        'run_id': run_id,
        'config_path': config_path,
        'paths': {
            'output_dir': str(output_dir),
            'log_dir': str(log_dir),
            'metrics_path': str(metrics_path),
            'model_path': str(model_path),
            'summary_path': str(summary_path),
        },
        'dataset_summary': {
            'vocab_size': vocab_size,
            'train_shape': [int(x) for x in X_train.shape],
            'val_shape': [int(x) for x in X_val.shape],
            'test_shape': [int(x) for x in X_test.shape],
            'train_anomaly_rate': float(y_train.mean()),
            'val_anomaly_rate': float(y_val.mean()),
            'test_anomaly_rate': float(y_test.mean()),
        },
        'lstm_v4': config.get('lstm_ae', {}),
        'metrics': metrics,
        'history': history,
        'threshold': float(detector.threshold),
        'calibration_info': detector.calibration_info
    }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"训练摘要已保存: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train v4 LSTM Autoencoder Anomaly Detector")
    parser.add_argument("--config", type=str, default="configs/experiment_config.yaml")
    args = parser.parse_args()
    main(args.config)