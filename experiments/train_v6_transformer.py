# experiments/train_v6_transformer.py
"""
训练脚本：专用于 v6 Causal Transformer 模型。
功能：加载配置 -> 数据加载 -> 模型训练 -> Top-K 评估。
"""
import argparse
import yaml
import numpy as np
import torch
import json
from datetime import datetime
from pathlib import Path
from models.transformer_v6_detector import LSTMAutoregressiveDetector as V6Detector
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
    # 按分数从高到低排序，取前 K 个的索引
    topk_precisions = {}
    sorted_indices = np.argsort(y_scores)[::-1] # 降序
    
    for k in k_values:
        if k > len(y_true):
            topk_precisions[f'p_at_{k}'] = None # 或者设为 np.nan
            continue
        top_k_indices = sorted_indices[:k]
        # 计算前 K 个中真实异常的比例
        precision_at_k = np.mean(y_true[top_k_indices])
        topk_precisions[f'p_at_{k}'] = float(precision_at_k)
    return topk_precisions

def main(config_path: str):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    log_dir = Path(config['evaluation'].get('log_dir', 'outputs/logs'))
    logger = get_logger('train_v6_transformer', log_dir=str(log_dir))
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger.info(f"🚀 开始训练 v6 Causal Transformer | 加载配置: {config_path}")

    # === 1. 加载数据 ===
    from data.data_loader import BGLDataLoader
    loader = BGLDataLoader(config)
    data = loader.load()
    X_train, X_val, X_test = data['X_train'], data['X_val'], data['X_test']
    y_train, y_val, y_test = data['y_train'], data['y_val'], data['y_test']
    vocab_size = int(data['vocab_size'])
    
    logger.info(f" 数据加载完成 | Vocab: {vocab_size} | Train: {X_train.shape} | Test: {X_test.shape}")

    # === 2. 初始化模型 ===
    # 强制使用 causal_transformer 模式
    detector = V6Detector(config, vocab_size=vocab_size)

    # === 3. 训练 ===
    logger.info(" 开始模型训练 (Causal Transformer)...")
    history = detector.fit(X_train, y_train, X_val, y_val)

    # === 4. 评估与 Top-K 计算 ===
    logger.info(" 开始测试集评估 (包含 Top-K 指标)...")
    # 获取分数（越高越异常）
    test_scores = detector.score(X_test)
    
    # 计算传统指标
    y_pred = (test_scores > detector.threshold).astype(int)
    auc = roc_auc_score(y_test, test_scores)
    
    # 计算 Top-K 精确率
    topk_metrics = calculate_topk_precision(y_test, test_scores, k_values=[100, 500, 1000])
    
    # 合并所有指标
    metrics = {
        'auc': auc,
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        **topk_metrics # 合并 p_at_100 等
    }
    
    logger.info(f"📊 最终评估结果: {metrics}")

    # === 5. 保存产出 ===
    output_dir = Path(config['evaluation']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存指标
    metrics_path = output_dir / "v6_transformer_evaluation_metrics.npz"
    np.savez(metrics_path, **metrics)
    
    # 保存模型
    model_path = output_dir / "v6_transformer_model.pt"
    torch.save(detector.model.state_dict(), model_path)
    
    # 保存摘要
    summary_path = log_dir / f"{run_id}_v6_training_summary.json"
    summary = {
        'run_id': run_id,
        'config_path': config_path,
        'metrics': metrics,
        'threshold': float(detector.threshold),
        'model_type': 'v6_causal_transformer'
    }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(f"✅ 训练完成。模型保存至: {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train v6 Causal Transformer Anomaly Detector")
    parser.add_argument("--config", type=str, default="configs/experiment_config.yaml")
    args = parser.parse_args()
    main(args.config)