# experiments/train_if.py
"""
训练脚本：Isolation Forest 基线模型。
修改点：增加了 Top-K (100, 500, 1000) 精确率评估。
"""
import argparse
import yaml
import numpy as np
import joblib
import json
from datetime import datetime
from pathlib import Path
from models.if_detector import IFDetector
from utils.feature_engineering import LogFeatureEngineer
from utils.logger import get_logger
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

def calculate_topk_precision(y_true, y_scores, k_values=[100, 500, 1000]):
    """
    计算 Top-K 精确率的辅助函数。
    Isolation Forest: score_samples() 返回值越小越异常。
    所以这里取负号，使高分=高异常。
    """
    # IF 的原始分数：越小越异常 -> 转换为：越大越异常
    anomaly_scores = -np.array(y_scores) 
    
    topk_precisions = {}
    sorted_indices = np.argsort(anomaly_scores)[::-1] # 降序排列索引
    
    for k in k_values:
        if k > len(y_true):
            topk_precisions[f'p_at_{k}'] = 0.0
            continue
        top_k_indices = sorted_indices[:k]
        precision_at_k = np.mean(y_true[top_k_indices])
        topk_precisions[f'p_at_{k}'] = float(precision_at_k)
    return topk_precisions

def main(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    log_dir = Path(config['evaluation'].get('log_dir', 'outputs/logs'))
    logger = get_logger('train_if', log_dir=str(log_dir))
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger.info(f"加载配置: {config_path}")

    # === 1. 数据加载 ===
    from data.data_loader import BGLDataLoader
    loader = BGLDataLoader(config)
    data = loader.load()
    
    # 特征工程
    if_max_features = config.get('if_model', {}).get('tfidf_max_features', 800)
    fe_cfg = {
        'ngram_range': (1, 2),
        'max_features': if_max_features,
        'sublinear_tf': True,
        'use_entropy': config.get('feature_engineering', {}).get('use_entropy', True),
        'use_failure_rate': config.get('feature_engineering', {}).get('use_failure_rate', True)
    }
    fe = LogFeatureEngineer(fe_cfg)
    
    X_train_tfidf = fe.fit_transform(data['X_train'], window_failure_rate=data['window_failure_rate_train'])
    X_val_tfidf = fe.transform(data['X_val'], window_failure_rate=data['window_failure_rate_val'])
    X_test_tfidf = fe.transform(data['X_test'], window_failure_rate=data['window_failure_rate_test'])
    
    y_train, y_val, y_test = data['y_train'], data['y_val'], data['y_test']

    # === 2. 模型训练 ===
    detector = IFDetector(config)
    detector.fit(X_train_tfidf, y_train, X_val_tfidf, y_val)

    # === 3. 评估 (包含 Top-K) ===
    logger.info("开始在测试集上评估 (包含 Top-K 指标)...")
    
    # 获取 IF 原始分数 (越小越异常)
    test_raw_scores = detector.model.score_samples(X_test_tfidf)
    
    # 预测标签 (用于传统指标)
    y_pred = detector.predict(X_test_tfidf)
    
    # 计算 AUC 和 F1
    auc = roc_auc_score(y_test, -test_raw_scores) # 注意符号转换
    
    # 计算 Top-K
    topk_metrics = calculate_topk_precision(y_test, test_raw_scores, k_values=[100, 500, 1000])

    metrics = {
        'auc': auc,
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        **topk_metrics
    }
    
    logger.info(f"测试集评估结果: {metrics}")

    # === 4. 保存 ===
    output_dir = Path(config['evaluation']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_path = output_dir / "if_evaluation_metrics.npz"
    np.savez(metrics_path, **metrics)
    
    summary_path = log_dir / f"{run_id}_if_training_summary.json"
    summary = {
        'run_id': run_id,
        'config_path': config_path,
        'metrics': metrics,
        'threshold': float(detector.threshold),
        'model_type': 'isolation_forest'
    }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(f"IF 模型训练完成，指标已保存。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Isolation Forest Anomaly Detector")
    parser.add_argument("--config", type=str, default="configs/experiment_config.yaml")
    args = parser.parse_args()
    main(args.config)