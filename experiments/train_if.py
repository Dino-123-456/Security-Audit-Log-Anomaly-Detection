# experiments/train_if.py
import argparse
import yaml
import numpy as np
import joblib
import json
from datetime import datetime
from platform import platform as platform_name
import sys
from pathlib import Path
from models.if_detector import IFDetector
from utils.feature_engineering import LogFeatureEngineer
from utils.logger import get_logger

def main(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    log_dir = Path(config['evaluation'].get('log_dir', 'outputs/logs'))
    logger = get_logger('train_if', log_dir=str(log_dir))
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')

    logger.info(f"加载配置: {config_path}")

    # === 1. 加载数据 ===
    from data.data_loader import BGLDataLoader
    loader = BGLDataLoader(config)
    data = loader.load()

    # 读取配置中的 max_features，默认 800，并构造 feature engineer 配置
    if_max_features = config.get('if_model', {}).get('tfidf_max_features', 800)
    fe_cfg = {
        'ngram_range': (1, 2),
        'max_features': if_max_features,
        'sublinear_tf': True,
        'use_entropy': config.get('feature_engineering', {}).get('use_entropy', True),
        'use_failure_rate': config.get('feature_engineering', {}).get('use_failure_rate', True)
    }

    # === 2. 特征工程 (Bigram TF-IDF + 可选熵/异常率) ===
    fe = LogFeatureEngineer(fe_cfg)
    X_train_tfidf = fe.fit_transform(data['X_train'], window_failure_rate=data['window_failure_rate_train'])
    X_val_tfidf = fe.transform(data['X_val'], window_failure_rate=data['window_failure_rate_val'])
    X_test_tfidf = fe.transform(data['X_test'], window_failure_rate=data['window_failure_rate_test'])
    y_train, y_val, y_test = data['y_train'], data['y_val'], data['y_test']

    dataset_summary = {
        'raw_path': config['data']['raw_path'],
        'processed_dir': config['data']['processed_dir'],
        'cache_path': str(loader.cache_path),
        'window_size': int(config['data']['window_size']),
        'step_size': int(config['data']['step_size']),
        'train_ratio': float(config['data']['train_ratio']),
        'val_ratio': float(config['data']['val_ratio']),
        'random_seed': int(config['data']['random_seed']),
        'vocab_size': int(data['vocab_size']),
        'n_train': int(len(data['X_train'])),
        'n_val': int(len(data['X_val'])),
        'n_test': int(len(data['X_test'])),
        'train_anomaly_rate': float(np.mean(y_train)),
        'val_anomaly_rate': float(np.mean(y_val)),
        'test_anomaly_rate': float(np.mean(y_test)),
        'train_window_failure_rate_mean': float(np.mean(data['window_failure_rate_train'])),
        'val_window_failure_rate_mean': float(np.mean(data['window_failure_rate_val'])),
        'test_window_failure_rate_mean': float(np.mean(data['window_failure_rate_test'])),
        'train_feature_shape': [int(x) for x in X_train_tfidf.shape],
        'val_feature_shape': [int(x) for x in X_val_tfidf.shape],
        'test_feature_shape': [int(x) for x in X_test_tfidf.shape],
        'n_feature_names': int(len(fe.get_feature_names())),
    }

    # 让 IF 默认更稳：避免在 WSL2 中把所有核心拉满
    if_cfg = dict(config.get('if_model', {}))
    if_cfg['n_jobs'] = int(if_cfg.get('n_jobs', 1))
    if_cfg['max_features'] = min(float(if_cfg.get('max_features', 0.8)), 0.8)
    config['if_model'] = if_cfg

    # === 3. 模型训练与评估 ===
    detector = IFDetector(config)
    detector.fit(X_train_tfidf, y_train, X_val_tfidf, y_val)
    metrics = detector.evaluate(X_test_tfidf, y_test)
    logger.info(f"测试集评估结果: {metrics}")

    # === 4. 完整保存所有产出物 ===
    output_dir = Path(config['evaluation']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 4.1 保存评估指标
    metrics_path = output_dir / "if_evaluation_metrics.npz"
    np.savez(metrics_path, **metrics)
    logger.info(f"评估指标已保存: {metrics_path}")
    
    # 4.2 保存 TF-IDF 向量化器
    vectorizer_path = output_dir / "if_tfidf_vectorizer.pkl"
    joblib.dump(fe.vectorizer, vectorizer_path)
    logger.info(f"TF-IDF向量化器已保存: {vectorizer_path}")
    
    # 4.3 保存训练好的 Isolation Forest 模型
    model_path = output_dir / "if_model.pkl"
    joblib.dump(detector.model, model_path)
    logger.info(f"Isolation Forest 模型已保存: {model_path}")

    # 保存完整 artifact 以便后续一键加载（包括 detector 与 feature 工程器/元信息）
    checkpoint_dir = Path(config['evaluation'].get('checkpoint_dir', 'outputs/checkpoints'))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = checkpoint_dir / 'if_detector_artifact.pkl'
    detector.save_artifact(str(artifact_path), feature_engineering=fe)

    summary_path = log_dir / f"{run_id}_if_training_summary.json"

    # 保存训练摘要，便于复现实验和快速比对
    summary = {
        'run_id': run_id,
        'config_path': config_path,
        'replay_command': f".venv/bin/python3 -m experiments.train_if --config {config_path}",
        'environment': {
            'python_version': sys.version,
            'platform': platform_name(),
        },
        'paths': {
            'output_dir': str(output_dir),
            'log_dir': str(log_dir),
            'checkpoint_dir': str(checkpoint_dir),
            'metrics_path': str(metrics_path),
            'vectorizer_path': str(vectorizer_path),
            'model_path': str(model_path),
            'artifact_path': str(artifact_path),
            'summary_path': str(summary_path),
        },
        'dataset_summary': dataset_summary,
        'threshold': detector.threshold,
        'threshold_strategy': detector.threshold_strategy,
        'calibration_info': detector.calibration_info,
        'metrics': metrics,
        'if_model': config.get('if_model', {}),
        'feature_engineering': fe_cfg,
        'feature_names_preview': fe.get_feature_names()[:20],
    }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"训练摘要已保存: {summary_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Isolation Forest anomaly detector")
    parser.add_argument("--config", type=str, default="configs/experiment_config.yaml")
    args = parser.parse_args()
    main(args.config)