# experiments/train_lstm_ae.py
import argparse
import yaml
import numpy as np
import torch
import json
from datetime import datetime
from pathlib import Path
from models.lstm_ae_detector import LSTMAEDetector
from models.lstm_v5_detector import LSTMAutoregressiveDetector
from models.lstm_v6_detector import LSTMAutoregressiveDetector as LSTMV6Detector
from utils.logger import get_logger

def main(config_path: str):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    log_dir = Path(config['evaluation'].get('log_dir', 'outputs/logs'))
    logger = get_logger('train_lstm_ae', log_dir=str(log_dir))
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')

    logger.info(f"加载配置: {config_path}")

    # === 1. 加载数据 ===
    from data.data_loader import BGLDataLoader
    loader = BGLDataLoader(config)
    data = loader.load()
    
    X_train, X_val, X_test = data['X_train'], data['X_val'], data['X_test']
    
    # 【关键修改】恢复使用 y (Any 标签)！
    # 因为 Masked + Max Pooling 范式能够捕捉窗口内任何位置的异常，
    # 它的视野与 Any 标签完美契合。
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']
    
    vocab_size = int(data['vocab_size'])
    
    logger.info(
        f"数据加载完成 | Vocab Size: {vocab_size} | "
        f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}"
    )
    logger.info(
        f"标签分布 (Any) -> "
        f"Train: {y_train.mean():.2%} | Val: {y_val.mean():.2%} | Test: {y_test.mean():.2%}"
    )

    # === 2. 初始化并训练检测器 ===
    lstm_mode = config.get('lstm_ae', {}).get('mode', 'masked')
    if lstm_mode == 'autoregressive':
        detector = LSTMAutoregressiveDetector(config, vocab_size=vocab_size)
    elif lstm_mode == 'causal_transformer':
        detector = LSTMV6Detector(config, vocab_size=vocab_size)
    else:
        detector = LSTMAEDetector(config, vocab_size=vocab_size)
    history = detector.fit(X_train, y_train, X_val, y_val)

    # === 3. 测试集评估 ===
    logger.info("开始在测试集上评估...")
    metrics = detector.evaluate(X_test, y_test)
    logger.info(f"测试集评估结果: {metrics}")

    # === 4. 完整保存所有产出物 ===
    output_dir = Path(config['evaluation']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    artifact_prefix = getattr(detector, 'artifact_prefix', 'lstm_ae')

    metrics_path = output_dir / f"{artifact_prefix}_evaluation_metrics.npz"
    np.savez(metrics_path, **metrics)
    logger.info(f"评估指标已保存: {metrics_path}")
    
    history_path = output_dir / f"{artifact_prefix}_training_history.npz"
    np.savez(history_path, **history)
    logger.info(f"训练历史已保存: {history_path}")
    
    model_path = output_dir / f"{artifact_prefix}_model.pt"
    torch.save(detector.model.state_dict(), model_path)
    logger.info(f"LSTM 模型已保存: {model_path}")

    # === 5. 保存训练摘要，便于复现实验 ===
    summary_path = log_dir / f"{run_id}_{artifact_prefix}_training_summary.json"
    summary = {
        'run_id': run_id,
        'config_path': config_path,
        'paths': {
            'output_dir': str(output_dir),
            'log_dir': str(log_dir),
            'metrics_path': str(metrics_path),
            'history_path': str(history_path),
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
        'lstm_ae': config.get('lstm_ae', {}),
        'lstm_mode': lstm_mode,
        'metrics': metrics,
        'history': history,
        'calibration_info': getattr(detector, 'calibration_info', None),
        'threshold': float(detector.threshold) if detector.threshold is not None else None,
        'artifact_prefix': artifact_prefix,
    }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"训练摘要已保存: {summary_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM anomaly detector")
    parser.add_argument("--config", type=str, default="configs/experiment_config.yaml")
    args = parser.parse_args()
    main(args.config)