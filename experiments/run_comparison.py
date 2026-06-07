# experiments/run_comparison.py
import argparse
import json
import csv
import torch
from pathlib import Path
import yaml

from data.data_loader import BGLDataLoader
from models.lstm_ae_detector import LSTMAEDetector
from utils.metrics import evaluate, measure_inference_time
from utils.logger import get_logger
from joblib import load as joblib_load


def _seqs_to_docs(X_seq):
    return [" ".join(map(str, row)) for row in X_seq]



def main(config_path: str):
    logger = get_logger("run_comparison")
    # 加载配置与数据
    from pathlib import Path
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f) if config_path.endswith('.json') else yaml.safe_load(open(config_path, 'r', encoding='utf-8'))

    loader = BGLDataLoader(config)
    data = loader.load()

    results = []

    # === Route A: IF ===
    logger.info("评估 Route A (IF)...")
    # 兼容 train_if.py 的输出位置 (outputs/results)
    out_dir = Path(config['evaluation']['output_dir'])
    vec_path = out_dir / 'if_tfidf_vectorizer.pkl'
    model_path = out_dir / 'if_model.pkl'
    if not vec_path.exists() or not model_path.exists():
        logger.error(f"无法找到 IF artifact: {vec_path} 或 {model_path}")
    else:
        vectorizer = joblib_load(str(vec_path))
        if_model = joblib_load(str(model_path))

        X_test_docs = _seqs_to_docs(data['X_test'])
        X_test_tfidf = vectorizer.transform(X_test_docs)

        # IsolationForest.predict 返回 {-1,1}，-1 表示异常
        y_pred_if = (if_model.predict(X_test_tfidf) == -1).astype(int)
        y_scores_if = if_model.score_samples(X_test_tfidf)

        metrics_if = evaluate(data['y_test'], y_pred_if, y_scores_if, higher_is_more_anomalous=False)
        metrics_if['inference_time'] = measure_inference_time(lambda X: (if_model.predict(X) == -1).astype(int), X_test_tfidf)
        metrics_if['model'] = 'IF+BigramTFIDF'
        results.append(metrics_if)
        logger.info(f"  IF => F1={metrics_if['f1']:.4f}, AUC={metrics_if['auc']:.4f}")

    # === Route B: LSTM-AE ===
    logger.info("评估 Route B (LSTM-AE)...")
    lstm_weights = Path(config['evaluation']['output_dir']) / 'lstm_ae_model.pt'
    if not lstm_weights.exists():
        logger.error(f"无法找到 LSTM 权重: {lstm_weights}")
    else:
        vocab_size = int(data['vocab_size'])
        lstm_detector = LSTMAEDetector(config, vocab_size=vocab_size)
        lstm_detector.model.load_state_dict(torch.load(str(lstm_weights), map_location=lstm_detector.device))

        # 基于验证集校准阈值（train_lstm_ae 中的做法）
        val_dataset = torch.utils.data.TensorDataset(
            torch.tensor(data['X_val'], dtype=torch.long),
            torch.tensor(data['y_val'], dtype=torch.float)
        )
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=lstm_detector.batch_size, shuffle=False)
        lstm_detector._calibrate_threshold(val_loader)

        y_pred_lstm = lstm_detector.predict(data['X_test'])
        y_scores_lstm = lstm_detector.score(data['X_test'])
        metrics_lstm = evaluate(data['y_test'], y_pred_lstm, y_scores_lstm, higher_is_more_anomalous=True)
        metrics_lstm['inference_time'] = measure_inference_time(lstm_detector.predict, data['X_test'])
        metrics_lstm['model'] = 'LSTM-AE'
        results.append(metrics_lstm)
        logger.info(f"  LSTM-AE => F1={metrics_lstm['f1']:.4f}, AUC={metrics_lstm['auc']:.4f}")

    # === 保存结构化结果 ===
    out_dir = Path("outputs/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "comparison_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=['model', 'precision', 'recall', 'f1', 'auc', 'inference_time'])
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"对比结果已保存: {csv_path}")

    # 控制台汇总
    print("\n" + "=" * 55)
    print(f"{'Model':<12} {'P':>6} {'R':>6} {'F1':>6} {'AUC':>6} {'Time(s)':>8}")
    print("-" * 55)
    for r in results:
        print(f"{r['model']:<12} {r['precision']:>6.4f} {r['recall']:>6.4f} "
              f"{r['f1']:>6.4f} {r['auc']:>6.4f} {r['inference_time']:>8.6f}")
    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_config.yaml")
    args = parser.parse_args()
    main(args.config)