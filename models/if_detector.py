# models/if_detector.py
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class IFDetector:
    """Isolation Forest 异常检测器，支持 Bigram 特征 + 阈值校准"""
    
    def __init__(self, config: dict):
        self.config = config
        if_cfg = config['if_model']
        
        # 【核心改造】强制 IF 在建树时进行特征采样，避免被少数高频特征主导
        # max_features=0.8 表示每棵树随机选择 80% 的特征
        max_feat_param = if_cfg.get('max_features', 0.8)
        n_jobs_param = int(if_cfg.get('n_jobs', 1))
        if n_jobs_param <= 0:
            n_jobs_param = 1
        
        self.model = IsolationForest(
            n_estimators=if_cfg['n_estimators'],
            max_samples=if_cfg['max_samples'],
            contamination=if_cfg['contamination'],
            max_features=max_feat_param,  # 特征采样
            bootstrap=if_cfg['bootstrap'],
            n_jobs=n_jobs_param,
            random_state=config['data']['random_seed']
        )
        self.threshold: Optional[float] = None
        self.threshold_strategy: str = ""
        self.calibration_info: Dict[str, float] = {}
        
        logger.info(
            f"IF 初始化: estimators={if_cfg['n_estimators']}, "
            f"samples={if_cfg['max_samples']}, max_features={max_feat_param}, n_jobs={n_jobs_param}"
        )

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """返回 IsolationForest 的连续分数（越大越正常）。"""
        return self.model.score_samples(X)

    def save_artifact(self, path: str, feature_engineering: object = None) -> None:
        """
        将 detector 与可选的 feature_engineering 一并持久化，便于后续一键加载。
        保存格式为一个包含 {'detector': self, 'feature_engineering': feature_engineering} 的 pickle。
        """
        import joblib
        artifact = {'detector': self, 'feature_engineering': feature_engineering}
        joblib.dump(artifact, path)
        logger.info(f"IF artifact 已保存: {path}")

    @classmethod
    def load_artifact(cls, path: str):
        """从 `save_artifact` 保存的文件中恢复 (detector, feature_engineering)。"""
        import joblib
        artifact = joblib.load(path)
        detector = artifact.get('detector')
        fe = artifact.get('feature_engineering')
        return detector, fe

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray) -> None:
        """训练 IF 模型并在验证集上校准异常阈值"""
        logger.info(f"训练 Isolation Forest: {X_train.shape}")
        self.model.fit(X_train)

        # 优先在验证集上按“高召回约束下最大化精度/ F1”的方式搜索阈值；若验证集不含双类样本，则退回正常样本百分位阈值
        calibration_cfg = self.config.get('if_model', {})
        min_recall = float(calibration_cfg.get('calibration_min_recall', 0.98))
        optimize_mode = calibration_cfg.get('calibration_mode', 'precision_at_recall_floor')

        val_scores = self.model.score_samples(X_val)
        if len(np.unique(y_val)) < 2:
            normal_mask = y_val == 0
            if normal_mask.sum() == 0:
                logger.warning("验证集中无正常样本，使用默认阈值 0")
                self.threshold = 0.0
                self.threshold_strategy = "fallback_default"
            else:
                normal_scores = val_scores[normal_mask]
                self.threshold = float(np.percentile(normal_scores, 5))
                self.threshold_strategy = "fallback_normal_percentile_5"
                self.calibration_info = {
                    'strategy': self.threshold_strategy,
                    'threshold': self.threshold,
                    'min_recall': min_recall,
                }
                logger.info(
                    f"阈值校准完成: {self.threshold:.4f} "
                    f"(5th percentile of normal val scores, "
                    f"mean={normal_scores.mean():.4f}, std={normal_scores.std():.4f})"
                )
            return

        anomaly_scores = -val_scores  # 越大越异常
        precision, recall, thresholds = precision_recall_curve(y_val, anomaly_scores)
        if thresholds.size == 0:
            self.threshold = float(np.percentile(val_scores[y_val == 0], 5)) if np.any(y_val == 0) else 0.0
            self.threshold_strategy = "fallback_no_thresholds"
            self.calibration_info = {
                'strategy': self.threshold_strategy,
                'threshold': self.threshold,
                'min_recall': min_recall,
            }
            logger.warning("PR 曲线未产生阈值，回退到百分位阈值")
            return

        f1_scores = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-12)
        candidate_mask = recall[:-1] >= min_recall

        if optimize_mode == 'f1_at_recall_floor' and np.any(candidate_mask):
            candidate_indices = np.where(candidate_mask)[0]
            candidate_best = candidate_indices[int(np.argmax(f1_scores[candidate_indices]))]
            strategy_desc = f"val F1 max with recall>={min_recall:.3f}"
        elif np.any(candidate_mask):
            candidate_indices = np.where(candidate_mask)[0]
            # 优先提升 precision，其次 F1，最终更少误报
            candidate_precision = precision[candidate_indices]
            candidate_f1 = f1_scores[candidate_indices]
            best_local = np.lexsort((candidate_f1, candidate_precision))[-1]
            candidate_best = int(candidate_indices[best_local])
            strategy_desc = f"precision max with recall>={min_recall:.3f}"
        else:
            # 若没有任何阈值能达到召回下限，则退回到全局 F1 最优
            candidate_best = int(np.argmax(f1_scores))
            strategy_desc = "global F1 max fallback"

        best_anomaly_threshold = float(thresholds[candidate_best])
        self.threshold = -best_anomaly_threshold
        self.threshold_strategy = optimize_mode if np.any(candidate_mask) else "validation_f1_max_fallback"

        best_precision = float(precision[candidate_best])
        best_recall = float(recall[candidate_best])
        best_f1 = float(f1_scores[candidate_best])
        self.calibration_info = {
            'strategy': self.threshold_strategy,
            'threshold': self.threshold,
            'precision': best_precision,
            'recall': best_recall,
            'f1': best_f1,
            'min_recall': min_recall,
        }
        logger.info(
            f"阈值校准完成: {self.threshold:.4f} "
            f"({strategy_desc}, precision={best_precision:.4f}, recall={best_recall:.4f}, f1={best_f1:.4f})"
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """根据校准阈值返回异常预测标签"""
        if self.threshold is None:
            raise RuntimeError("Model not fitted. Call fit() before predict().")
        scores = self.model.score_samples(X)
        return (scores < self.threshold).astype(int)

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """在测试集上计算评估指标"""
        logger.info("开始在测试集上评估 IF 模型...")
        scores = self.model.score_samples(X)
        y_pred = (scores < self.threshold).astype(int)
        
        metrics = {
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
        }
        
        # AUC 需要连续分数，若全为同一预测值则无法计算
        try:
            # 取负使高分=高异常 (score_samples 越小越异常)
            metrics['auc'] = roc_auc_score(y_true, -scores)  
        except ValueError:
            metrics['auc'] = 0.0
            logger.warning("AUC 计算失败（预测值无变化），设为 0.0")

        # 输出分数分布诊断信息
        normal_scores = scores[y_true == 0]
        anomaly_scores = scores[y_true == 1]
        logger.info(f"  正常样本 IF 分数: mean={normal_scores.mean():.4f}, std={normal_scores.std():.4f}")
        logger.info(f"  异常样本 IF 分数: mean={anomaly_scores.mean():.4f}, std={anomaly_scores.std():.4f}")
        logger.info(f"  分数差值 (正常-异常): {normal_scores.mean() - anomaly_scores.mean():.4f}")

        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}")
            
        return metrics