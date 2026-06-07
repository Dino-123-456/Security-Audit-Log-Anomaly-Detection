# models/lstm_ae_detector.py
"""
LSTM 日志异常检测器 - 连续置信度打分范式
核心设计：
    - 全位置 Masked 预测（双向 LSTM 天然支持）
    - 默认使用 True Token Confidence (1-p) 作为连续异常分数
    - 可切换到 -log(p) 作为对数放大版本
    - 不做温度缩放，避免大词表下分布过度锐化
"""
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from typing import Dict, Optional
from tqdm import tqdm

logger = logging.getLogger(__name__)


class LSTMMaskedPredictor(nn.Module):
    """双向 LSTM，用于 Masked Token 预测"""
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.mask_idx = vocab_size - 1
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim * 2, vocab_size - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        logits = self.fc(lstm_out)
        return logits


class LSTMAEDetector:
    """LSTM 异常检测器 (Log-Confidence 打分范式)"""

    def __init__(self, config: dict, vocab_size: int):
        self.config = config
        model_cfg = config['lstm_ae']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")

        self.model = LSTMMaskedPredictor(
            vocab_size=vocab_size + 1,
            embedding_dim=model_cfg['embedding_dim'],
            hidden_dim=model_cfg['hidden_dim'],
            num_layers=model_cfg['num_layers'],
            dropout=model_cfg['dropout']
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=model_cfg['learning_rate'],
            weight_decay=model_cfg.get('weight_decay', 0)
        )
        self.epochs = model_cfg['epochs']
        self.batch_size = model_cfg['batch_size']
        self.patience = model_cfg['early_stopping_patience']

        # 打分参数
        self.threshold_pct = model_cfg.get('threshold_pct', 95)
        self.score_transform = model_cfg.get('score_transform', 'one_minus_p')
        logger.info(f"打分范式: {self.score_transform}, threshold_pct={self.threshold_pct}")

        self.threshold: Optional[float] = None
        self.calibration_info: Optional[Dict] = None

        self.checkpoint_dir = Path(config['evaluation']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.checkpoint_dir / "lstm_best.pt"

    # ==================== 训练用 Mask ====================
    def _create_masked_batch(self, batch_x: torch.Tensor, mask_ratio: float = 0.15):
        """随机遮蔽 15% 的 token，用于训练"""
        inputs = batch_x.clone()
        mask = torch.rand(inputs.shape, device=self.device) < mask_ratio
        mask = mask & (inputs != 0)

        targets = inputs.clone()
        targets[~mask] = -100

        inputs[mask] = self.model.mask_idx
        return inputs, targets

    # ==================== 核心：Log-Confidence 打分 ====================
    def _compute_anomaly_score(self, batch_x: torch.Tensor) -> torch.Tensor:
        """
        【核心】Log-Confidence 打分

        对每个非 padding 位置 i：
          1. 将位置 i 替换为 [MASK]，保留所有其他真实 token
          2. 双向 LSTM 利用完整上下文预测位置 i
          3. 取真实 token 的 softmax 概率 p
          4. 计算 -log(p + eps) 作为该位置的异常贡献

        返回每个样本的平均 -log(p)（越高越异常）

        与 v3 的区别：
          - 不做温度缩放 → 避免大词表下分布过度锐化
          - 全位置打分 → 最大化信噪比
          - eps=1e-10 → 防止 log(0)，同时不会引入极端值
        """
        self.model.eval()
        batch_size, seq_len = batch_x.shape

        valid_mask = (batch_x != 0)  # (B, L)
        valid_counts = valid_mask.sum(dim=1).float()  # (B,)

        positions = valid_mask.nonzero(as_tuple=False)  # (total_valid, 2)
        total_valid = positions.shape[0]
        if total_valid == 0:
            return torch.zeros(batch_size, device=self.device)

        # 为每个有效位置创建打分输入：将每个样本重复其有效位置次数
        flat_inputs = batch_x.repeat_interleave(valid_counts.long(), dim=0)
        row_indices = positions[:, 0]
        col_indices = positions[:, 1]

        # flat_inputs 的行顺序与 positions 保持一致（先是样本0的所有有效位置，再样本1的...）
        # 因此用顺序索引选取每一行对应的位置进行 mask
        seq_idx = torch.arange(total_valid, device=self.device)
        flat_inputs[seq_idx, col_indices] = self.model.mask_idx

        # 前向传播
        with torch.no_grad():
            logits = self.model(flat_inputs)  # (total_valid, seq_len, V-1)

        # 提取打分位置的 logits：使用顺序索引访问 flat_inputs 对应行
        logits_at_pos = logits[seq_idx, col_indices, :]  # (total_valid, V-1)

        # softmax → 取真实 token 的概率（不做温度缩放！）
        probs = F.softmax(logits_at_pos, dim=1)  # (total_valid, V-1)
        true_tokens = batch_x[row_indices, col_indices] - 1  # 1-indexed → 0-indexed
        true_probs = probs.gather(1, true_tokens.unsqueeze(1)).squeeze(1)  # (total_valid,)

        eps = 1e-10
        if self.score_transform == 'neg_log_p':
            # -log(p + eps)
            pos_scores = -torch.log(true_probs + eps)
        elif self.score_transform == 'one_minus_p':
            pos_scores = 1.0 - true_probs
        else:
            # fallback: neg_log_p
            pos_scores = -torch.log(true_probs + eps)

        # 汇总到每个样本：计算平均 token-level 分数
        sample_scores = torch.zeros(batch_size, device=self.device)
        sample_scores.scatter_add_(0, row_indices, pos_scores)
        avg_scores = sample_scores / (valid_counts + 1e-8)

        return avg_scores

    # ==================== 训练流程 ====================
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """训练模型"""
        train_loader = torch.utils.data.DataLoader(
            torch.tensor(X_train, dtype=torch.long),
            batch_size=self.batch_size, shuffle=True
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_val, dtype=torch.long),
            torch.tensor(y_val, dtype=torch.float)
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False
        )

        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(self.epochs):
            # === 训练阶段 ===
            self.model.train()
            train_losses = []
            train_pbar = tqdm(train_loader,
                              desc=f"Epoch {epoch+1}/{self.epochs} [Train]", leave=False)
            for batch_x in train_pbar:
                batch_x = batch_x.to(self.device)
                inputs, targets = self._create_masked_batch(batch_x, mask_ratio=0.15)

                self.optimizer.zero_grad()
                logits = self.model(inputs)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    ignore_index=-100
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()

                loss_val = loss.item()
                train_losses.append(loss_val)
                train_pbar.set_postfix({'loss': f"{loss_val:.4f}"})
            train_pbar.close()

            # === 验证阶段 ===
            self.model.eval()
            val_losses = []
            val_pbar = tqdm(val_loader,
                            desc=f"Epoch {epoch+1}/{self.epochs} [Val]  ", leave=False)
            with torch.no_grad():
                for batch_x, _ in val_pbar:
                    batch_x = batch_x.to(self.device)
                    inputs, targets = self._create_masked_batch(batch_x, mask_ratio=0.15)
                    logits = self.model(inputs)
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        targets.reshape(-1),
                        ignore_index=-100
                    )
                    val_losses.append(loss.item())
            val_pbar.close()

            avg_train = np.mean(train_losses)
            avg_val = np.mean(val_losses)
            history['train_loss'].append(avg_train)
            history['val_loss'].append(avg_val)

            tqdm.write(
                f"[Epoch {epoch+1}/{self.epochs}] "
                f"Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}"
            )

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                patience_counter = 0
                torch.save(self.model.state_dict(), self.checkpoint_path)
                tqdm.write(f"  -> 验证集 Loss 下降，已保存最佳模型")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(f"触发 Early Stopping (patience={self.patience})，停止训练。")
                    break

        logger.info(f"加载最佳模型权重进行阈值校准...")
        # 使用 map_location 以兼容 CPU/GPU 环境，避免错误参数传递
        state = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state)
        self._calibrate_threshold(val_loader)
        return history

    # ==================== 阈值校准 ====================
    def _calibrate_threshold(self, val_loader):
        """在验证集上计算分数与标签；保存校准信息并基于验证集搜索最佳 F1 阈值。

        同时保留基于正常样本百分位的参考阈值以供对比。
        """
        self.model.eval()
        all_scores = []
        all_labels = []
        cal_pbar = tqdm(val_loader, desc="Calibrating Threshold", leave=False)
        with torch.no_grad():
            for batch_x, batch_y in cal_pbar:
                batch_x = batch_x.to(self.device)
                scores = self._compute_anomaly_score(batch_x)
                all_scores.append(scores.cpu().numpy())
                all_labels.append(batch_y.numpy())
        cal_pbar.close()

        if not all_scores:
            self.threshold = 1.0
            self.calibration_info = {'note': 'no validation data, default threshold set'}
            logger.warning("验证集为空，使用默认阈值 1.0")
            return

        scores = np.concatenate(all_scores)
        labels = np.concatenate(all_labels)

        # 参考：基于正常样本的百分位阈值
        normal_scores = scores[labels == 0]
        if normal_scores.size > 0:
            pct_threshold = float(np.percentile(normal_scores, self.threshold_pct))
        else:
            pct_threshold = float(np.median(scores))

        # 在验证集上按阈值网格搜索最佳 F1
        thr_min, thr_max = float(scores.min()), float(scores.max())
        thresholds = np.linspace(thr_min, thr_max, num=200)
        best_f1 = -1.0
        best_thr = pct_threshold
        f1s = []
        for t in thresholds:
            preds = (scores > t).astype(int)
            f1 = f1_score(labels, preds, zero_division=0)
            f1s.append(float(f1))
            if f1 > best_f1:
                best_f1 = float(f1)
                best_thr = float(t)

        self.threshold = best_thr
        self.calibration_info = {
            'threshold_pct': self.threshold_pct,
            'threshold_pct_value': pct_threshold,
            'best_threshold': best_thr,
            'best_f1': best_f1,
            'threshold_search': {
                'min': thr_min,
                'max': thr_max,
                'n_steps': len(thresholds)
            },
            'score_transform': self.score_transform,
        }

        logger.info(
            f"阈值校准完成: chosen={self.threshold:.4f} (best F1={best_f1:.4f}), "
            f"pct_ref={pct_threshold:.4f}"
        )

    # ==================== 打分与预测 ====================
    def score(self, X: np.ndarray) -> np.ndarray:
        """返回每个样本的平均 -log(p)（越高越异常）"""
        self.model.eval()
        loader = torch.utils.data.DataLoader(
            torch.tensor(X, dtype=torch.long), batch_size=self.batch_size
        )
        all_scores = []
        score_pbar = tqdm(loader, desc="Scoring (Log-Conf)", leave=False)
        with torch.no_grad():
            for batch_x in score_pbar:
                batch_x = batch_x.to(self.device)
                scores = self._compute_anomaly_score(batch_x)
                all_scores.append(scores.cpu().numpy())
        score_pbar.close()
        return np.concatenate(all_scores)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.threshold is None:
            raise RuntimeError("Model not fitted. Call fit() before predict().")
        scores = self.score(X)
        return (scores > self.threshold).astype(int)

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """完整评估"""
        logger.info(f"开始推理打分 (Log-Confidence)...")
        scores = self.score(X)
        y_pred = (scores > self.threshold).astype(int)

        metrics = {
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
        }
        try:
            metrics['auc'] = roc_auc_score(y_true, scores)
        except ValueError:
            metrics['auc'] = 0.0
            logger.warning("AUC 计算失败（预测值无变化），设为 0.0")

        # 分数分布诊断
        normal_scores = scores[y_true == 0]
        anomaly_scores = scores[y_true == 1]
        logger.info(f"  正常样本 Log-Conf: mean={normal_scores.mean():.4f}, std={normal_scores.std():.4f}")
        logger.info(f"  异常样本 Log-Conf: mean={anomaly_scores.mean():.4f}, std={anomaly_scores.std():.4f}")
        logger.info(f"  差值 (异常-正常): {anomaly_scores.mean() - normal_scores.mean():.4f}")

        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}")
        return metrics