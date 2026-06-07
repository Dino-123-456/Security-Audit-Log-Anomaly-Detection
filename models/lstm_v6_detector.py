# models/lstm_v6_detector.py
"""
LSTM v6: 更强的因果序列模型。

核心思想：
- 单向因果建模，只看历史，不看未来
- 使用 TransformerEncoder 替代 LSTM，提高长依赖建模能力
- Teacher forcing 训练 next-token prediction
- 推理时以平均 next-token NLL 作为序列异常分数
"""
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm

logger = logging.getLogger(__name__)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class CausalTransformerPredictor(nn.Module):
    """因果 Transformer next-token 预测器。"""

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int,
                 num_layers: int = 2, dropout: float = 0.2, num_heads: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.input_proj = nn.Linear(embedding_dim, hidden_dim) if embedding_dim != hidden_dim else nn.Identity()
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padding_mask = x.eq(0)
        embedded = self.embedding(x)
        hidden = self.input_proj(embedded)
        hidden = self.pos_encoding(hidden)
        attn_mask = self._causal_mask(x.size(1), x.device)
        encoded = self.encoder(hidden, mask=attn_mask, src_key_padding_mask=padding_mask)
        logits = self.fc(encoded)
        return logits


class LSTMAutoregressiveDetector:
    """兼容训练入口的 v6 检测器名称。"""

    def __init__(self, config: dict, vocab_size: int):
        self.config = config
        model_cfg = config['lstm_ae']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")

        self.model = CausalTransformerPredictor(
            vocab_size=vocab_size,
            embedding_dim=model_cfg['embedding_dim'],
            hidden_dim=model_cfg['hidden_dim'],
            num_layers=model_cfg['num_layers'],
            dropout=model_cfg['dropout'],
            num_heads=model_cfg.get('num_heads', 4),
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=model_cfg['learning_rate'],
            weight_decay=model_cfg.get('weight_decay', 0)
        )
        self.epochs = model_cfg['epochs']
        self.batch_size = model_cfg['batch_size']
        self.patience = model_cfg['early_stopping_patience']
        self.threshold_pct = model_cfg.get('threshold_pct', 95)

        self.threshold: Optional[float] = None
        self.calibration_info: Optional[Dict] = None
        self.mode = 'causal_transformer'
        self.artifact_prefix = 'lstm_v6'

        self.checkpoint_dir = Path(config['evaluation']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.checkpoint_dir / f"{self.artifact_prefix}_best.pt"

    def _prepare_batch(self, batch_x: torch.Tensor):
        inputs = batch_x[:, :-1]
        targets = batch_x[:, 1:]
        return inputs, targets

    def _compute_anomaly_score(self, batch_x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        inputs, targets = self._prepare_batch(batch_x)
        with torch.no_grad():
            logits = self.model(inputs)
            log_probs = F.log_softmax(logits, dim=-1)
        target_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        token_nll = -target_log_probs
        valid_mask = (targets != 0).float()
        valid_counts = valid_mask.sum(dim=1).clamp(min=1.0)
        return (token_nll * valid_mask).sum(dim=1) / valid_counts

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray) -> Dict:
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
            self.model.train()
            train_losses = []
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]", leave=False)
            for batch_x in train_pbar:
                batch_x = batch_x.to(self.device)
                inputs, targets = self._prepare_batch(batch_x)

                self.optimizer.zero_grad()
                logits = self.model(inputs)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    ignore_index=0
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()

                loss_val = loss.item()
                train_losses.append(loss_val)
                train_pbar.set_postfix({'loss': f"{loss_val:.4f}"})
            train_pbar.close()

            self.model.eval()
            val_losses = []
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Val]  ", leave=False)
            with torch.no_grad():
                for batch_x, _ in val_pbar:
                    batch_x = batch_x.to(self.device)
                    inputs, targets = self._prepare_batch(batch_x)
                    logits = self.model(inputs)
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        targets.reshape(-1),
                        ignore_index=0
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
                tqdm.write("  -> 验证集 Loss 下降，已保存最佳模型")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(f"触发 Early Stopping (patience={self.patience})，停止训练。")
                    break

        logger.info("加载最佳模型权重进行阈值校准...")
        state = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state)
        self._calibrate_threshold(val_loader)
        return history

    def _calibrate_threshold(self, val_loader):
        self.model.eval()
        all_scores = []
        all_labels = []
        cal_pbar = tqdm(val_loader, desc="Calibrating Threshold (v6 NLL)", leave=False)
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
            logger.warning("验证集中无数据，使用默认阈值 1.0")
            return

        scores = np.concatenate(all_scores)
        labels = np.concatenate(all_labels)

        normal_scores = scores[labels == 0]
        if normal_scores.size > 0:
            pct_threshold = float(np.percentile(normal_scores, self.threshold_pct))
        else:
            pct_threshold = float(np.median(scores))

        thr_min, thr_max = float(scores.min()), float(scores.max())
        thresholds = np.linspace(thr_min, thr_max, num=200)
        best_f1 = -1.0
        best_thr = pct_threshold
        for t in thresholds:
            preds = (scores > t).astype(int)
            f1 = f1_score(labels, preds, zero_division=0)
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
            'score_type': 'ar_nll',
            'mode': self.mode,
        }
        logger.info(
            f"阈值校准完成: chosen={self.threshold:.4f} (best F1={best_f1:.4f}), "
            f"pct_ref={pct_threshold:.4f}"
        )

    def score(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        loader = torch.utils.data.DataLoader(
            torch.tensor(X, dtype=torch.long), batch_size=self.batch_size
        )
        all_scores = []
        score_pbar = tqdm(loader, desc="Scoring (v6 NLL)", leave=False)
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
        logger.info("开始推理打分 (v6 Causal Transformer NLL)...")
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

        normal_scores = scores[y_true == 0]
        anomaly_scores = scores[y_true == 1]
        logger.info(f"  正常样本 NLL: mean={normal_scores.mean():.4f}, std={normal_scores.std():.4f}")
        logger.info(f"  异常样本 NLL: mean={anomaly_scores.mean():.4f}, std={anomaly_scores.std():.4f}")
        logger.info(f"  差值 (异常-正常): {anomaly_scores.mean() - normal_scores.mean():.4f}")

        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}")
        return metrics
