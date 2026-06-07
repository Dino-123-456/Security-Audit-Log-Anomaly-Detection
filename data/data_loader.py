# data/data_loader.py
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class BGLDataLoader:
    """BGL日志数据加载器，负责解析、窗口切分、缓存读写"""
    def __init__(self, config: dict):
        self.config = config['data']
        self.raw_path = Path(self.config['raw_path'])
        self.processed_dir = Path(self.config['processed_dir'])
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        self.window_size = self.config['window_size']
        self.step_size = self.config['step_size']
        self.train_ratio = self.config['train_ratio']
        self.val_ratio = self.config['val_ratio']
        self.seed = self.config['random_seed']

    @property
    def cache_path(self) -> Path:
        # 【修改】移除 _dual 后缀，回归最纯粹的缓存命名
        return self.processed_dir / (
            f"BGL_structured_ws{self.window_size}"
            f"_ss{self.step_size}"
            f"_tr{self.train_ratio}"
            f"_vr{self.val_ratio}_shuffled_cache.npz"
        )

    def load(self) -> Dict[str, np.ndarray]:
        """加载数据，优先读缓存；缓存缺失或过期时自动重建"""
        if self.cache_path.exists():
            logger.info(f"[DataLoader] 从缓存加载: {self.cache_path}")
            data = np.load(self.cache_path, allow_pickle=True)
            
            # 【修改】校验缓存完整性 (仅包含单标签 y)
            required_keys = {
                'X_train', 'y_train',
                'X_val', 'y_val',
                'X_test', 'y_test', 
                'vocab_size',
                'window_failure_rate'
            }
            missing = required_keys - set(data.files)
            if missing:
                logger.warning(f"[DataLoader] 缓存缺少字段 {missing}，将重新生成")
                self.cache_path.unlink()
            else:
                return dict(data)

        logger.info("[DataLoader] 缓存不存在或不完整，开始构建...")
        return self._build_and_cache()

    def _build_and_cache(self) -> Dict[str, np.ndarray]:
        """从原始CSV构建窗口数据并写入缓存"""
        df = pd.read_csv(self.raw_path)
        
        event_ids = df['EventId'].astype(str).values
        labels = df['Label'].apply(lambda x: 1 if x != '-' else 0).values.astype(np.int8)
        
        unique_events = sorted(set(event_ids))
        event2idx = {e: i + 1 for i, e in enumerate(unique_events)}
        vocab_size = len(unique_events) + 1  # 0 留给 padding
        
        sequences, seq_labels = [], []
        window_failure_rates = []
        
        for start in range(0, len(event_ids) - self.window_size + 1, self.step_size):
            window_events = event_ids[start:start + self.window_size]
            window_labels = labels[start:start + self.window_size]
            
            # 【统一标签】窗口内任一异常即为 1 (完美契合 Masked+MaxPooling 及所有传统模型)
            label = int(window_labels.max())
            
            seq = [event2idx.get(e, 0) for e in window_events]
            sequences.append(seq)
            seq_labels.append(label)
            # 窗口内的异常率（基于原始行级标签计算），用于特征工程
            failure_rate = window_labels.mean()
            window_failure_rates.append(failure_rate)
            
        X = np.array(sequences, dtype=np.int32)
        y = np.array(seq_labels, dtype=np.int8)
        window_failure_rates = np.array(window_failure_rates, dtype=np.float32)

        # 全局 Shuffle
        n = len(X)
        rng = np.random.default_rng(self.seed)
        indices = rng.permutation(n)
        X = X[indices]
        y = y[indices]
        window_failure_rates = window_failure_rates[indices]

        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))
        
        result = {
            'X_train': X[:train_end], 'y_train': y[:train_end],
            'X_val': X[train_end:val_end], 'y_val': y[train_end:val_end],
            'X_test': X[val_end:], 'y_test': y[val_end:],
            'vocab_size': np.array(vocab_size),
            'window_failure_rate_train': window_failure_rates[:train_end],
            'window_failure_rate_val': window_failure_rates[train_end:val_end],
            'window_failure_rate_test': window_failure_rates[val_end:]
        }
        
        np.savez_compressed(self.cache_path, **result)
        logger.info(f"[DataLoader] 缓存已保存: {self.cache_path} | vocab_size={vocab_size}")
        logger.info(f"[DataLoader] 全局异常比例: {y.mean():.2%}")
        
        return result