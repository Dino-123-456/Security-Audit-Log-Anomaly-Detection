import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any

logger = logging.getLogger(__name__)

class BGLDataLoader:
    """
    BGL 数据集加载器
    核心特性：严格时序划分 + 防词典泄露 (Vocab Leakage Prevention)
    """
    def __init__(self, config: Dict[str, Any]):
        data_cfg = config.get('data', {})
        
        self.raw_path = data_cfg.get('raw_path', 'data/BGL/BGL.log_structured.csv')
        self.cache_dir = data_cfg.get('cache_dir', 'data/cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 【修复】使用新的缓存文件名，强制废弃旧缓存
        self.cache_path = os.path.join(self.cache_dir, 'bgl_data_timesplit_unk.npz')
        
        self.window_size = data_cfg.get('window_size', 10)
        self.step_size = data_cfg.get('step_size', 1)
        self.train_ratio = data_cfg.get('train_ratio', 0.8)
        self.val_ratio = data_cfg.get('val_ratio', 0.1)
        self.seed = data_cfg.get('seed', 42)

    def load(self) -> Dict[str, np.ndarray]:
        """加载数据，优先读取缓存，若缓存缺失字段则自动更新"""
        required_fields = {
            'X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test', 
            'vocab_size', 'window_failure_rate_train', 'window_failure_rate_val', 'window_failure_rate_test'
        }

        if os.path.exists(self.cache_path):
            logger.info(f"[DataLoader] 发现缓存文件: {self.cache_path}")
            cache_data = dict(np.load(self.cache_path, allow_pickle=True))
            
            missing_fields = required_fields - set(cache_data.keys())
            if missing_fields:
                logger.warning(f"[DataLoader] 缓存缺少字段 {missing_fields}，将触发安全更新...")
                cache_data = self._update_cache(cache_data)
                
            logger.info("[DataLoader] 缓存加载成功 (严格时序划分 + UNK机制)。")
            return cache_data
        else:
            logger.info(f"[DataLoader] 未找到缓存，开始从原始 CSV 构建...")
            return self._build_and_cache()

    def _build_and_cache(self) -> Dict[str, np.ndarray]:
        """从原始CSV构建窗口数据并写入缓存 (防词典泄露版)"""
        if not os.path.exists(self.raw_path):
            raise FileNotFoundError(f"找不到原始数据文件: {self.raw_path}")
            
        df = pd.read_csv(self.raw_path)
        logger.info(f"[DataLoader] 读取原始日志: {len(df)} 条")
        
        event_ids = df['EventId'].astype(str).values
        labels = df['Label'].apply(lambda x: 1 if x != '-' else 0).values.astype(np.int8)
        
        raw_sequences, seq_labels = [], []
        window_failure_rates = []
        
        # 1. 严格按时间顺序滑动窗口 (先保留原始 EventId 字符串)
        for start in range(0, len(event_ids) - self.window_size + 1, self.step_size):
            window_events = event_ids[start:start + self.window_size]
            window_labels = labels[start:start + self.window_size]
            
            label = int(window_labels.max())
            raw_sequences.append(window_events)
            seq_labels.append(label)
            window_failure_rates.append(window_labels.mean())
            
        raw_X = np.array(raw_sequences, dtype=object)
        y = np.array(seq_labels, dtype=np.int8)
        window_failure_rates = np.array(window_failure_rates, dtype=np.float32)
        
        n = len(raw_X)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))
        
        # 【核心修复 1】仅使用训练集的窗口来构建词典！
        train_raw_X = raw_X[:train_end]
        # 展平训练集的所有 EventId 并去重
        train_unique_events = sorted(set(e for seq in train_raw_X for e in seq))
        
        # 0: <PAD>, 1: <UNK>, 2...: 正常 EventId
        event2idx = {e: i + 2 for i, e in enumerate(train_unique_events)}
        vocab_size = len(train_unique_events) + 2 
        
        logger.info(f"[DataLoader] 训练集词汇量: {len(train_unique_events)} | 总 vocab_size: {vocab_size}")
        
        # 2. 将所有数据映射为 Index，未知词映射为 1 (<UNK>)
        def map_to_idx(seq):
            return [event2idx.get(e, 1) for e in seq] # 1 是 <UNK>

        X = np.array([map_to_idx(seq) for seq in raw_X], dtype=np.int32)
        
        logger.info("="*50)
        logger.info(f"[DataLoader] ⚠️ 采用严格时序划分 + 防词典泄露，已禁用全局 Shuffle。")
        logger.info(f"[DataLoader] 训练集: 0 ~ {train_end} | 验证集: {train_end} ~ {val_end} | 测试集: {val_end} ~ {n}")
        
        # 统计测试集中的 UNK 比例，用于汇报时展示模型的泛化难度
        test_X = X[val_end:]
        total_test_tokens = test_X.size
        unk_test_tokens = np.sum(test_X == 1)
        logger.info(f"[DataLoader] 测试集中 <UNK> (未知日志) 比例: {unk_test_tokens / total_test_tokens:.2%}")
        logger.info("="*50)
        
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
        logger.info(f"[DataLoader] 缓存已保存: {self.cache_path}")
        logger.info(f"[DataLoader] 全局异常比例: {y.mean():.2%}")
        
        return result

    def _update_cache(self, existing_cache: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """安全更新缓存：补充缺失字段"""
        logger.info("[DataLoader] 开始重新计算缺失字段...")
        if not os.path.exists(self.raw_path):
            raise FileNotFoundError(f"更新缓存需要原始数据文件: {self.raw_path}")
            
        df = pd.read_csv(self.raw_path)
        labels = df['Label'].apply(lambda x: 1 if x != '-' else 0).values.astype(np.int8)
        
        window_failure_rates = []
        for start in range(0, len(labels) - self.window_size + 1, self.step_size):
            window_labels = labels[start:start + self.window_size]
            window_failure_rates.append(window_labels.mean())
            
        window_failure_rates = np.array(window_failure_rates, dtype=np.float32)
        
        n = len(window_failure_rates)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))
        
        existing_cache['window_failure_rate_train'] = window_failure_rates[:train_end]
        existing_cache['window_failure_rate_val'] = window_failure_rates[train_end:val_end]
        existing_cache['window_failure_rate_test'] = window_failure_rates[val_end:]
        
        np.savez_compressed(self.cache_path, **existing_cache)
        logger.info(f"[DataLoader] 缓存已更新并保存: {self.cache_path}")
        return existing_cache