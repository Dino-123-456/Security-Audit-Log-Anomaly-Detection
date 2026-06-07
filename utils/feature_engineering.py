# utils/feature_engineering.py
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, Tuple, Optional
from scipy import sparse


class LogFeatureEngineer:
    """
    通用的日志特征工程器，用于生成 TF-IDF(n-gram) 特征并可选附加统计特征：熵、窗口异常率。

    配置示例：
      {
        'ngram_range': (1,2),
        'max_features': 800,
        'sublinear_tf': True,
        'use_entropy': True,
        'use_failure_rate': True
      }
    """

    def __init__(self, config: Dict):
        cfg = config or {}
        self.ngram_range = tuple(cfg.get('ngram_range', (1, 2)))
        self.max_features = int(cfg.get('max_features', 800))
        self.sublinear_tf = bool(cfg.get('sublinear_tf', True))
        self.use_entropy = bool(cfg.get('use_entropy', False))
        self.use_failure_rate = bool(cfg.get('use_failure_rate', False))

        self.vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            sublinear_tf=self.sublinear_tf,
            token_pattern=r'(?u)\b\w+\b'
        )
        self._is_fitted = False

    @staticmethod
    def _compute_entropy_from_sequence_window(window: np.ndarray) -> float:
        # window: 1D array of token ids (ints or strings)
        vals, counts = np.unique(window, return_counts=True)
        probs = counts / counts.sum()
        # 使用自然对数
        return float(-(probs * np.log(probs + 1e-12)).sum())

    def fit_transform(self, X_event_ids: np.ndarray, window_failure_rate: Optional[np.ndarray] = None):
        """
        在训练集上拟合并转换为数值特征矩阵。

        Args:
            X_event_ids: shape (n_samples, window_size) 的 int/string 矩阵
            window_failure_rate: 可选的 shape (n_samples,) 的浮点数组
        Returns:
            Sparse matrix shape (n_samples, n_features)
        """
        docs = [' '.join(map(str, window)) for window in X_event_ids]
        X_tfidf = self.vectorizer.fit_transform(docs)
        self._is_fitted = True

        extras = []
        if self.use_entropy:
            ent = np.array([self._compute_entropy_from_sequence_window(w) for w in X_event_ids], dtype=np.float32)
            extras.append(sparse.csr_matrix(ent.reshape(-1, 1)))

        if self.use_failure_rate:
            if window_failure_rate is None:
                raise RuntimeError('feature_engineering configured to use failure_rate but none provided')
            extras.append(sparse.csr_matrix(window_failure_rate.reshape(-1, 1)))

        if extras:
            X_extra = sparse.hstack(extras, format='csr')
            return sparse.hstack([X_tfidf, X_extra], format='csr')
        return X_tfidf

    def transform(self, X_event_ids: np.ndarray, window_failure_rate: Optional[np.ndarray] = None):
        if not self._is_fitted:
            raise RuntimeError("必须先调用 fit_transform() 再调用 transform()")
        docs = [' '.join(map(str, window)) for window in X_event_ids]
        X_tfidf = self.vectorizer.transform(docs)

        extras = []
        if self.use_entropy:
            ent = np.array([self._compute_entropy_from_sequence_window(w) for w in X_event_ids], dtype=np.float32)
            extras.append(sparse.csr_matrix(ent.reshape(-1, 1)))
        if self.use_failure_rate:
            if window_failure_rate is None:
                raise RuntimeError('feature_engineering configured to use failure_rate but none provided')
            extras.append(sparse.csr_matrix(window_failure_rate.reshape(-1, 1)))

        if extras:
            X_extra = sparse.hstack(extras, format='csr')
            return sparse.hstack([X_tfidf, X_extra], format='csr')
        return X_tfidf

    def get_feature_names(self) -> list:
        names = list(self.vectorizer.get_feature_names_out())
        if self.use_entropy:
            names.append('entropy')
        if self.use_failure_rate:
            names.append('window_failure_rate')
        return names