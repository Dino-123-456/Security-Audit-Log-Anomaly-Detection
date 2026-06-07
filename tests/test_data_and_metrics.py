# test_foundation.py
from data.data_loader import LogDataLoader
from utils.metrics import evaluate, measure_inference_time
import numpy as np

# 1. 验证数据加载与划分
loader = LogDataLoader()
data = loader.load_and_split()

print("\n=== 数据划分统计 ===")
for k, v in data['metadata'].items():
    print(f"  {k}: {v}")

# 2. 验证划分可复现性
data2 = LogDataLoader().load_and_split()
assert np.array_equal(data['X_train'], data2['X_train']), "❌ 划分不可复现!"
print("\n✅ 划分可复现性验证通过")

# 3. 验证标签语义
assert set(np.unique(data['y_train'])).issubset({0, 1}), "❌ 标签包含非0/1值!"
assert set(np.unique(data['y_test'])).issubset({0, 1}), "❌ 测试集标签包含非0/1值!"
print("✅ 标签语义验证通过 (仅含0和1)")

# 4. 验证评估函数
dummy_pred = np.zeros_like(data['y_test'])
metrics = evaluate(data['y_test'], dummy_pred)
print(f"\n=== 评估函数验证 (全0预测) ===")
print(f"  {metrics}")
assert metrics['precision'] == 0 and metrics['recall'] == 0, "❌ 评估函数异常!"
print("✅ 评估函数验证通过")

# 5. 验证耗时测量
dummy_fn = lambda x: x
t = measure_inference_time(dummy_fn, np.zeros(100))
print(f"\n✅ 推理耗时测量验证通过: {t:.6f}s")

print("\n🎉 地基三件套全部验收通过!")