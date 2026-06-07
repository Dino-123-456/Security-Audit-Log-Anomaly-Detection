# Security Audit Log Anomaly Detection

## 项目简介
本项目面向 BGL 超算日志数据集，研究安全审计日志的异常检测问题。核心任务是将原始日志通过 Drain 解析为 Event ID 序列，再按滑动窗口切分为二分类样本，比较传统机器学习路线与深度序列建模路线在日志异常检测中的表现。

当前项目已经完成数据管道、模型训练、评估与结果落盘流程，并在 `outputs/logs` 中保存了可复现实验的训练摘要。项目经历了 **6 个版本的 LSTM 打分范式演进**，沉淀了针对“大词表+短序列”场景的深度实验复盘。

## 项目目标
- 将原始 BGL 日志转换为结构化 Event 序列
- 基于滑动窗口构造二分类样本，采用 Any 标签策略：窗口内只要包含任一异常事件，就记为异常
- 对比两条检测路线：
  - 传统机器学习：Isolation Forest
  - 深度学习：LSTM / Transformer 序列模型
- 保存训练过程、模型产物和实验摘要，便于复现实验
- 为后续改进 DeepLog、LogBERT 或其他更强时序模型提供实验基线与避坑指南

## 环境准备
本项目基于 Windows + WSL2 (Ubuntu) + VS Code + Python venv。

### 1. 创建虚拟环境
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. 安装依赖
```bash
pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  --trusted-host pypi.tuna.tsinghua.edu.cn \
  --trusted-host download.pytorch.org

pip install numpy pandas scikit-learn matplotlib seaborn tqdm drain3 pyyaml joblib scipy
```

### 3. 推荐运行方式
建议始终通过虚拟环境解释器执行训练脚本：
```bash
.venv/bin/python3 -m experiments.train_if --config configs/experiment_config.yaml
.venv/bin/python3 -m experiments.train_lstm_ae --config configs/experiment_config.yaml
.venv/bin/python3 -m experiments.run_comparison --config configs/experiment_config.yaml
```

## 数据流程
- `scripts/parse_bgl.py` 使用 Drain3 将原始 BGL 日志解析为结构化 CSV
- `data/data_loader.py` 将 Event ID 序列按窗口大小切分，并生成 Any 标签
- 数据会自动缓存到 `data/processed/`，避免重复构建
- 训练阶段会读取缓存并保存新的训练摘要到 `outputs/logs/`

**当前窗口配置**
- Window Size: 10
- Step Size: 5
- Train / Val / Test: 0.6 / 0.2 / 0.2

## 方法设计

### 路线选择与对比策略
结合现有参考文献与本项目的任务背景，更合理的策略不是只选一条路线，而是采用“双路线并行、统一对比”的实验组织方式：
- Isolation Forest 更适合作为轻量、可解释、易复现的传统基线，便于验证特征工程是否有效，参考文献中的相关工作也支持这一定位，见 [1]、[2]、[6]、[10]
- LSTM 及其他深度序列模型更适合承接“日志事件顺序建模”的主线，相关综述与方法论文表明深度学习路线更适合复杂时序场景，见 [3]、[4]、[5]、[7]、[8]、[9]

因此，本项目采用“IF 作为基线、LSTM 作为序列建模主线”的双轨方案，而不是把二者视为互斥选项。

对比时建议统一关注以下维度：
- 检测效果：Precision、Recall、F1、AUC
- 代价开销：训练时间、推理时间、内存占用
- 工程可用性：解释性、复现难度、对词表规模和特征表示的敏感性

### 路线 A: Isolation Forest
当前 IF 路线不是简单的纯 TF-IDF，而是一个更稳定的复合特征方案：
- Bigram TF-IDF，保留局部时序关系
- 熵特征，用于反映窗口内事件分布复杂度
- failure rate 特征，用于反映窗口内异常事件占比
- 稀疏矩阵端到端传递，避免 WSL2 下的内存峰值
- 阈值校准采用验证集策略，而不是固定百分位硬阈值

### 路线 B: LSTM / Transformer (序列建模主线)
LSTM 路线作为序列建模主线，负责捕捉事件顺序和上下文依赖。基于完整的 v1–v6 实验对比，我们经历了从“离散阈值”到“连续概率”，从“双向重构”到“单向预测”的完整探索。

## LSTM 实验对比（v1–v6）

下表汇总了在当前数据与设置下对几种 LSTM/Transformer 打分/聚合策略的测试结果与关键发现：

| 版本 | 核心范式 | AUC | F1 | 关键发现 / 致命缺陷 |
| :--- | :--- | :--- | :--- | :--- |
| **v1** | Top‑K (K=5) | 0.63 | 0.08 | 词表≈843，K=5 命中率极低，几乎全部 miss（Top‑K 在大词表下不可行） |
| **v2** | True Token Confidence (1-p) | 0.73 | 0.16 | 模型学到区分能力；概率被压缩到 0.99x 窄带，需放大/校准以便阈值敏感化 |
| **v3** | NLL + 温度 (temp=0.5) | 0.50 | 0.03 | 温度过激导致 Perplexity 爆炸，正常样本也被极端惩罚，方法不稳健 |
| **v4** | Log‑Confidence (-log p) | 0.66 | 0.04 | 差异放大、Precision 提升；但在双向/Masked 训练下发生后文泄露，导致 Recall 极低 (2%) |
| **v5** | 单向 LSTM 自回归 + NLL | 0.62 | 0.22 | **短序列冷启动惩罚**：前几个位置缺乏上下文导致 NLL 极高，阈值崩溃，无差别报警 (Recall=1.0) |
| **v6** | 单向 Transformer + NLL | 0.43 | 0.15 | Transformer 对短上下文更敏感，排序能力完全丧失 (AUC < 0.5)，阈值同样崩溃 |

### 💡 核心方法论沉淀与战略调整

通过 v1-v6 的迭代，我们得出了针对 **"大词表 (800+) + 极短序列 (ws=10)"** 场景的黄金法则：

1. **大词表下放弃离散 Top-K**：必须使用连续概率打分（如 $-\log p$），否则特征空间会坍塌。
2. **警惕双向模型的“信息泄露”**：v4 证明，双向 Masked 预测会让后文“救回”异常点，导致 Recall 极低。日志是强因果时序，理论上必须用单向预测。
3. **短序列上的“单向预测”陷阱 (v5/v6 教训)**：DeepLog 风格的纯单向自回归在长序列中有效，但在长度为 10 的窗口中，前 3 个位置因缺乏上下文（冷启动）会产生巨大的 NLL 噪声，直接淹没真正的异常信号，导致阈值校准失效。


## IF 方法定位与文献依据
结合 [docs/参考文献.txt](docs/参考文献.txt) 中的相关工作，Isolation Forest 在本项目中被用作传统方法基线与对比实验方法，主要承担以下角色：
- 为日志异常检测提供一个轻量、可解释、易复现的无监督基线
- 用于验证特征工程是否能为传统异常检测器提供有效信息
- 作为与 LSTM 序列建模路线的对照组，帮助分析序列建模的增益

相关文献中，Isolation Forest 常被用于日志异常检测或无监督异常检测基线，见文献 [1]、[2]、[6]、[10]。当前项目的 IF 实现也围绕这一定位展开，已补齐数据解析、窗口化、特征工程、训练、阈值校准、结果保存、日志保存和 artifact 保存等完整实验环节。

## LSTM 方法定位
在本项目中，LSTM 负责提供序列建模基线，用于观察“保留事件顺序”后模型能获得多少额外信息。这个定位是有文献依据的：日志异常检测综述普遍将深度学习路线视为适合复杂时序依赖和上下文建模的方向，尤其是 LSTM / LSTM-AE / DeepLog 一类方法在日志序列异常检测中被反复使用，见 [3]、[4]、[5]、[7]。

因此，本项目把 LSTM 定位为“序列建模主线的基线实现”。相较于 Isolation Forest，LSTM 更适合承接后续的序列学习改造；但针对当前较大的词表规模和极短的窗口长度，具体架构与打分策略必须严格结合实验反馈（如 v5/v6 的翻车教训）进行调整，而不能盲目套用经典论文设置。

## 实验与复现
训练脚本会把每次运行的记录保存到 `outputs/logs/`，包括：
- 运行时间与配置来源
- 数据集统计信息
- 特征工程配置
- 阈值校准策略
- 模型评估结果
- 产物路径


## 项目结构

```text
Security-Audit-Log-Anomaly-Detection/
├── README.md
├── configs/              # 核心训练与评估入口
    ├── experiment_config.yaml
│   └── experiment_config_smoke.yaml
├── data/                   # 数据加载与缓存
│   ├── raw/
│   ├── processed/
│   └── data_loader.py
├── docs/
│   ├── 参考文献.txt
│   ├── figures/
│   └── references/
├── experiments/             # 核心训练与评估入口
│   ├── train_if.py
│   ├── train_lstm_ae.py
│   └── run_comparison.py
├── models/
│   ├── if_detector.py
│   └── lstm_ae_detector.py
├── outputs/                  # 训练产物 (日志、模型权重、评估指标)
│   ├── checkpoints/
│   ├── logs/
│   └── results/
├── scripts/                 # 数据解析脚本 (Drain3)
│   └── parse_bgl.py
├── utils/                   # 特征工程、日志、评估指标
    ├── feature_engineering.py
    ├── logger.py
    └── metrics.py
└── tests/                    
    └── test_data_and_metrics.py
```

## 参考文献
本项目的参考文献以 `docs/参考文献.txt` 为准。
- [1,2,6,10] Isolation Forest 在无监督异常检测中的基线应用。
- [3,4,5,7] LSTM/DeepLog 等深度序列模型在日志异常检测中的主线地位。

## 当前状态与后续方向

**当前项目已经完成：**
- 数据解析与窗口化
- IF 路线训练、校准与日志落盘
- LSTM 路线 v1-v6 的完整范式探索与避坑复盘
- 训练日志与摘要落盘、对比实验入口的兼容修复


## 快速启动

**训练 IF**
```bash
.venv/bin/python3 -m experiments.train_if --config configs/experiment_config.yaml
```

**训练 LSTM**
```bash
.venv/bin/python3 -m experiments.train_lstm_ae --config configs/experiment_config.yaml
```

**运行对比实验**
```bash
.venv/bin/python3 -m experiments.run_comparison --config configs/experiment_config.yaml
```

**运行冒烟测试 (验证数据管道与评估函数)**
```bash
.venv/bin/python3 -m tests.test_data_and_metrics
```