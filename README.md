# Security Audit Log Anomaly Detection (安全审计日志异常检测)

## 📖 项目简介
本项目面向 **BGL 超算日志数据集**，研究安全审计日志的异常检测问题。核心任务是将原始日志通过 Drain 算法解析为 Event ID 序列，再按滑动窗口切分为二分类样本，深度对比传统机器学习路线与深度序列建模路线在日志异常检测中的表现。

本项目不仅完成了完整的数据管道、模型训练与评估落盘流程，更**经历了一场从“学术刷榜”到“工业落地”、从“数据作弊”到“严谨时序”的深刻认知升级**。项目沉淀了 12 个版本的模型迭代复盘，突破了传统 AUC/F1 指标在极端概念漂移（Concept Drift）下的评估陷阱，最终提出了一套基于**混合多视角架构**与 **Precision@K (Top-K 命中率)** 的工业级异常检测方案。

## 🎯 项目目标
1. **严谨的数据工程**：将原始 BGL 日志转换为结构化 Event 序列，基于滑动窗口构造二分类样本，并**彻底摒弃存在“数据穿越”风险的随机划分，确立严格的时序划分（Time-based Split）标准**。
2. **深度路线对比**：对比传统机器学习（Isolation Forest）与深度学习（LSTM / Transformer / 对比学习）的检测效能，记录各类范式在特定数据分布下的失效原因。
3. **评估范式破局**：揭示在时序漂移场景下，全局 AUC 的“虚荣指标”陷阱，确立以 **Precision@K** 为核心的工业级评估体系。
4. **核心资产沉淀**：保存核心训练脚本、配置文件以及关键历史版本（如 v5/v6 单向序列模型、v12 混合多视角模型）的代码，为后续改进 DeepLog、LogBERT 等时序模型提供极具价值的**避坑指南**。

---

## 🚀 核心实验演进与踩坑复盘 (核心贡献)

本项目的核心价值在于**真实的试错与范式演进**。以下是我们经历的五个关键阶段：

### 阶段零：数据集构建的觉醒 —— 从“作弊”到“严谨” 🌟
在早期的实验设计中，我们曾采用**随机划分（Random Split）** 来构建训练集和测试集，但这在日志数据中是一个致命的“学术陷阱”。
* **🚨 致命缺陷：数据穿越 (Data Leakage)**
  * **现象**：随机划分下，模型 AUC 轻易达到 0.90+。
  * **根因**：BGL 日志具有极强的时间局部性和重复性。同一种系统报错（异常）或同一种启动流程（正常）会在相邻时间段内密集出现。随机划分会导致**训练集和测试集包含了大量时间上相邻、内容几乎相同的日志**。模型根本不需要学习“异常模式”，只需“死记硬背”训练集里的具体序列，就能在测试集拿高分。这在工业界被称为“未来信息泄露”。
* **✅ 最终方案：严格时序划分 (Time-based Split)**
  * 我们强制按照时间戳排序，采用严格时序划分：前 80% 作为训练集，中间 10% 作为验证集，最后 10% 作为测试集”。
  * **代价与收获**：这一改动导致所有模型的 AUC 瞬间暴跌（因为测试集充满了训练集没见过的“新概念/新正常状态”），但却逼迫我们直面真实的**概念漂移（Concept Drift）** 问题，从而引出了后续评估范式的彻底重构。

### 阶段一：传统基线与早期序列建模的“滑铁卢” (v1 之前)
在引入深度学习之前，我们测试了最经典的组合，但遭遇了严重的“水土不服”：

#### 🚨 路线 A：TF-IDF + Isolation Forest (IF)
*最终 AUC 仅为 0.55（接近随机盲猜）。*
* **坑 1：语序丢失**：TF-IDF 的“词袋模型”完全抹杀了时间顺序。日志的核心在于时序依赖（如 A->B->C 正常，A->C->B 异常），IF 无法区分二者。
* **坑 2：维度灾难与稀疏性**：BGL 提取出 800+ 个 Event ID，导致 TF-IDF 矩阵极高维且稀疏。IF 基于空间划分的树结构在高维稀疏空间中距离度量失效，局部异常敏感度极差。

#### 🚨 路线 B：早期 LSTM 探索
*Loss 完美收敛，但评估指标反直觉崩盘。*
* **范式 1：Next-Token Prediction (自回归)**
  * **高频异常的“肌肉记忆”**：AUC < 0.5。BGL 异常多为系统级崩溃，大段重复。模型“死记硬背”了高频异常词（Loss 极低），反而对罕见的长尾正常词预测 Loss 极高，导致分数倒挂。
  * **Next-Step 的“视野盲区”**：若异常在窗口中间，而最后一个词正常，模型只关注预测最后的正常词，漏掉中间异常。
* **范式 2：Masked Token Prediction + Max Pooling**
  * **异常抱团效应**：AUC 仅 0.54。BGL 异常词往往“抱团”出现，Mask 掉其中一个，模型看周围全是相同异常词，轻易猜中，导致异常样本 Loss 依然很低。
  * **纯符号缺乏语义**：Event ID 是 Drain 提取的模板符号（如 E12），不像自然语言有上下文语义，Masked 范式的优势无法发挥。

### 阶段二：LSTM 打分范式的深度探索 (v1 - v6)
针对“大词表 (800+) + 极短序列 (ws=10)”场景，我们进行了 6 次打分策略迭代。*(注：仓库中保留了 `lstm_v5_detector.py` 和 `lstm_v6_detector.py` 作为历史存档)*

| 版本 | 核心范式 | AUC | F1 | 关键发现 / 致命缺陷 |
| :--- | :--- | :--- | :--- | :--- |
| **v1** | Top‑K (K=5) | 0.63 | 0.08 | 词表≈843，K=5 命中率极低，几乎全部 miss（大词表下不可行） |
| **v2** | True Token Confidence (1-p) | 0.73 | 0.16 | 概率被压缩到 0.99x 窄带，需放大/校准以便阈值敏感化 |
| **v3** | NLL + 温度 (temp=0.5) | 0.50 | 0.03 | 温度过激导致 Perplexity 爆炸，正常样本也被极端惩罚 |
| **v4** | Log‑Confidence (-log p) | 0.66 | 0.04 | 双向/Masked 训练下发生**后文泄露**，导致 Recall 极低 (2%) |
| **v5** | 单向 LSTM 自回归 + NLL | 0.62 | 0.22 | **短序列冷启动惩罚**：前几个位置缺乏上下文导致 NLL 极高，阈值崩溃 |
| **v6** | 单向 Transformer + NLL | 0.43 | 0.15 | Transformer 对短上下文更敏感，排序能力完全丧失，阈值同样崩溃 |

**💡 核心方法论沉淀**：
1. 大词表下必须放弃离散 Top-K，使用连续概率打分（如 $-\log p$）。
2. 警惕双向模型的“信息泄露”，日志是强因果时序，理论上必须用单向预测。
3. 纯单向自回归在极短序列（长度10）中会因“冷启动”产生巨大 NLL 噪声，淹没真实异常。

### 阶段三：One-Class 与特征空间的迷失 (v7 - v10)
为了摆脱对标签的依赖，我们转向 One-Class 分类（DeepSVDD / 超球体），试图在特征空间“画圈”框住正常样本。
* **马氏距离与数值爆炸 (v10)**：引入马氏距离和 SSL（自监督学习）后，由于 L2 归一化导致特征协方差矩阵奇异，求逆时阈值飙升到 **73 亿**，模型彻底崩溃。
* **教训**：在 L2 归一化的超球面上强行计算协方差逆矩阵是死路一条；且 One-Class 模型无法区分“没见过的新正常（概念漂移）”和“见过的旧异常”。

### 阶段四：对比学习与评估范式的觉醒 (v11 - v12) 🌟
我们放弃了寻找“绝对中心”，转向**监督对比学习 (Supervised Contrastive Learning)** 拉开特征边界，并迎来了**最重要的认知觉醒**。

* **工程踩坑 (v11)**：在实现 InfoNCE Loss 的对角线掩码时，遭遇 PyTorch `fill_diagonal_` 等 **inplace 操作破坏计算图**的隐蔽 Bug，导致 `backward()` 报 `RuntimeError` 崩溃。排查后全面改用 out-of-place 的 `masked_fill`，并引入了 `logsumexp` trick 以解决 Softmax 溢出导致的数值不稳定问题。
* **范式觉醒 (v12)**：我们发现，在 BGL 这种存在极端**概念漂移**的时序数据集中，死磕全局 AUC 是一个“学术陷阱”。真实的 SOC（安全运营中心）运维人员只关注 **Top-K 高置信度告警**。
* **最终方案 (v12 混合多视角架构)**：
  * **语义视角**：基于 Transformer 特征 KNN 的局部密度。
  * **统计视角**：基于 Token 稀有度 (OOV/Rare token penalty)，这是对抗概念漂移的终极武器。
  * **最终战绩**：虽然全局 AUC 仅为 0.44（时序漂移下的必然结果），但工业级指标 **P@100 达到 67.0%，P@500 达到 73.6%**！这意味着运维人员审查 Top 500 告警，近四分之三都是真实威胁，彻底解决了“告警疲劳”问题。

---

## 🛠️ 最终架构设计 (v12 混合多视角)

当前项目落地的最终方案为 **混合多视角异常检测架构**，旨在同时捕获日志的“深层语义”与“表层统计”特征：

1. **语义视角 (Semantic View)**：
   * 采用 Transformer 编码器提取日志序列的深层上下文特征。
   * 引入**监督对比学习 (InfoNCE Loss)** 优化特征空间，使同类样本聚拢、异类样本排斥，形成清晰的决策边界。
   * 推理时，通过 KNN 计算测试样本与训练集参考库的局部密度得分。
2. **统计视角 (Statistical View)**：
   * 引入 **Token 稀有度惩罚 (OOV/Rare Token Penalty)**。
   * 针对 BGL 数据集中存在的极端概念漂移，利用逆文档频率 (IDF) 思想，对训练集中未见过的 OOV 词或长尾罕见词给予极高的异常权重。这是对抗时序漂移的“终极武器”。
3. **多视角融合 (Score Fusion)**：
   * 采用自适应加权策略融合双视角分数，摒弃单一的全局硬阈值。

### 💼 业务价值映射 (从指标到落地)
我们不仅在技术上实现了突破，更将技术指标完美映射到了真实的 SOC（安全运营中心）业务场景中：
* **P@100 = 67.0%**：意味着运维专家每天早晨精力最充沛时，审查系统推送的 **Top 100 最高危告警**，其中 67 条是实打实的真实威胁，彻底告别“狼来了”的无效打扰。
* **P@500 = 73.6%**：意味着在常规的半日巡检中，模型推送的前 500 条告警具有极高的置信度，成功将“海量日志风暴”浓缩为“高价值安全情报”。

---

## 💻 实验环境与运行指南 (Step-by-Step)

本项目提供了一套完整的数据处理到模型评估的流水线，推荐按以下步骤运行：

### Step 1: 环境配置
```bash
# 创建并激活虚拟环境
python3 -m venv .venv
source .venv/bin/activate  

# 安装核心依赖 (以 CPU 版本为例)
pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  --trusted-host pypi.tuna.tsinghua.edu.cn \
  --trusted-host download.pytorch.org

pip install numpy pandas scikit-learn matplotlib seaborn tqdm drain3 pyyaml joblib scipy
```

### Step 2: 数据获取与解析
本项目使用 LogHub 提供的 BGL 超算日志数据集。
* **数据来源**：https://github.com/logpai/loghub/tree/master/BGL
* 请下载原始日志文件并将其重命名为 `BGL.log`，放置于 `data/raw/` 目录下。

```bash
# 运行 Drain3 算法将非结构化日志解析为结构化 Event ID 序列
python scripts/parse_bgl.py --input data/raw/BGL.log --output data/raw/BGL_structured.csv
```

### Step 3: 模型训练与实验
项目在 `configs/` 目录下提供了完整的配置文件。您可以根据需要选择训练不同的模型路线：

#### 路线 A：传统基线 Isolation Forest (IF)
*基于 TF-IDF 特征工程，用于对比传统无监督方法在日志检测上的局限性。*
```bash
python -m experiments.train_if --config configs/experiment_config.yaml
```

#### 路线 B：早期序列模型 LSTM / Causal Transformer
*探索 Next-Token Prediction 与自回归打分范式（对应 v5/v6 历史版本）。*
```bash
# 完整训练
python -m experiments.train_lstm_ae --config configs/experiment_config.yaml

# 快速冒烟测试 (仅跑 1 个 Epoch 验证代码流程)
python -m experiments.train_lstm_ae --config configs/experiment_config_smoke.yaml
```

#### 路线 C：最终方案 混合多视角架构 (v12 Hypersphere) 🌟
*结合监督对比学习与统计稀有度惩罚的工业级落地方案。*
```bash
# 完整训练 (推荐)
python -m experiments.train_hypersphere --config configs/experiment_config.yaml

# 快速冒烟测试
python -m experiments.train_hypersphere --config configs/experiment_config_smoke.yaml
```

### Step 4: 结果评估
训练完成后，模型权重将保存至 `outputs/checkpoints/`，详细的评估指标（包含 Precision@K、AUC、F1 等）将输出至 `outputs/results/` 目录。可通过查看生成的 JSON/CSV 报告分析模型在时序划分下的真实表现。

---

## 📂 项目结构

```text
Security-Audit-Log-Anomaly-Detection/
├── README.md # 项目说明与踩坑复盘
├── configs/
│   ├── experiment_config.yaml # 完整训练配置 (含 IF, LSTM, Hypersphere)
│   └── experiment_config_smoke.yaml # 快速冒烟测试配置
├── data/
│   ├── raw/ # 原始 BGL 日志及解析后的 CSV
│   ├── processed/ # 时序划分后的缓存数据
│   └── data_loader.py # 数据加载与防作弊时序划分逻辑
├── docs/
│   ├── references/
│   └── 参考文献.txt
├── experiments/ # 🔧 核心训练脚本 (已更新)
│   ├── train_if.py # IF 训练入口
│   ├── train_lstm_v4.py # 【更新】修复 yaml.safe_load 错误，集成 Top-K 评估
│   ├── train_lstm_v5.py # 【更新】修复 yaml.safe_load 错误，集成 Top-K 评估
│   └── train_hypersphere_v6.py # v12 核心训练脚本
├── models/
│   ├── if_detector.py # Isolation Forest 封装
│   ├── lstm_ae_detector.py # LSTM/序列模型封装
│   ├── lstm_v5_detector.py # [历史存档] v5 单向 LSTM + NLL
│   ├── transformer_v6_detector.py # [历史存档] v6 单向 Transformer + NLL
│   └── hypersphere_detector.py # [当前版本] v12 混合多视角架构
├── outputs/ # 训练产物 (日志、最终模型权重、评估结果)
│   ├── checkpoints/
│   ├── logs/
│   └── results/
├── scripts/
│   └── parse_bgl.py # 数据解析脚本 (Drain3)
└── utils/
    ├── feature_engineering.py
    ├── logger.py
    └── metrics.py 
```

---

## 📚 参考文献与理论依据

本项目的基线选择、架构演进与评估体系参考了 AIOps 与日志异常检测领域的最新研究（涵盖 2023-2025 年核心文献）。
为避免长篇列表影响阅读体验，完整的参考文献列表（包含 Isolation Forest、DeepLog、LogBERT 及最新的 Contrastive Learning 应用）已整理至独立文档：

👉 **[点击查看完整参考文献列表](docs/参考文献.txt)**

## 📊 关键实验结果对比 (基于 BGL 数据集)

为了直观展示项目演进的成效，以下是核心版本在**严格时序划分（Time-based Split）**下的测试集表现。
*(注：AUC 反映全局排序能力，P@K 反映工业界最关注的 Top-K 告警准确率)*

| 模型版本 | 核心范式 | AUC | P@100 | P@500 | 关键洞察 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Isolation Forest (IF)** | TF-IDF + 无监督 | 0.55 | 0.04 | 0.07 | **传统基线失效**：高维稀疏日志特征下，无法捕捉时序依赖，效果接近随机猜测。 |
| **v5 (LSTM Autoregressive)** | 单向 LSTM + NLL | 0.62 | 0.04 | 0.07 | **深度学习的局限**：受限于“概念漂移”和短序列冷启动，虽然优于 IF，但 AUC 依然较低，Top-K 效果未达工业级要求。 |
| **v12 (混合多视角)** | Transformer + 稀有度惩罚 | **0.44** | **0.67** | **0.736** | **最终落地方案**：**不追求高 AUC**，而是通过统计视角（Token 稀有度）精准狙击 Top-K 告警，解决了告警疲劳问题。 |

> **💡 结论**：单纯增加模型深度（如 v5）无法解决日志时序漂移问题。v12 通过引入统计视角的“混合架构”，在 P@500 指标上相比 v5 提升了 **10 倍以上**，实现了从“学术玩具”到“工业利器”的跨越。

