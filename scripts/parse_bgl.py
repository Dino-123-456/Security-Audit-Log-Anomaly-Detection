# scripts/parse_bgl.py
import re
import pandas as pd
from pathlib import Path
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from tqdm import tqdm


def parse_bgl(raw_log_path: str, output_csv_path: str):
    """
    将原始BGL.log解析为结构化CSV，包含EventId和Label列。
    BGL日志格式: RAS_MSG L1 TOS NODE_ID JOB_ID TIMESTAMP LEVEL MESSAGE
    """
    # === 配置Drain3针对BGL的解析规则 ===
    config = TemplateMinerConfig()
    config.profiling_enabled = False
    config.drain_max_clusters = 10000       # BGL日志模式丰富，需扩大聚类上限
    config.drain_similarity_threshold = 0.4 # BGL变量多，适当降低相似度阈值
    config.masking = [
        {"regex_pattern": r"((?<=\s)|^)\d{4}\.\d{2}\.\d{2}", "mask_with": "<DATE>"},
        {"regex_pattern": r"((?<=\s)|^)\d{2}:\d{2}:\d{2}", "mask_with": "<TIME>"},
        {"regex_pattern": r"\b[0-9a-fA-F]{8,}\b", "mask_with": "<HEX>"},
        {"regex_pattern": r"\b\d+\b", "mask_with": "<NUM>"},
    ]

    template_miner = TemplateMiner(config=config)

    records = []
    raw_path = Path(raw_log_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"未找到原始日志文件: {raw_log_path}")

    print(f"[Parser] 开始解析: {raw_log_path}")
    with open(raw_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Parsing BGL"):
        line = line.strip()
        if not line:
            continue

        # BGL标签规则: 以 "-" 开头为正常，否则为异常
        label = '-' if line.startswith('-') else line[0]

        # 去除首字符标签后送入Drain3
        content = line[1:].strip() if line[0] != '-' else line.strip()
        result = template_miner.add_log_message(content)

        records.append({
            "EventId": f"E{result['cluster_id']}",
            "Content": content,
            "Label": label
        })

    df = pd.DataFrame(records)
    df.to_csv(output_csv_path, index=False)

    print(f"[Parser] 解析完成!")
    print(f"  总行数: {len(df)}")
    print(f"  唯一EventId数: {df['EventId'].nunique()}")
    print(f"  异常比例: {(df['Label'] != '-').mean():.4f}")
    print(f"  输出文件: {output_csv_path}")


if __name__ == "__main__":
    parse_bgl(
        raw_log_path="data/raw/BGL.log",
        output_csv_path="data/raw/BGL_structured.csv"
    )