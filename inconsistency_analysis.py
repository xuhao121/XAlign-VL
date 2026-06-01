'''
跨语言一致性评估 - 不一致来源分析

对每个语言对, 分析不一致来自哪里:
  - NLI: 判断两个描述的逻辑关系 (entailment / neutral / contradiction)
         双向都跑, 区分"信息缺失"和"内容矛盾"
  - 信息维度: 用 spaCy 提取 物体/属性/数量/空间/动作, 算每个维度的 Jaccard 重合度

用法:
  python eval/inconsistency_analysis.py --method ours
输出:
  <method>_inconsistency.csv   每张图每个语言对的 NLI 标签 + 各维度 Jaccard
'''
import argparse
import os

import pandas as pd
import spacy
import torch
from transformers import pipeline

import eval_config as cfg
from eval_common import load_generations, comparable_texts, jaccard, info_dimensions, INFO_DIMENSIONS


def build_nli(device):
    return pipeline(
        "text-classification",
        model=cfg.NLI_MODEL,
        device=device,
        batch_size=16,
        truncation=True,
    )


def nli_labels(nli, premises, hypotheses):
    """批量跑 NLI, 返回每对的标签 (大写: ENTAILMENT/NEUTRAL/CONTRADICTION)。"""
    inputs = [{"text": p, "text_pair": h} for p, h in zip(premises, hypotheses)]
    return [r["label"].upper() for r in nli(inputs)]


def analyze(method):
    samples = load_generations(method)
    print(f"[{method}] 共 {len(samples)} 张图")

    image_ids = [s["image_id"] for s in samples]
    texts = [comparable_texts(s) for s in samples]

    device = 0 if torch.cuda.is_available() else -1
    print("加载 NLI 模型...")
    nli = build_nli(device)
    print("加载 spaCy 模型...")
    nlp = spacy.load(cfg.SPACY_MODEL)

    # 每种语言的文本只做一次 spaCy 解析, 提取信息维度
    dims = {}  # dims[lang][i] = {维度: 词集合}
    for lang in cfg.LANGUAGES:
        docs = nlp.pipe([t[lang] for t in texts])
        dims[lang] = [info_dimensions(d) for d in docs]

    rows = []
    for la, lb in cfg.LANGUAGE_PAIRS:
        pair = f"{la}-{lb}"
        text_a = [t[la] for t in texts]
        text_b = [t[lb] for t in texts]

        # NLI 双向: a->b 和 b->a
        label_ab = nli_labels(nli, text_a, text_b)
        label_ba = nli_labels(nli, text_b, text_a)

        for i, img in enumerate(image_ids):
            row = {
                "image_id": img,
                "pair": pair,
                "nli_ab": label_ab[i],
                "nli_ba": label_ba[i],
            }
            for dim in INFO_DIMENSIONS:
                row[f"jac_{dim}"] = jaccard(dims[la][i][dim], dims[lb][i][dim])
            rows.append(row)
        print(f"  {pair} 完成")

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="不一致来源分析 (NLI + 信息维度)")
    ap.add_argument("--method", default=cfg.PRIMARY_METHOD, help="要分析的方法名")
    args = ap.parse_args()

    df = analyze(args.method)
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(cfg.RESULTS_DIR, f"{args.method}_inconsistency.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    # 简要汇总
    print(f"\n[{args.method}] NLI 标签分布 (a->b):")
    print(df["nli_ab"].value_counts().to_string())
    print(f"\n[{args.method}] 各信息维度平均 Jaccard:")
    for dim in INFO_DIMENSIONS:
        print(f"  {dim:11s}: {df[f'jac_{dim}'].mean():.4f}")
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
