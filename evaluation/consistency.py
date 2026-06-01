'''
跨语言一致性评估 - SBERT (同模型 pred(L1) vs pred(L2), 跨语言)

对每个模型, 用多语言 SBERT 编码全部预测, 按 image_id 分组, 对
LANGUAGE_PAIRS 里的 6 个语言对算 pairwise 余弦相似度。

报告:
  - 每个模型每个语言对的均值±标准差
  - 每个模型的 overall (6 对取平均)
  - 三模型横向对比表

用法:
  python eval/consistency.py
输出 (写到 eval_config.RESULTS_DIR):
  consistency_<model>_scores.csv    每张图每个语言对的分数
  consistency_<model>_summary.json  每个模型 per-pair + overall 均值
  consistency_compare.json          三模型横向对比
'''
import argparse
import json
import os

import pandas as pd
from sentence_transformers import SentenceTransformer, util

import eval_config as cfg
from eval_common import load_pairs_for_model


def consistency_for_model(sbert, model_name):
    """对一个模型, 按 image_id 分组算 6 个语言对的余弦相似度。"""
    samples = load_pairs_for_model(model_name)
    print(f"[{model_name}] 编码 {len(samples)} 条预测...")

    # 按 (image_id, lang) 索引出预测文本
    by_image = {}
    for s in samples:
        by_image.setdefault(s["image_id"], {})[s["lang"]] = s["prediction"]

    image_ids = sorted(by_image.keys())
    # 校验每张图都有 4 语言
    for img in image_ids:
        missing = [l for l in cfg.LANGUAGES if l not in by_image[img]]
        if missing:
            raise ValueError(f"[{model_name}] 图 {img} 缺少语言: {missing}")

    # 一次性编码: 对每种语言把所有图的文本拼成一个 list, batch 编码
    embeddings = {}  # lang -> tensor (n_images, d)
    for lang in cfg.LANGUAGES:
        texts = [by_image[img][lang] for img in image_ids]
        embeddings[lang] = sbert.encode(
            texts, convert_to_tensor=True, show_progress_bar=True, batch_size=64
        )

    # 对每个语言对, pairwise cos sim (逐图一一对应)
    rows = []
    for l1, l2 in cfg.LANGUAGE_PAIRS:
        cos = util.pairwise_cos_sim(embeddings[l1], embeddings[l2]).cpu().tolist()
        pair_name = f"{l1}-{l2}"
        for img, c in zip(image_ids, cos):
            rows.append({"image_id": img, "pair": pair_name, "cos": c})
    return pd.DataFrame(rows)


def summarize(df):
    summary = {"per_pair": {}}
    for pair in df["pair"].unique():
        sub = df[df["pair"] == pair]
        summary["per_pair"][pair] = {
            "n": int(len(sub)),
            "cos_mean": float(sub["cos"].mean()),
            "cos_std": float(sub["cos"].std()),
        }
    summary["overall"] = {
        "n_pairs": len(summary["per_pair"]),
        # overall: 先对每张图取 6 对的均值, 再对所有图取均值
        # (等价于先按 pair 取均值再取均值, 因为每张图都有全部 6 对)
        "cos_mean": float(df["cos"].mean()),
        "cos_std": float(df["cos"].std()),
    }
    return summary


def main():
    argparse.ArgumentParser(description="SBERT 跨语言一致性评估 (三个模型)").parse_args()

    print(f"加载多语言 SBERT: {cfg.SENTENCE_BERT_MODEL}")
    sbert = SentenceTransformer(cfg.SENTENCE_BERT_MODEL)

    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    compare = {}

    for model_name in cfg.MODEL_FILES:
        df = consistency_for_model(sbert, model_name)
        csv_path = os.path.join(cfg.RESULTS_DIR, f"consistency_{model_name}_scores.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        summary = summarize(df)
        sum_path = os.path.join(cfg.RESULTS_DIR, f"consistency_{model_name}_summary.json")
        with open(sum_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        compare[model_name] = summary
        print(f"  → {csv_path}")
        print(f"  → {sum_path}")

    cmp_path = os.path.join(cfg.RESULTS_DIR, "consistency_compare.json")
    with open(cmp_path, "w", encoding="utf-8") as f:
        json.dump(compare, f, ensure_ascii=False, indent=2)

    # 横向对比表
    pair_names = [f"{l1}-{l2}" for l1, l2 in cfg.LANGUAGE_PAIRS]
    print()
    print("=" * 110)
    print("跨语言一致性 (SBERT 余弦相似度, mean)")
    print("=" * 110)
    header = f"{'model':<24}" + "".join(f"{p:>10}" for p in pair_names) + f"{'overall':>10}"
    print(header)
    print("-" * len(header))
    for model_name, s in compare.items():
        row = f"{model_name:<24}"
        for p in pair_names:
            v = s["per_pair"].get(p)
            row += f"{v['cos_mean']:>10.4f}" if v else f"{'-':>10}"
        row += f"{s['overall']['cos_mean']:>10.4f}"
        print(row)
    print()
    print(f"对比表已保存: {cmp_path}")


if __name__ == "__main__":
    main()
