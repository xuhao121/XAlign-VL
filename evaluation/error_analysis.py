'''
跨语言一致性评估 - 错误分析

挑出一致性最低的 K 张图, 导出成方便人工分析的 Markdown:
  - 4 种语言的对比文本 (EN 原文 + 其它语言的英译)
  - 6 个语言对的 BERTScore F1
  - 每种语言提取到的信息维度 (物体/属性/数量/空间/动作)
人工在此基础上归纳不一致的原因 (论文最加分的部分)。

前置: 先用 evaluate.py 跑出 <method>_scores.csv

用法:
  python eval/error_analysis.py --method ours --top-k 20
输出:
  error_analysis_<method>.md
'''
import argparse
import os

import pandas as pd
import spacy

import eval_config as cfg
from eval_common import load_generations, comparable_texts, info_dimensions, INFO_DIMENSIONS


def worst_images(method, top_k):
    """返回一致性最低的 top_k 个 (image_id, 平均F1)。"""
    path = os.path.join(cfg.RESULTS_DIR, f"{method}_scores.csv")
    if not os.path.exists(path):
        raise SystemExit(f"找不到 {path}, 请先跑 evaluate.py --method {method}")
    df = pd.read_csv(path)
    per_img = df.groupby("image_id")["bertscore_f1"].mean().sort_values()
    pair_f1 = df.set_index(["image_id", "pair"])["bertscore_f1"]
    return per_img.head(top_k), pair_f1


def write_report(method, top_k):
    per_img, pair_f1 = worst_images(method, top_k)
    samples = {s["image_id"]: s for s in load_generations(method)}
    nlp = spacy.load(cfg.SPACY_MODEL)

    lines = [
        f"# 错误分析: {method}",
        "",
        f"一致性最低的 {len(per_img)} 张图 (一致性 = 6 个语言对 BERTScore F1 的平均)。",
        "在每张图下方的 `人工分析` 处填写不一致原因。",
        "",
    ]

    for rank, (img, score) in enumerate(per_img.items(), 1):
        sample = samples.get(img)
        if sample is None:
            continue
        texts = comparable_texts(sample)
        lines += [f"## {rank}. {img}  (平均 F1 = {score:.4f})", ""]

        # 各语言对比文本
        lines.append("**对比文本 (英文 / 英译):**")
        lines.append("")
        for lang in cfg.LANGUAGES:
            lines.append(f"- `{lang}`: {texts[lang]}")
        lines.append("")

        # 6 个语言对的 F1
        lines.append("**各语言对 BERTScore F1:**")
        lines.append("")
        for la, lb in cfg.LANGUAGE_PAIRS:
            f1 = pair_f1.get((img, f"{la}-{lb}"), float("nan"))
            lines.append(f"- {la}-{lb}: {f1:.4f}")
        lines.append("")

        # 信息维度
        lines.append("**信息维度覆盖:**")
        lines.append("")
        dims = {lang: info_dimensions(nlp(texts[lang])) for lang in cfg.LANGUAGES}
        header = "| 维度 | " + " | ".join(cfg.LANGUAGES) + " |"
        sep = "|" + "---|" * (len(cfg.LANGUAGES) + 1)
        lines += [header, sep]
        for dim in INFO_DIMENSIONS:
            cells = [", ".join(sorted(dims[lang][dim])) or "-" for lang in cfg.LANGUAGES]
            lines.append(f"| {dim} | " + " | ".join(cells) + " |")
        lines += ["", "**人工分析:** _(待填写)_", "", "---", ""]

    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(cfg.RESULTS_DIR, f"error_analysis_{method}.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"已导出 {len(per_img)} 张图的错误分析: {out_path}")


def main():
    ap = argparse.ArgumentParser(description="错误分析: 导出一致性最低的图片")
    ap.add_argument("--method", default=cfg.PRIMARY_METHOD, help="要分析的方法名")
    ap.add_argument("--top-k", type=int, default=cfg.ERROR_ANALYSIS_TOP_K, help="导出最差的几张图")
    args = ap.parse_args()
    write_report(args.method, args.top_k)


if __name__ == "__main__":
    main()
