'''
质量评估 - BERTScore (pred vs ref, 同语言)

对 eval_config.MODEL_FILES 里注册的每个模型, 算每条 (prediction, reference) 的:
  - BERTScore F1   (用多语言 mBERT, 4 语言共用一个模型)

按 lang 分组报告 (每种语言一组均值±标准差) + overall 总均值。
最后输出一个横向对比表: 3 模型 × 4 语言 = 12 个分数。

用法:
  python eval/evaluate.py
输出 (写到 eval_config.RESULTS_DIR):
  bertscore_<model>_scores.csv     每个模型的逐条分数
  bertscore_<model>_summary.json   每个模型的按语言分组均值
  bertscore_compare.json           三个模型的横向对比 (3×4 = 12 个分数)
'''
import argparse
import json
import os

import pandas as pd
from bert_score import BERTScorer

import eval_config as cfg
from eval_common import load_pairs_for_model


def score_model(scorer, model_name):
    """对一个模型的所有 (pred, ref) 算 BERTScore F1, 返回 DataFrame。"""
    samples = load_pairs_for_model(model_name)
    print(f"[{model_name}] 共 {len(samples)} 条样本")

    cands = [s["prediction"] for s in samples]
    refs = [s["reference"] for s in samples]
    _, _, f1 = scorer.score(cands, refs)

    return pd.DataFrame({
        "id": [s["id"] for s in samples],
        "image_id": [s["image_id"] for s in samples],
        "lang": [s["lang"] for s in samples],
        "prediction": cands,
        "reference": refs,
        "bertscore_f1": f1.tolist(),
    })


def _stats(df):
    return {
        "n": int(len(df)),
        "bertscore_f1_mean": float(df["bertscore_f1"].mean()),
        "bertscore_f1_std": float(df["bertscore_f1"].std()),
    }


def summarize(df):
    summary = {"overall": _stats(df), "per_language": {}}
    for lang in cfg.LANGUAGES:
        sub = df[df["lang"] == lang]
        if len(sub) > 0:
            summary["per_language"][lang] = _stats(sub)
    return summary


def main():
    argparse.ArgumentParser(description="BERTScore 质量评估 (三个模型)").parse_args()

    print("加载 BERTScore 模型...")
    scorer = BERTScorer(
        model_type=cfg.BERTSCORE_MODEL,
        lang=cfg.BERTSCORE_LANG,
        rescale_with_baseline=cfg.BERTSCORE_RESCALE,
    )

    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    compare = {}

    for model_name in cfg.MODEL_FILES:
        df = score_model(scorer, model_name)
        csv_path = os.path.join(cfg.RESULTS_DIR, f"bertscore_{model_name}_scores.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        summary = summarize(df)
        sum_path = os.path.join(cfg.RESULTS_DIR, f"bertscore_{model_name}_summary.json")
        with open(sum_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        compare[model_name] = summary
        print(f"  → {csv_path}")
        print(f"  → {sum_path}")

    cmp_path = os.path.join(cfg.RESULTS_DIR, "bertscore_compare.json")
    with open(cmp_path, "w", encoding="utf-8") as f:
        json.dump(compare, f, ensure_ascii=False, indent=2)

    # 打印横向对比表
    print()
    print("=" * 78)
    print("BERTScore F1 横向对比 (mean ± std)")
    print("=" * 78)
    header = f"{'model':<24}" + "".join(f"{lang:>12}" for lang in cfg.LANGUAGES) + f"{'overall':>12}"
    print(header)
    print("-" * len(header))
    for model_name, s in compare.items():
        row = f"{model_name:<24}"
        for lang in cfg.LANGUAGES:
            v = s["per_language"].get(lang)
            row += f"{v['bertscore_f1_mean']:>12.4f}" if v else f"{'-':>12}"
        row += f"{s['overall']['bertscore_f1_mean']:>12.4f}"
        print(row)
    print()
    print(f"对比表已保存: {cmp_path}")


if __name__ == "__main__":
    main()
