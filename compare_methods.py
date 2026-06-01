'''
跨语言一致性评估 - 方法对比 + 统计显著性

把多个方法的评分结果放在一起对比, 并检验差异是否显著:
  - 对比表: 每个方法的整体 / 每个语言对的 BERTScore F1
  - 配对 t 检验: 每个方法 vs 主方法, 用每张图的一致性分数做 paired t-test
                p < alpha 才算差异显著 (不是运气)

前置: 先用 evaluate.py 跑出各方法的 <method>_scores.csv

用法:
  python eval/compare_methods.py                       # 对比所有有结果的方法
  python eval/compare_methods.py --methods zero-shot ours
输出:
  comparison_table.csv     方法 x 语言对的 BERTScore F1 对比
  significance.csv         各方法 vs 主方法的配对 t 检验结果
'''
import argparse
import os

import pandas as pd
from scipy import stats

import eval_config as cfg


def load_scores(method):
    """读取某方法的逐图分数, 不存在则返回 None。"""
    path = os.path.join(cfg.RESULTS_DIR, f"{method}_scores.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def per_image_consistency(df):
    """每张图的一致性分数 = 该图 6 个语言对 BERTScore F1 的平均。"""
    return df.groupby("image_id")["bertscore_f1"].mean()


def build_comparison_table(scores):
    """方法 x [整体 + 各语言对] 的 BERTScore F1 均值表。"""
    rows = []
    for method, df in scores.items():
        row = {"method": method, "overall": df["bertscore_f1"].mean()}
        for pair, g in df.groupby("pair"):
            row[pair] = g["bertscore_f1"].mean()
        rows.append(row)
    cols = ["method", "overall"] + [f"{a}-{b}" for a, b in cfg.LANGUAGE_PAIRS]
    return pd.DataFrame(rows)[cols]


def significance_tests(scores, primary):
    """每个方法 vs 主方法的配对 t 检验 (按 image_id 对齐)。"""
    base = per_image_consistency(scores[primary])
    rows = []
    for method, df in scores.items():
        if method == primary:
            continue
        other = per_image_consistency(df)
        common = base.index.intersection(other.index)
        if len(common) == 0:
            print(f"[警告] {method} 与 {primary} 没有共同图片, 跳过检验")
            continue
        a = base.loc[common]
        b = other.loc[common]
        t, p = stats.ttest_rel(a, b)
        rows.append({
            "method": method,
            "vs": primary,
            "n_images": len(common),
            "mean_other": b.mean(),
            "mean_primary": a.mean(),
            "mean_diff": a.mean() - b.mean(),   # >0 表示主方法更好
            "t_stat": t,
            "p_value": p,
            "significant": bool(p < cfg.SIGNIFICANCE_ALPHA),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="方法对比 + 统计显著性检验")
    ap.add_argument("--methods", nargs="+", help="要对比的方法名 (默认: 所有有结果的)")
    ap.add_argument("--primary", default=cfg.PRIMARY_METHOD, help="作为对照基准的主方法")
    args = ap.parse_args()

    candidates = args.methods or list(cfg.METHOD_FILES.keys())
    scores = {}
    for m in candidates:
        df = load_scores(m)
        if df is None:
            print(f"[{m}] 跳过: 没有 {m}_scores.csv, 请先跑 evaluate.py")
            continue
        scores[m] = df

    if args.primary not in scores:
        raise SystemExit(f"主方法 '{args.primary}' 没有评分结果, 无法做对比")

    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    table = build_comparison_table(scores)
    table_path = os.path.join(cfg.RESULTS_DIR, "comparison_table.csv")
    table.to_csv(table_path, index=False, encoding="utf-8-sig")
    print("\n=== 方法对比 (BERTScore F1) ===")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    sig = significance_tests(scores, args.primary)
    if not sig.empty:
        sig_path = os.path.join(cfg.RESULTS_DIR, "significance.csv")
        sig.to_csv(sig_path, index=False, encoding="utf-8-sig")
        print(f"\n=== 配对 t 检验 (vs {args.primary}, alpha={cfg.SIGNIFICANCE_ALPHA}) ===")
        print(sig.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(f"\n结果已保存: {table_path}, {sig_path}")
    else:
        print(f"\n结果已保存: {table_path}")


if __name__ == "__main__":
    main()
