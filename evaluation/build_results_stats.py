"""
Aggregate extended statistics from the per-image score CSVs in
data/EvalResults/ and dump them to results_extended.json so they can be
quoted directly in the results report.

Computes, per model in {trained, nontrained, nontrained_translated}:
  - BERTScore F1 mean/std/median per language and overall
  - Per-image paired win rate of trained vs nontrained (per language)
  - Wilcoxon signed-rank p-value, trained vs nontrained (per language)
  - SBERT consistency mean/std per pair and overall
  - Per-language anchored consistency (avg of the 3 pairs that include L)
  - Pearson r between per-image BERTScore F1 and that image's mean
    pairwise consistency
"""
import json
import os
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "data", "EvalResults")
LANGS = ["EN", "DE", "FR", "CS"]
PAIRS = [f"{a}-{b}" for a, b in combinations(LANGS, 2)]
MODELS = ["trained", "nontrained", "nontrained_translated"]


def load_bertscore(model):
    path = os.path.join(RES, f"bertscore_{model}_scores.csv")
    df = pd.read_csv(path)
    return df


def load_consistency(model):
    path = os.path.join(RES, f"consistency_{model}_scores.csv")
    df = pd.read_csv(path)
    return df


def bertscore_block(df):
    out = {"per_language": {}}
    for L in LANGS:
        s = df.loc[df["lang"] == L, "bertscore_f1"].to_numpy()
        out["per_language"][L] = {
            "n": int(s.size),
            "mean": float(s.mean()),
            "std": float(s.std(ddof=1)),
            "median": float(np.median(s)),
        }
    s = df["bertscore_f1"].to_numpy()
    out["overall"] = {
        "n": int(s.size),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)),
        "median": float(np.median(s)),
    }
    return out


def consistency_block(df):
    out = {"per_pair": {}, "per_language_anchored": {}}
    for p in PAIRS:
        s = df.loc[df["pair"] == p, "cos"].to_numpy()
        out["per_pair"][p] = {
            "n": int(s.size),
            "mean": float(s.mean()),
            "std": float(s.std(ddof=1)),
            "median": float(np.median(s)),
        }
    # anchored: for each language L, mean of cos over the 3 pairs that include L
    for L in LANGS:
        keep = df["pair"].str.contains(L)
        s = df.loc[keep, "cos"].to_numpy()
        out["per_language_anchored"][L] = {
            "n": int(s.size),
            "mean": float(s.mean()),
            "std": float(s.std(ddof=1)),
        }
    s = df["cos"].to_numpy()
    out["overall"] = {
        "n_pairs": len(PAIRS),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)),
    }
    return out


def paired_stats(df_a, df_b):
    """trained vs nontrained, per-language paired comparison on BERTScore."""
    out = {}
    merged = df_a.merge(
        df_b, on=["image_id", "lang"], suffixes=("_a", "_b")
    )
    for L in LANGS:
        sub = merged[merged["lang"] == L]
        a = sub["bertscore_f1_a"].to_numpy()
        b = sub["bertscore_f1_b"].to_numpy()
        wins = int((a > b).sum())
        losses = int((a < b).sum())
        ties = int((a == b).sum())
        try:
            w = stats.wilcoxon(a, b, alternative="greater")
            pval = float(w.pvalue)
        except ValueError:
            pval = float("nan")
        out[L] = {
            "n": int(len(sub)),
            "win_rate_trained": wins / len(sub),
            "loss_rate_trained": losses / len(sub),
            "tie_rate": ties / len(sub),
            "mean_delta": float((a - b).mean()),
            "wilcoxon_p_trained_greater": pval,
        }
    return out


def correlation_quality_vs_consistency(bs_df, cons_df):
    """Per image, mean BERTScore across 4 languages vs mean cos across 6 pairs."""
    q = bs_df.groupby("image_id")["bertscore_f1"].mean().reset_index(
        name="q"
    )
    c = cons_df.groupby("image_id")["cos"].mean().reset_index(name="c")
    m = q.merge(c, on="image_id")
    r, p = stats.pearsonr(m["q"], m["c"])
    return {"n_images": int(len(m)), "pearson_r": float(r), "p_value": float(p)}


def main():
    out = {"models": {}}
    bs = {m: load_bertscore(m) for m in MODELS}
    cs = {m: load_consistency(m) for m in MODELS}

    for m in MODELS:
        out["models"][m] = {
            "bertscore": bertscore_block(bs[m]),
            "consistency": consistency_block(cs[m]),
            "quality_consistency_correlation":
                correlation_quality_vs_consistency(bs[m], cs[m]),
        }

    out["paired"] = {
        "trained_vs_nontrained_bertscore":
            paired_stats(bs["trained"], bs["nontrained"]),
        "trained_vs_nontrained_translated_bertscore":
            paired_stats(bs["trained"], bs["nontrained_translated"]),
    }

    # Pairwise consistency deltas
    out["consistency_deltas"] = {}
    for p in PAIRS + ["overall"]:
        def get(model, p):
            block = out["models"][model]["consistency"]
            return block["per_pair"][p]["mean"] if p != "overall" else block["overall"]["mean"]
        out["consistency_deltas"][p] = {
            "trained_minus_nontrained":
                get("trained", p) - get("nontrained", p),
        }

    dest = os.path.join(RES, "results_extended.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote {dest}")


if __name__ == "__main__":
    main()
