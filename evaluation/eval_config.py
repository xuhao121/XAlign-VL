# ============================================================
#  跨语言一致性评估配置
#  修改这里的参数即可,不用动评估脚本
# ============================================================
import os
from itertools import combinations

# 项目根 = 当前文件 (eval/eval_config.py) 的上一级, 确保不受 CWD 影响
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- 语言 ---
LANGUAGES = ["EN", "DE", "FR", "CS"]
# 6 个语言对 = C(4,2)
LANGUAGE_PAIRS = list(combinations(LANGUAGES, 2))

# --- 输入: 训练方交付的生成结果 ---
GENERATIONS_DIR = os.path.join(_PROJECT_ROOT, "Data")

# 参考答案 (人工 ground truth, 4 语言交错: index i → image_id=i//4, lang=LANGUAGES[i%4])
REF_FILE = "ref.json"

# 三个待评模型的预测文件 (格式同 ref: 纯字符串 list, 4 语言交错)
#   trained                — 本组微调训练后的模型, 每种语言原生生成
#   nontrained             — 未训练 baseline VLM, 每种语言原生生成
#   nontrained_translated  — 未训练 baseline 先英文生成再翻译成 de/fr/cs
MODEL_FILES = {
    "trained":               "preds.json",
    "nontrained":            "predictions nontrained.json",
    "nontrained_translated": "predictionsnontrained_translated.json",
}

# 兼容旧脚本 (单模型评分)
PRED_FILE = MODEL_FILES["trained"]

# 以下为旧的跨语言一致性评估配置, 当前 evaluate.py 不再使用
METHOD_FILES = {
    # 基线
    "zero-shot":            "zero_shot.json",
    "standard-lora":        "standard_lora.json",
    "translation-pipeline": "translation_pipeline.json",
    # 我们的方法
    "ours":                 "ours.json",
    # 消融
    "ours-no-cl":           "ours_no_cl.json",    # 去掉一致性 loss
    "ours-only-cl":         "ours_only_cl.json",  # 只用一致性 loss
}

PRIMARY_METHOD = "ours"
BASELINES = ["zero-shot", "standard-lora", "translation-pipeline"]
ABLATIONS = ["ours", "ours-no-cl", "ours-only-cl"]

# --- 输出 ---
RESULTS_DIR = os.path.join(_PROJECT_ROOT, "Data", "EvalResults")

# --- 评分模型 ---
# BERTScore (同语言 pred vs ref, 衡量生成质量): 用多语言 mBERT 一个模型覆盖 4 语言
BERTSCORE_MODEL = "bert-base-multilingual-cased"
BERTSCORE_LANG = "en"            # BERTScorer 初始化需要, 实际 4 语言共用同一模型
BERTSCORE_RESCALE = False        # True 会用 baseline 重标定, 分数更易读
# Sentence-BERT (跨语言 pred(L1) vs pred(L2), 衡量一致性): 必须用多语言模型
SENTENCE_BERT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
NLI_MODEL = "roberta-large-mnli"
SPACY_MODEL = "en_core_web_sm"

# --- 错误分析 ---
ERROR_ANALYSIS_TOP_K = 20        # 取一致性最低的 K 张图做人工分析

# --- 统计显著性 ---
SIGNIFICANCE_ALPHA = 0.05
