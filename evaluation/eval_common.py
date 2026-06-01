'''
评估脚本共用模块: 数据加载 + 信息维度提取。

训练方交付的预测/参考文件格式 (放在 GENERATIONS_DIR 下):
  preds.json: ["pred_0", "pred_1", ...]    # 纯字符串 list
  ref.json:   ["ref_0",  "ref_1",  ...]    # 纯字符串 list
两个文件等长, 长度必须是 4 的倍数, 按 "4 语言交错" 排列:
  index i → image_id = i // 4, lang = LANGUAGES[i % 4]
跨方法对比 (compare_methods.py) 用的多方法结果文件格式见 load_generations。
'''
import json
import os

import eval_config as cfg


def _load_flat_list(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到文件: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} 必须是 list of str")
    return data


def load_pairs_for_model(model_name):
    """读取指定模型的预测 + 参考, 按 4 语言交错切分。

    返回: [{"id", "image_id", "lang", "prediction", "reference"}, ...]
      id        — 全局行号 (0..N-1, 唯一)
      image_id  — 图片分组号 (i // 4), 同一张图的 4 个语言共享
      lang      — LANGUAGES[i % 4]
    """
    if model_name not in cfg.MODEL_FILES:
        raise KeyError(f"未注册的模型: {model_name}, 可选: {list(cfg.MODEL_FILES)}")
    preds = _load_flat_list(os.path.join(cfg.GENERATIONS_DIR, cfg.MODEL_FILES[model_name]))
    refs = _load_flat_list(os.path.join(cfg.GENERATIONS_DIR, cfg.REF_FILE))

    if len(preds) != len(refs):
        raise ValueError(f"{model_name}: 预测和参考长度不一致 {len(preds)} vs {len(refs)}")
    n_langs = len(cfg.LANGUAGES)
    if len(preds) % n_langs != 0:
        raise ValueError(f"{model_name}: 样本数 {len(preds)} 不是语言数 {n_langs} 的倍数")

    samples = []
    for i, (pred, ref) in enumerate(zip(preds, refs)):
        if not isinstance(pred, str) or not isinstance(ref, str):
            raise ValueError(f"{model_name} index={i}: 元素必须是字符串")
        samples.append({
            "id": i,
            "image_id": i // n_langs,
            "lang": cfg.LANGUAGES[i % n_langs],
            "prediction": pred,
            "reference": ref,
        })
    return samples


def load_pairs():
    """向后兼容: 默认加载 trained 模型。"""
    default_model = next(iter(cfg.MODEL_FILES))
    return load_pairs_for_model(default_model)


def load_generations(method):
    """加载某个方法的生成结果, 返回 samples 列表。"""
    path = os.path.join(cfg.GENERATIONS_DIR, cfg.METHOD_FILES[method])
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到方法 '{method}' 的生成文件: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = data["samples"] if isinstance(data, dict) else data
    for s in samples:
        _validate_sample(s, path)
    return samples


def _validate_sample(s, path):
    """在数据边界处校验训练方交付的数据是否完整。"""
    if "image_id" not in s:
        raise ValueError(f"{path}: 有样本缺少 image_id 字段")
    img = s["image_id"]
    gens = s.get("generations", {})
    trans = s.get("translations", {})
    for lang in cfg.LANGUAGES:
        if lang not in gens or not gens[lang]:
            raise ValueError(f"{path} 图 {img}: 缺少 {lang} 的生成结果")
        if lang != "EN" and (lang not in trans or not trans[lang]):
            raise ValueError(f"{path} 图 {img}: 缺少 {lang} 的英文翻译 (应由翻译方提供)")


def comparable_texts(sample):
    """返回 {语言: 用于比较的英文文本}。EN 用原文, 其它语言用翻译方给的英译。"""
    out = {"EN": sample["generations"]["EN"]}
    for lang in cfg.LANGUAGES:
        if lang != "EN":
            out[lang] = sample["translations"][lang]
    return out


def jaccard(set_a, set_b):
    """两个集合的 Jaccard 相似度。两个都为空视为完全一致 (都没提到该维度)。"""
    set_a, set_b = set(set_a), set(set_b)
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


# 信息维度: 物体 / 属性 / 数量 / 空间关系 / 动作 (用 spaCy 词性+依存自动提取)
# "场景/氛围" 维度难以靠词性自动提取, 留给错误分析里人工判断。
def info_dimensions(doc):
    """从一个 spaCy doc 中提取各信息维度的词集合。"""
    return {
        "objects":    {t.lemma_.lower() for t in doc if t.pos_ in ("NOUN", "PROPN")},
        "attributes": {t.lemma_.lower() for t in doc if t.pos_ == "ADJ"},
        "quantity":   {t.text.lower() for t in doc if t.pos_ == "NUM" or t.dep_ == "nummod"},
        "spatial":    {t.lemma_.lower() for t in doc if t.pos_ == "ADP"},
        "actions":    {t.lemma_.lower() for t in doc if t.pos_ == "VERB"},
    }


INFO_DIMENSIONS = ["objects", "attributes", "quantity", "spatial", "actions"]
