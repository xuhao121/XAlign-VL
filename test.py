u"""
测试脚本: 生成预测结果和标准答案
用法: python test.py
"""

import json
import torch
from tqdm import tqdm
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from train import load_model
import train_config as cfg

# ============ 配置 ============
TEST_DATA = "./Data/Processed Data/qwen_finetune_test.json"
LORA_PATH = "./output-lora/final"
MAX_NEW_TOKENS = 256

# ============ 加载模型 + LoRA ============
model, processor = load_model()
model = PeftModel.from_pretrained(model, LORA_PATH)
model.eval()

# ============ 加载数据 ============
with open(TEST_DATA, "r", encoding="utf-8") as f:
    data = json.load(f) if TEST_DATA.endswith(".json") else [json.loads(l) for l in f]
print(f"测试数据: {len(data)} 条\n")

# ============ 生成 ============
predictions = []
references = []

for i, item in enumerate(tqdm(data, desc="生成中")):
    # user 消息
    user_msg = item["conversations"][0]
    content = user_msg["content"] if isinstance(user_msg["content"], list) else [{"type": "text", "text": user_msg["content"]}]
    messages = [{"role": "user", "content": content}]

    # 标准答案
    ref = item["conversations"][1]["content"]
    if isinstance(ref, list):
        ref = " ".join(c["text"] for c in ref if c.get("type") == "text")

    # 生成
    try:
        image_inputs, video_inputs = process_vision_info(messages)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=image_inputs or None, videos=video_inputs or None,
                           padding=True, truncation=True, max_length=cfg.MAX_LENGTH, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        pred = processor.batch_decode(output_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
    except Exception as e:
        print(f"\n第 {i} 条失败: {e}")
        pred = ""

    predictions.append(pred)
    references.append(ref)

# ============ 保存 ============
with open("predictions.json", "w", encoding="utf-8") as f:
    json.dump(predictions, f, ensure_ascii=False, indent=2)
with open("references.json", "w", encoding="utf-8") as f:
    json.dump(references, f, ensure_ascii=False, indent=2)

print(f"\n完成! predictions.json 和 references.json 已保存 ({len(predictions)} 条)")
