
'''
Qwen2-VL-7B QLoRA 微调训练脚本
适用环境: 8GB 显存 (RTX 5060)
用法:     python train.py
配置:     修改 train_config.py
'''

import json
import torch

from torch.utils.data import Dataset
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info
import train_config as cfg


# ================================================================
#  第一步: 加载模型 (4-bit 量化)
# ================================================================
def load_model():
    print(f"正在加载模型: {cfg.MODEL_NAME}")
    print("使用 4-bit NF4 量化...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg.MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.config.use_cache = False
    processor = AutoProcessor.from_pretrained(
        cfg.MODEL_NAME,
        min_pixels=cfg.MIN_PIXELS,
        max_pixels=cfg.MAX_PIXELS,
    )

    print("模型加载完成 ✓")
    return model, processor


# ================================================================
#  第二步: 挂载 QLoRA
# ================================================================
def setup_lora(model):
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True)
    lora_config = LoraConfig(
        r=cfg.LORA_R,
        lora_alpha=cfg.LORA_ALPHA,
        lora_dropout=cfg.LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg.LORA_TARGET_MODULES,
        exclude_modules=["visual.*"],  # 排除视觉编码器
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ================================================================
#  第三步: 数据集加载
#  支持 .json (整个文件是一个数组) 和 .jsonl (每行一条)
# ================================================================
class VLDataset(Dataset):
    """
    预期数据格式 (每条数据):
    {
      "conversations": [
        {
          "role": "user",
          "content": [
            {"type": "image", "image": "file:///path/to/img.jpg"},
            {"type": "text", "text": "问题..."}
          ]
        },
        {
          "role": "assistant",
          "content": [
            {"type": "text", "text": "回答..."}
          ]
        }
      ]
    }

    也兼容 assistant content 直接是字符串的格式:
    {"role": "assistant", "content": "回答..."}
    """

    def __init__(self, data_path, processor):
        self.processor = processor
        self.data = self._load(data_path)
        print(f"数据集加载完成: {len(self.data)} 条样本 ✓")

    def _load(self, path):
        if path.endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]
        else:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        conversations = item["conversations"]

        # 标准化消息格式
        messages = []
        for turn in conversations:
            role = turn["role"]
            raw_content = turn["content"]

            # 兼容 content 是字符串的情况
            if isinstance(raw_content, str):
                content = [{"type": "text", "text": raw_content}]
            else:
                content = raw_content

            messages.append({"role": role, "content": content})

        # processor 处理
        prompt_text = self.processor.apply_chat_template([messages[0]],tokenize=False, add_generation_prompt=True)
        prompt_ids = self.processor.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_ids)
        full_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[full_text],
            images=image_inputs if image_inputs else None,
            videos=video_inputs if video_inputs else None,
            padding="max_length",
            truncation=True,
            max_length=cfg.MAX_LENGTH,
            return_tensors="pt",
        )
        # 构建 labels (和 input_ids 一致, Trainer 会自动做 shift)
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100  # padding 位置不计算 loss

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        # 图片/视频相关字段 (可能不存在)
        if "pixel_values" in inputs and inputs["pixel_values"] is not None:
            result["pixel_values"] = inputs["pixel_values"]
        if "image_grid_thw" in inputs and inputs["image_grid_thw"] is not None:
            result["image_grid_thw"] = inputs["image_grid_thw"]
        return result


def vl_data_collator(batch):
    """
    pixel_values 和 image_grid_thw 每个样本维度不同,
    不能用默认的 stack, 必须用 cat 拼接。
    """
    result = {}

    # 常规字段: stack
    result["input_ids"] = torch.stack([b["input_ids"] for b in batch])
    result["attention_mask"] = torch.stack([b["attention_mask"] for b in batch])
    result["labels"] = torch.stack([b["labels"] for b in batch])

    # 多模态字段: cat (拼接所有样本的 patches)
    if "pixel_values" in batch[0] and batch[0]["pixel_values"] is not None:
        result["pixel_values"] = torch.cat([b["pixel_values"] for b in batch], dim=0)
    if "image_grid_thw" in batch[0] and batch[0]["image_grid_thw"] is not None:
        result["image_grid_thw"] = torch.cat([b["image_grid_thw"] for b in batch], dim=0)

    return result
# ================================================================
#  第四步: 训练
# ================================================================
def train():
    # 加载模型
    model, processor = load_model()

    # 挂载 LoRA
    model = setup_lora(model)

    # 加载数据
    dataset = VLDataset(cfg.TRAIN_DATA, processor)
    # 训练参数
    training_args = TrainingArguments(
        output_dir=cfg.OUTPUT_DIR,
        num_train_epochs=cfg.EPOCHS,
        per_device_train_batch_size=cfg.BATCH_SIZE,
        gradient_accumulation_steps=cfg.GRAD_ACCUM_STEPS,
        gradient_checkpointing=True,
        learning_rate=cfg.LEARNING_RATE,
        lr_scheduler_type=cfg.LR_SCHEDULER,
        warmup_ratio=cfg.WARMUP_RATIO,
        weight_decay=cfg.WEIGHT_DECAY,
        fp16=True,
        logging_steps=cfg.LOGGING_STEPS,
        save_steps=cfg.SAVE_STEPS,
        save_total_limit=cfg.SAVE_TOTAL_LIMIT,
        dataloader_num_workers=2,
        remove_unused_columns=False,         # 多模态必须 False
        optim="paged_adamw_8bit",            # 8-bit 优化器省显存
        max_grad_norm=1.0,
        report_to="none",
        dataloader_pin_memory=False,         # 省内存
    )

    # 开始训练
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=vl_data_collator,
    )
    print("\n" + "=" * 50)
    print("  开始训练")
    print("=" * 50)
    trainer.train()

    # 保存 LoRA 权重
    final_path = f"{cfg.OUTPUT_DIR}/final"
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    print(f"\n训练完成! LoRA 权重已保存到: {final_path}")


if __name__ == "__main__":
    train()
