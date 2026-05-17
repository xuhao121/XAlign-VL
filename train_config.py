# ============================================================
#  Qwen2-VL-7B QLoRA 训练配置
#  修改这里的参数即可，不用动训练脚本
# ============================================================

# --- 模型 ---
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"   # 模型路径（HF 名或本地路径）

# --- 数据 ---
TRAIN_DATA = "./Data/Processed Data/qwen_finetune_train.json"            # 预处理好的 JSON/JSONL 文件路径
MAX_LENGTH = 512                            # 最大序列长度（爆显存就降到 512）
MIN_PIXELS = 256 * 28 * 28                   # 图片最小像素数
MAX_PIXELS = 512 * 28 * 28                   # 图片最大像素数（爆显存就降到 256*28*28）

# --- QLoRA ---
LORA_R = 16                                 # LoRA 秩（4/8/16，显存紧就用 4）
LORA_ALPHA = 32                            # 缩放因子（通常 = 2 × r）
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# --- 训练 ---
EPOCHS = 1
BATCH_SIZE = 1                               # 8GB 显存只能用 1
GRAD_ACCUM_STEPS = 16                         # 等效 batch = 1 × 8 = 8
LEARNING_RATE = 1e-4
LR_SCHEDULER = "cosine"
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01

# --- 输出 ---
OUTPUT_DIR = "./output-lora"
SAVE_STEPS = 100
SAVE_TOTAL_LIMIT = 3
LOGGING_STEPS = 10
