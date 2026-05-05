# 跨语言生成一致性研究：全部所需知识

> 本文档覆盖这个研究项目需要理解的所有核心概念，从基础到进阶。

---

## 第一章：基础概念

### 1.1 什么是 Embedding（向量表示）

AI 模型不能直接理解文字或图片，它需要把所有东西转化成**数字向量**。

```
"cat" → [0.12, -0.34, 0.78, ..., 0.56]  （一个512维的向量）
"dog" → [0.15, -0.31, 0.72, ..., 0.49]  （和cat很接近）
"car" → [0.89, 0.12, -0.45, ..., 0.23]  （和cat/dog很远）
```

核心思想：**语义相近的东西，向量也相近**。
衡量两个向量相近程度的方法叫 **cosine similarity（余弦相似度）**，值域 [-1, 1]，越大越相似。

### 1.2 什么是 Transformer

Transformer 是 2017 年 Google 提出的神经网络架构，现在几乎所有大模型都基于它。

核心机制是 **Self-Attention（自注意力）**：

```
输入句子："The cat sits on a mat"

对于每个词，模型会问：
  "cat" 和句子里每个其他词的关系有多强？
  → "cat" 和 "sits" 关系强（主谓）
  → "cat" 和 "The" 关系弱（只是冠词）

结果：每个词的表示融合了整个句子的上下文信息
```

Attention 的数学公式（了解即可，不需要手推）：

```
Attention(Q, K, V) = softmax(QK^T / √d) × V

Q = Query（我在找什么）
K = Key（我有什么信息）
V = Value（实际的信息内容）
d = 向量维度（用于缩放）
```

### 1.3 什么是预训练和微调

```
预训练（Pre-training）：
  用海量数据训练模型，让它学会通用的语言/视觉理解
  比如：读完整个互联网的文本
  耗时：数周到数月，需要上千张GPU
  → 我们不做这一步

微调（Fine-tuning）：
  在预训练好的模型基础上，用小量特定数据继续训练
  让模型适应具体任务
  耗时：数小时到数天，一张GPU就够
  → 我们做的是这一步
```

---

## 第二章：语言模型的演化

### 2.1 BERT（2018）

第一个用 Transformer 做预训练的语言模型。

```
训练方式：完形填空（Masked Language Modeling）
  输入："The [MASK] sits on a mat"
  目标：预测 [MASK] = "cat"

特点：双向理解（同时看左边和右边的上下文）
局限：只能理解文本，不能生成文本
```

### 2.2 GPT 系列（2018-今）

和 BERT 相反，GPT 是**单向的生成模型**。

```
训练方式：下一个词预测
  输入："The cat sits on"
  目标：预测下一个词 = "a"

特点：能生成文本（一个词一个词地往下写）
GPT-2 → GPT-3 → GPT-4：越来越大，越来越强
```

### 2.3 XLM-RoBERTa（2020）

多语言版的 BERT，对理解 mCLIP 很重要。

```
训练方式：和 BERT 一样，但训练数据是 100 种语言混合的
  英文："The [MASK] sits"     → cat
  德文："Die [MASK] sitzt"    → Katze
  中文："猫[MASK]在垫子上"     → 坐

关键发现：
  虽然没有明确告诉模型"cat = Katze = 猫"
  但因为这些词出现在相似的上下文中
  模型自动学会了跨语言对齐
  → "cat"、"Katze"、"猫" 的向量在空间中很接近
```

### 2.4 Qwen2.5（2024-2025，阿里巴巴）

我们实验用的模型的语言底座。

```
参数规模：0.5B 到 72B 不等
训练数据：18万亿 tokens
支持语言：29+种，中英文为核心
特点：在数学、代码、多语言上特别强
```

---

## 第三章：视觉语言模型（VLM）

### 3.1 Vision Transformer（ViT）

把 Transformer 用到图片上。

```
传统方法：CNN（卷积神经网络）处理图片
ViT 的做法：

  1. 把图片切成小块（patch），比如 16×16 像素一块
  
     一张 224×224 的图片
     → 切成 14×14 = 196 个 patch
  
  2. 每个 patch 拉平成向量
  
     16×16×3（RGB）= 768 维向量
  
  3. 加上位置编码（告诉模型每个 patch 在图片的哪个位置）
  
  4. 送进 Transformer
     → 通过 Self-Attention，每个 patch 和其他 patch 交互
     → 最终得到每个 patch 的上下文表示
  
  5. 取一个特殊的 [CLS] token 作为整张图的表示
     → 一个向量代表整张图
```

### 3.2 CLIP（2021，OpenAI）

第一个真正把图片和文字对齐的模型，**所有后续 VLM 的基础**。

```
架构：
  图片 → ViT → 图像向量 v（512维）
  文字 → Text Transformer → 文字向量 t（512维）

训练数据：从互联网爬取的 4 亿个（图片, 文字描述）对

训练方式：对比学习（Contrastive Learning）
  同一对的 cos(v, t) 要大
  不同对的 cos(v, t) 要小
```

对比学习的直觉理解：

```
一个 batch 有 N 张图和 N 段文字

正确配对（对角线）：    "一只猫" ↔ 猫的图片  → 相似度要高
错误配对（非对角线）：  "一只猫" ↔ 狗的图片  → 相似度要低

这就是 InfoNCE Loss：
  对每张图，在 N 段文字里找到唯一正确的那段
  本质上是 N 选 1 的分类问题
```

CLIP 的局限：
- 只能算相似度，不能生成文字
- 只支持英文
- 不能回答问题

### 3.3 mCLIP（2022-2023）

在 CLIP 基础上加了多语言能力。

```
做法：知识蒸馏（Knowledge Distillation）

老师模型：CLIP 的英文 Text Encoder（冻结，不改）
学生模型：XLM-RoBERTa（多语言，要训练）

训练数据：英文-其他语言的平行翻译对
  EN: "A dog runs" → CLIP text encoder → 目标向量 t_EN
  DE: "Ein Hund läuft" → XLM-R + Linear → 学生向量 t_DE

训练目标：让 t_DE 尽量靠近 t_EN
  → 这样 t_DE 也能和图片向量匹配

局限：
  所有语言都通过英文间接对齐（英文枢轴问题）
  不是针对图文任务直接优化的
  低资源语言效果差
```

### 3.4 生成式 VLM 的演化

从"只能算相似度"到"能看图说话"。

```
BLIP-2（2023）：
  引入 Q-Former 桥接模块
  32个可学习的 query token 通过交叉注意力从视觉特征中提取信息
  然后输入给 LLM 生成文字
  → 模型第一次能"回答关于图片的问题"

LLaVA（2023）：
  更简单粗暴的方法
  图片 → CLIP ViT → 线性层 → 直接拼接到 LLM 的输入序列
  用 GPT-4 生成的指令数据微调
  → 证明了简单方法也能做得很好

Qwen2.5-VL（2025，我们用的模型）：
  在 LLaVA 思路上大幅改进
  → 接下来详细讲
```

---

## 第四章：Qwen2.5-VL 详解（我们的核心模型）

### 4.1 整体架构

```
输入：图片 + 文字提问
        ↓         ↓
   Vision Encoder  Tokenizer
   （ViT，处理图片）  （文字变token）
        ↓         ↓
   视觉 tokens    文字 tokens
        ↓         ↓
   ┌──────────────────────┐
   │   MLP Merger         │  ← 把视觉tokens压缩，对齐到文字维度
   └──────────┬───────────┘
              ↓
   ┌──────────────────────┐
   │   Qwen2.5 LLM        │  ← 语言模型，处理混合的视觉+文字序列
   │   （自回归生成）       │
   └──────────┬───────────┘
              ↓
          生成文字答案
```

### 4.2 Vision Encoder

```
不是直接用 CLIP 的 ViT，而是从零训练的 NaViT。

关键特性：动态分辨率
  旧方法：所有图片强制缩放到 224×224 → 丢失细节
  Qwen：保持原始分辨率，动态切成不同数量的 patch

  小图（200×300）→ 少量 token
  大图（1200×800）→ 大量 token
  → 信息保留更完整
```

### 4.3 MLP Merger

```
视觉 token 数量太多，直接塞进 LLM 太贵

做法：把相邻的 patch 分组 → 拼接 → 用 MLP 压缩
  比如 4 个相邻 patch → 拼成一个大向量 → MLP → 一个 token
  token 数量减少 4 倍

类比：把一本详细的报告压缩成一份摘要
```

### 4.4 LLM 部分

```
基于 Qwen2.5 的语言模型
自回归生成：一个 token 一个 token 地预测下一个

输入序列长这样：
  [视觉token1][视觉token2]...[视觉tokenN][Describe][this][image]

模型看到这个序列后，逐个生成：
  → [A] → [golden] → [retriever] → [is] → [running] → ...

直到生成结束符 [EOS]
```

### 4.5 多语言能力

```
Qwen2.5 LLM 本身就是多语言的（29+种语言）
所以 Qwen2.5-VL 天然支持多语言图片描述

但问题是：
  - 训练数据中图文对绝大部分是中文和英文
  - 其他语言的图文对很少
  - 这可能导致非中英语言的图片描述质量下降
  → 这就是我们研究的核心问题
```

---

## 第五章：LoRA 微调

### 5.1 为什么需要 LoRA

```
全量微调（Full Fine-tuning）：
  修改模型的所有参数
  Qwen2.5-VL-7B 有 70 亿参数
  需要存储：参数 + 梯度 + 优化器状态 ≈ 60GB 显存
  → 我们的 8GB GPU 完全不够

LoRA 的思路：
  不改原始参数，在旁边加一个"小旁路"
  只训练这个旁路（参数量 < 原模型的 1%）
  → 8GB 显存够了
```

### 5.2 LoRA 的数学原理

```
原始 Transformer 中的一个线性层：
  y = Wx    （W 是 d×d 的矩阵，比如 4096×4096）

LoRA 的改动：
  y = Wx + BAx

  其中：
  A 是 d×r 的矩阵（r 很小，比如 16）
  B 是 r×d 的矩阵

  W：4096×4096 = 16,777,216 个参数（冻结，不训练）
  A：4096×16 = 65,536 个参数（训练）
  B：16×4096 = 65,536 个参数（训练）

  新增参数：131,072（是原来的 0.8%）
```

直觉理解：

```
原始矩阵 W 是一个 4096 维空间的完整变换
LoRA 说：微调只需要在一个 16 维的"子空间"里做调整就够了
A 把输入投影到这个小子空间
B 再投影回来
→ 用极少的参数实现"定向微调"
```

### 5.3 量化（Quantization）

```
模型参数通常是 32 位浮点数（FP32）：
  每个参数占 4 字节
  7B 模型 = 7×10^9 × 4 = 28GB  → 放不下

4bit 量化：
  把每个参数压缩到 4 位
  7B 模型 = 7×10^9 × 0.5 = 3.5GB  → 放得下

代价：精度略有损失，但实践中影响很小

我们的组合：4bit量化（省显存存模型）+ LoRA（省显存做训练）
  → 总共约 6.5GB，8GB 显存能跑
```

---

## 第六章：评估方法

### 6.1 BERTScore（核心评估指标）

```
传统方法（BLEU）：数两段文字有多少词一样
  问题："a big dog" 和 "a large canine" 零分（没有重叠词）
        但其实语义完全一样

BERTScore：用 BERT 的向量来比较语义

步骤：
  1. 句子A的每个词 → BERT → 词向量
     句子B的每个词 → BERT → 词向量

  2. 对A中每个词，找B中最相似的词（余弦相似度最高的）
     → 这叫 Precision

  3. 对B中每个词，找A中最相似的词
     → 这叫 Recall

  4. F1 = 2 × Precision × Recall / (Precision + Recall)

优点：
  "big" 和 "large" 的 BERT 向量很接近 → 高分
  捕捉了语义相似性，不只是词面匹配
```

BERTScore 在我们实验中的用法：

```
比较同一张图的不同语言生成结果：
  EN 生成："A golden retriever runs through a park"
  DE 生成（翻译后）："A dog is running on grass"
  
  BERTScore F1 = 0.85（较高，但有信息差异）
  → "golden retriever" vs "dog"（具体 vs 泛化）
  → "park" vs "grass"（不同侧重）
```

### 6.2 Sentence-BERT（辅助指标）

```
BERTScore 是词级别的匹配
Sentence-BERT 是句子级别的匹配

做法：
  整个句子 → BERT → 取 [CLS] token → 一个向量代表整句话

  cos(sent_A, sent_B) = 句子级语义相似度

优点：一个数字概括整句话的相似度
缺点：不知道具体哪些信息丢失了
```

### 6.3 NLI（自然语言推理，分析不一致来源用）

```
NLI 模型判断两段文字的逻辑关系：

前提（Premise）："A man is playing guitar in a park"
假设（Hypothesis）：

  "A person is making music outdoors"
  → Entailment（蕴含）：前提支持假设

  "Someone is in a park"
  → Entailment：前提也支持这个

  "A woman is cooking"
  → Contradiction（矛盾）：和前提冲突

  "The man is wearing a red hat"
  → Neutral（中性）：前提没提到，无法判断
```

NLI 在我们实验中的用法：

```
EN 生成（作为前提）："A golden retriever runs through a green park 
                      with two people in the background"

DE 生成翻译后（作为假设）："A dog is running on grass"

NLI 判断：Entailment（信息一致，但细节缺失）
  → 缺失了：golden retriever → dog（具体性丢失）
  → 缺失了：two people（人物信息丢失）
  → 缺失了：green park → grass（场景细节丢失）
```

### 6.4 MarianMT（翻译工具）

```
Helsinki-NLP 开发的开源翻译模型
每个语言对一个模型：
  DE→EN: Helsinki-NLP/opus-mt-de-en
  FR→EN: Helsinki-NLP/opus-mt-fr-en
  CS→EN: Helsinki-NLP/opus-mt-cs-en

特点：
  完全离线运行，不需要 API
  模型很小（约 300MB 一个）
  翻译质量中等偏上
  
注意：翻译本身会引入误差
  → 论文里需要讨论这个局限性
  → 可以用 Multi30K 的人工翻译验证 MarianMT 的翻译质量
```

---

## 第七章：Multi30K 数据集

### 7.1 基本信息

```
来源：Flickr30K 的多语言扩展
图片数量：31,014 张日常生活照片
每张图的描述：5个英文描述 + 对应的翻译

语言版本：
  英文 EN：原始描述（人工写的）
  德文 DE：人工翻译
  法文 FR：人工翻译
  捷克文 CS：人工翻译

数据划分：
  训练集：29,000 张
  验证集：1,014 张
  测试集：1,000 张
```

### 7.2 为什么选这个数据集

```
优势：
  1. 同一张图有四种语言的人工翻译（不是机器翻译）
     → 质量有保证
     → 语义应该高度一致
     → 是衡量模型一致性的理想参照

  2. 学术界广泛使用，论文引用多
     → 老师认可度高

  3. 数据量适中（3万张）
     → 足够做统计分析
     → 不至于跑不完

  4. 有标准的 train/val/test split
     → 实验可复现
```

### 7.3 数据格式示例

```
图片：一张狗在公园跑的照片

EN: "A brown dog is running through the tall grass."
DE: "Ein brauner Hund rennt durch das hohe Gras."
FR: "Un chien brun court dans les hautes herbes."
CS: "Hnědý pes běží vysokou trávou."
```

### 7.4 在我们实验中的两种用法

```
用法1（阶段一二）：用图片跑推理
  → 只用图片，不用人工描述
  → 让模型自己生成描述，然后互相比较

用法2（阶段三）：用人工描述做微调
  → 图片 + 非英语 prompt → 训练目标是人工描述
  → 同时用英文描述做一致性约束
```

---

## 第八章：我们实验的技术细节

### 8.1 Prompt 设计

```
我们给模型的指令需要在四种语言中语义完全等价。

设计原则：
  - 简洁明确，减少歧义
  - 避免文化特定的表述
  - 每种语言让母语者检查
  
推荐 prompt：
  EN: "Describe this image in detail."
  DE: "Beschreibe dieses Bild im Detail."
  FR: "Décrivez cette image en détail."
  CS: "Popište tento obrázek podrobně."

可选的补充实验（不同粒度的 prompt）：
  粗粒度："What is in this image?"
  中粒度："Describe this image in detail."
  细粒度："Describe every object, their colors, positions, 
           and actions in this image."
  → 不同粒度下的一致性是否不同？
```

### 8.2 一致性分数的计算

```
每张图有 4 个生成描述（EN/DE/FR/CS）
两两比较，共有 C(4,2) = 6 个语言对：

  EN-DE, EN-FR, EN-CS
  DE-FR, DE-CS
  FR-CS

对每个语言对，计算 BERTScore F1

最终报告：
  每个语言对的平均分 ± 标准差
  整体平均分
```

### 8.3 不一致分析的信息维度

```
我们检查每个描述是否包含以下类型的信息：

  物体（Objects）：提到了什么东西
    例："a dog, a tree, two people"

  属性（Attributes）：颜色、大小、材质
    例："brown dog", "tall tree", "red shirt"

  数量（Quantity）：几个
    例："two people", "three cars"

  空间关系（Spatial）：位置
    例："on the left", "behind the tree", "next to"

  动作（Actions）：在做什么
    例："running", "sitting", "playing"

  场景/氛围（Scene）：整体描述
    例："in a park", "on a sunny day"

检测方法：用 spaCy 做词性标注和依存分析，自动提取上述信息
```

### 8.4 LoRA 微调的训练配置

```
模型：Qwen2.5-VL-7B-Instruct
量化：4bit（bitsandbytes）
LoRA 配置：
  rank = 16（子空间维度）
  alpha = 32（缩放系数）
  target_modules = ["q_proj", "v_proj"]（只对注意力的 Q 和 V 加 LoRA）

训练参数：
  batch_size = 1（显存限制）
  gradient_accumulation_steps = 8（等效 batch_size = 8）
  learning_rate = 2e-4
  epochs = 3-5
  warmup_ratio = 0.1（前10%的步数学习率逐渐升高）

训练数据：
  Multi30K 训练集（约 29K 张图）
  每张图配一种非英语描述作为目标
  → 一轮训练约 29K 步
```

### 8.5 一致性约束 Loss 的设计

```
标准微调 loss：
  L_LM = CrossEntropy(模型生成的token, 目标描述的token)
  → 让模型学会生成正确的非英语描述

一致性约束 loss：
  同一张图，模型生成英文描述 out_EN 和非英语描述 out_X
  out_EN → sentence-BERT → embedding_EN
  out_X 翻译后 → sentence-BERT → embedding_X
  L_CL = 1 - cos(embedding_EN, embedding_X)
  → 让两种语言的描述在语义空间中尽量一致

最终 loss：
  L = L_LM + λ × L_CL
  λ 是超参数，控制一致性约束的强度（建议从 0.1 开始试）

注意：
  sentence-BERT 计算不在计算图里（detach），不会反向传播
  一致性 loss 只通过非英语描述的生成部分传播梯度
  → 这是一个近似方法，但实践中有效
```

---

## 第九章：论文写作需要理解的关键概念

### 9.1 Ablation Study（消融实验）

```
目的：证明你方法的每个组件都有贡献

做法：一次去掉一个组件，看性能变化

我们的消融实验：
  完整方法：L_LM + λ × L_CL              → 分数 A
  去掉一致性 loss：只有 L_LM             → 分数 B
  只用一致性 loss：只有 L_CL             → 分数 C

  如果 A > B > C → 证明两个 loss 都有贡献，且 L_LM 更重要
  如果 A > C > B → 一致性 loss 的贡献更大
```

### 9.2 Baseline（基线方法）

```
用来和你的方法对比的"参照物"

我们的 baselines：
  1. Zero-shot：直接用 Qwen2.5-VL，不做任何微调
  2. Standard LoRA：用 Multi30K 微调，但不加一致性 loss
  3. Translation pipeline：先英文生成，再翻译成目标语言

你的方法需要比这三个 baseline 都好（至少比部分好）
```

### 9.3 Statistical Significance（统计显著性）

```
如果你的方法比 baseline 高了 0.5 分，怎么证明这不是运气？

做法：
  对测试集的结果做 paired t-test 或 bootstrap test
  如果 p-value < 0.05 → 差异是显著的（不是运气）

工具：scipy.stats.ttest_rel
```

### 9.4 Error Analysis（错误分析）

```
论文最加分的部分之一。

做法：
  1. 找出一致性最低的 20 张图
  2. 手动分析不一致的原因
  3. 分类归纳（定性分析）

典型发现示例：
  "模型在描述复杂场景时，英文倾向于列举所有物体，
   而德文倾向于描述整体氛围，导致信息侧重不同。"

这种分析比数字更有说服力。
```

---

## 第十章：工具链速查

### 10.1 需要安装的 Python 库

```
核心模型：
  transformers          — HuggingFace 模型库
  peft                  — LoRA 实现
  bitsandbytes          — 4bit 量化
  accelerate            — 训练加速
  qwen-vl-utils         — Qwen-VL 专用工具

评估工具：
  bert-score            — BERTScore 计算
  sentence-transformers — Sentence-BERT
  sacrebleu             — BLEU（可选的补充指标）

数据处理：
  datasets              — HuggingFace 数据集库
  spacy                 — 词性标注、依存分析
  
翻译：
  Helsinki-NLP/opus-mt  — MarianMT 翻译模型

可视化：
  matplotlib / seaborn  — 画图
  pandas                — 数据表格
```

### 10.2 推荐开发环境

```
首选：Linux（Ubuntu 22.04+）
  bitsandbytes 在 Linux 上兼容性最好

备选：Google Colab Pro（15GB GPU / 40GB A100）
  如果本地显存不够

IDE：VSCode + Cursor
  写代码时可以问 AI
```

---

## 附录：关键术语中英对照

```
跨语言一致性    Cross-lingual Consistency
视觉语言模型    Vision-Language Model (VLM)
多模态大模型    Multimodal Large Language Model (MLLM)
对比学习       Contrastive Learning
知识蒸馏       Knowledge Distillation
低秩适配       Low-Rank Adaptation (LoRA)
量化           Quantization
自回归生成      Autoregressive Generation
注意力机制      Attention Mechanism
余弦相似度      Cosine Similarity
信息检索       Information Retrieval
自然语言推理    Natural Language Inference (NLI)
消融实验       Ablation Study
基线方法       Baseline
微调           Fine-tuning
预训练         Pre-training
嵌入/向量表示   Embedding
```
