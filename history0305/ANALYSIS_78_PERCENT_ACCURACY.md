# 牙科选择题模型准确率停滞在78.31%的原因分析

## 摘要
经过200次迭代训练，模型准确率仍为78.31%，高于初始基线但未达预期提升。根据代码分析，问题源于**训练目标与评估指标的不匹配**。

---

## 问题根源分析

### 1. **双目标训练设计缺陷**（最严重的问题）

**当前实现（train_dental_lora7.py 第180行）：**
```python
teacher_ans = get_teacher_answer(prompt, teacher_model, teacher_tokenizer)
target = f"{teacher_ans}|{new_answer}"  # 双目标格式，如 "B|B" 或 "A|E"
target_new.append(target)
```

**问题：**
- 模型训练目标是输出 `"B|B"` 这样的字符串
- 但测试评估（autoTest7.py）期望模型只输出单个字母 `"B"`
- 双目标信号 `teacher_ans|human_answer` 包含两个不同的答案：
  - 当 `teacher_ans ≠ human_answer` 时（如 `"A|E"`），模型看到自相矛盾的目标
  - 这会导致训练梯度冲突，模型无法学到一致的映射

**数据统计（来自日志）：**
```
Case 815: target="B|A"  (teacher=B, human=A，不一致)
Case 823: target="B|D"  (teacher=B, human=D，不一致)
Case 839: target="B|C"  (teacher=B, human=C，不一致)
```
大量样本中 teacher 和 human 答案不一致，导致训练信号混乱。

---

### 2. **评估指标的严格匹配问题**

**BaseEditor 中的评估流程（easyeditor/evaluate/evaluate_utils.py）：**
- `test_prediction_acc()` 使用 **token-level 完全匹配**
- 它期望模型生成的 token 序列完全等于 target_new
- 对于 target `"B|B"` 的情况，模型必须生成 4 个 token：`"B"`, `"|"`, `"B"` 才能判定为正确

**实际情况：**
- 在生成时，模型可能只输出 `"B"` 或 `"B " ` 等变体
- 由于不完全匹配，评估失败

---

### 3. **过拟合到训练集**

**日志中的现象：**
```
case_id: 808, pre: {'rewrite_acc': [0.0]}, post: {'rewrite_acc': [1.0]}
case_id: 809, pre: {'rewrite_acc': [0.0]}, post: {'rewrite_acc': [1.0]}
...
```

**含义：**
- `pre`: 训练前该样本失败（0%）
- `post`: 训练后该样本成功（100%）
- 模型逐个样本地学习了训练集，导致快速的单样本编辑成功
- 但这种高度过拟合无法泛化到测试集

**对比：**
- 训练准确率：单样本 100% ✓
- 测试准确率：全集 78.31% ✗
- **差距：21.69个百分点**（典型过拟合信号）

---

### 4. **数据增强与选项打乱的问题**

**train_dental_lora7.py 第165-175行：**
```python
# 选项随机打乱（增强鲁棒性）
shuffled_options = options.copy()
random.shuffle(shuffled_options)

# 重建选项文本和答案映射
option_text = "\n".join([f"{opt[0]}. {opt[1]}" for opt in shuffled_options])
# 找到打乱后的答案
new_answer = ""
for opt_char, _ in shuffled_options:
    if opt_char == original_answer:
        new_answer = opt_char
        break
```

**潜在问题：**
- 选项字母（A/B/C/D/E）本身被打乱，但选项顺序变了
- 模型看到的是无序的选项和变化的答案映射
- 没有考虑选项**位置信息**（第一个选项、第二个选项等）
- 这导致模型学不到稳定的选择逻辑

---

### 5. **不足的训练步数与学习率**

**当前超参数（train_dental_lora7.py 第326-335行）：**
```python
total_steps = 200
lr = 5e-5  # 相对较低
batch_size = 1  # 极小批量
weight_decay = 0.01
lora_dropout = 0.1
```

**问题：**
- 200步 ≈ 2-3个 epoch（数据量约800条）
- 极小批量（batch_size=1）导致梯度噪声大
- 学习率 5e-5 较保守，可能训练不充分
- 没有学习率调度（warmup/decay）

---

## 对比另一个脚本

**autoTest7.py 中的评估：**
```python
SAMPLING_PARAMS = SamplingParams(
    temperature=0.0,  # 确定性
    max_tokens=10,
    stop=["<|endoftext|>", "</s>"]
)

def extract_answer_char(answer_text):
    """提取纯字母（A/B/C/D/E）"""
    for char in answer_text.strip().upper():
        if char in ["A", "B", "C", "D", "E"]:
            return char
    return ""
```

**问题：**
- autoTest7.py 只评估单个字母输出
- 但模型在训练时被教导输出 `"B|B"` 格式
- **训练目标与评估目标不匹配！**

---

## 数字证据

### 训练过程中的准确率模式
```
训练期间（单样本）：
- pre_edit rewrite_acc:  0.0 ~ 0.33  （训练前）
- post_edit rewrite_acc: 1.0         （训练后，100%）

测试集评估：
- 最终准确率：78.31%
```

### 规律
- **每个样本的训练准确率 = 100%**
- **整体测试准确率 = 78.31%**
- 这说明模型学到的是样本特定的知识，而非通用能力

---

## 建议的解决方案

### 方案1：统一训练与评估目标（推荐）
```python
# 修改 train_dental_lora7.py
# 只用人工答案，不用双目标
target_new = new_answer  # 仅 "B"，不是 "B|B"
# ground_truth 也只用单个答案
```

### 方案2：改进双目标训练机制
```python
# 如果必须用双目标，使用多任务学习
# 而不是直接拼接
target_new = json.dumps({"teacher": teacher_ans, "student": new_answer})
# 在评估时也按此格式评估
```

### 方案3：增加训练数据和迭代次数
```python
total_steps = 500  # 从200增加到500（5个epoch）
batch_size = 4     # 从1增加到4
lr = 1e-4          # 从5e-5增加到1e-4
# 添加学习率预热和衰减
```

### 方案4：改进正则化与数据增强
```python
# 不要打乱选项字母本身，只打乱选项顺序
# 这样 A/B/C/D/E 始终对应固定的选项
# 但选项出现的顺序会变化
```

### 方案5：使用对比学习或其他先进技术
```python
# 使用对比损失函数，让模型学习
# 正确答案与错误答案之间的区别
```

---

## 预期改进

如果采用**方案1（统一目标）**：
- 消除梯度冲突 → 准确率可能提升至 82-85%
- LoRA 秩从 16 增加到 32 + 方案3 → 可能达到 88-92%
- 完整重设计（方案4+5） → 可能达到 93-96%

---

## 总结

| 问题 | 严重性 | 影响 | 解决难度 |
|------|--------|------|---------|
| 双目标训练不匹配 | ⚠️⚠️⚠️ 极高 | 梯度冲突、过拟合 | 低 |
| 评估指标不一致 | ⚠️⚠️⚠️ 极高 | 测试准确率虚低 | 低 |
| 选项打乱策略 | ⚠️⚠️ 中高 | 模型学不到位置信息 | 低 |
| 训练步数不足 | ⚠️⚠️ 中 | 欠拟合 | 低 |
| 批量大小太小 | ⚠️ 中 | 梯度噪声大 | 低 |

**建议优先级：** 方案1 → 方案4 → 方案3 → 方案5
