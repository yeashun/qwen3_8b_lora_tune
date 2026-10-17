# 从零微调 Qwen3-8B：基于 LoRA 与 LLaMA-Factory 打造专属 AI 编程助教

> 本文面向希望入门大语言模型微调的开发者，以 Qwen3-8B 为基础模型，使用 LLaMA-Factory 和 LoRA 实现一个具有教学风格的 AI 编程助教。重点介绍微调原理、数据集构建、训练配置、推理部署和效果评估。

> 本仓库可直接运行的命令以 [README](README.md) 为入口。以下实践固定使用 LLaMA-Factory v0.9.3，所有命令从本项目根目录执行。模型权重与第三方源码另行下载，不包含在 GitHub 仓库中。

## 一、为什么大模型已经很强了，还需要微调？

如今，通用大语言模型已经能够完成代码生成、程序调试、算法解释等多种任务。

例如，我们直接向 Qwen3-8B 提问：

> Python 中的列表和元组有什么区别？

模型通常能够给出比较完整的解释。

那么，为什么还要微调？

原因在于：**模型能够完成一个任务，不代表它能够按照我们的特定要求稳定地完成任务。**

假设我们要开发一个面向初学者的 AI 编程助教，希望它具备以下特点：

- **语法讲解**：用通俗易懂的语言解释概念，并提供简短示例。
- **代码调试**：准确定位错误，解释错误原因，并给出修改建议。
- **算法辅导**：优先引导学生分析思路，而不是直接给出完整答案。
- **循序渐进**：根据学生当前的理解程度，逐步增加提示。
- **教学风格**：尽量避免生硬、复杂或不必要的专业术语。

通用模型虽然已经具备相关知识，但不一定能始终按照这些教学规范回答。

我们当然可以通过系统提示词（System Prompt）约束模型行为，但提示词不能保证在所有场景下都能稳定生效。

如果希望模型长期遵循某种回答模式，就可以考虑进行监督微调（Supervised Fine-Tuning，SFT）。

这里需要明确：

**本项目微调的主要目标不是让 Qwen3-8B 学会编程，而是让它学会如何更好地教别人编程。**

### 1.1 微调与 RAG、Prompt 的区别

| 方法 | 核心思想 | 适用场景 |
| --- | --- | --- |
| Prompt Engineering | 通过提示词引导模型 | 快速调整回答风格与行为 |
| RAG | 检索外部知识后提供给模型 | 需要最新资料、私有知识 |
| SFT | 利用示例更新模型参数 | 学习固定任务、回答格式与行为 |
| LoRA | 低秩参数高效微调 | 计算资源有限时进行 SFT |

一般建议先使用 Prompt 建立基线，再根据实际效果判断是否需要微调。

---

## 二、监督微调（SFT）的基本原理

### 2.1 大模型是如何生成回答的？

大语言模型本质上是在给定上下文的情况下，预测下一个 Token 出现的概率。

例如，输入：

```text
Python 是一种
```

模型可能预测后面的内容是：

```text
高级编程语言
```

这个过程可以表示为：

```math
P(y_t \mid x,y_{<t};\theta)
```

其中：

- \(x\)：输入提示和上下文。
- \(y_t\)：当前需要预测的 Token。
- \(y_{<t}\)：之前已经生成的 Token。
- \(\theta\)：模型参数。

训练的目标是让模型提高正确 Token 的预测概率。

### 2.2 SFT 是如何进行训练的？

监督微调需要准备输入和对应的期望输出。

例如：

**输入：**

```text
学生：为什么 Python 列表下标会越界？
```

**期望输出：**

```text
Python 列表下标从 0 开始。

例如：
items = [10, 20, 30]

合法下标是 0、1、2。

如果访问 items[3]，就会出现 IndexError。
```

训练过程中，模型会根据输入预测回答，并将预测结果与参考回答进行比较。

常用损失函数为交叉熵损失：

```math
\mathcal{L}_{SFT}
=-\frac{1}{T}\sum_{t=1}^{T}
\log P_{\theta}(y_t^*\mid x,y_{<t}^*)
```

其中：

- \(T\)：参与损失计算的目标 Token 数量。
- \(y_t^*\)：参考回答中的第 \(t\) 个 Token。
- \(P_{\theta}\)：模型对目标 Token 的预测概率。

模型不断通过反向传播更新参数，使其生成结果逐渐接近训练样本中的回答方式。

需要注意的是，**训练损失下降不等于模型实际能力一定提升**，还需要独立的测试数据验证。

---

## 三、LoRA 为什么能够降低微调成本？

### 3.1 全参数微调的问题

Qwen3-8B 包含约 80 亿规模的参数。

如果采用全参数微调，需要对模型中的大量权重进行梯度计算和更新。

除了保存模型参数，还需要额外存储：

- 梯度
- 优化器状态
- 中间激活值

因此，全参数微调通常需要较多 GPU 显存。

### 3.2 LoRA 的核心思想

LoRA（Low-Rank Adaptation，低秩适配）并不直接修改原始权重矩阵，而是冻结原模型参数，额外引入两个可以训练的小矩阵。

对于原始权重：

```math
W_0\in\mathbb{R}^{d\times k}
```

LoRA 将权重更新表示为：

```math
W=W_0+\Delta W
```

其中：

```math
\Delta W=BA
```

两个低秩矩阵分别为：

```math
B\in\mathbb{R}^{d\times r},
\quad A\in\mathbb{R}^{r\times k}
```

这里的 \(r\) 就是 LoRA Rank。

实际计算时通常还引入缩放因子：

```math
h=W_0x+\frac{\alpha}{r}BAx
```

其中 \(\alpha\) 为 LoRA Alpha。

### 3.3 举一个简单的例子

假设某个权重矩阵大小为：

```text
4096 × 4096
```

原始参数数量为：

```text
4096 × 4096 = 16,777,216
```

如果采用 Rank = 8 的 LoRA：

```text
矩阵 A：8 × 4096
矩阵 B：4096 × 8
```

需要训练的参数为：

```text
8 × 4096 + 4096 × 8 = 65,536
```

只相当于原始矩阵参数量的约 0.39%。

当然，这只是单个矩阵的示例，并不代表整个模型的实际可训练参数比例。

### 3.4 LoRA 与 QLoRA

LoRA 主要减少可训练参数以及对应的梯度和优化器开销，但基础模型权重仍然占用显存。

QLoRA 则进一步使用低比特量化保存冻结的基础模型权重，在此基础上训练 LoRA 适配器。

简单理解：

- **LoRA**：冻结原模型，训练少量新增参数。
- **QLoRA**：量化并冻结原模型，再训练少量新增参数。

本教程首先使用 LoRA，便于理解完整流程。

论文参考：[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)。

---

## 四、环境准备

### 4.1 硬件与软件

本教程采用以下技术路线：

| 组件 | 说明 |
| --- | --- |
| 操作系统 | Linux |
| GPU | NVIDIA GPU，推荐较大显存 |
| Python | 3.10 |
| 基础模型 | Qwen/Qwen3-8B |
| 训练框架 | LLaMA-Factory |
| 微调方式 | LoRA |
| 精度 | BF16（需要 GPU 支持） |

显存需求与模型精度、序列长度、Batch Size、梯度检查点和优化器等有关。

如果显存不足，可以考虑缩短序列长度、减小 Batch Size，或者采用 QLoRA。

### 4.2 创建 Conda 环境

```bash
conda create -n qwen3-lora python=3.10 -y
conda activate qwen3-lora
```

确认 Python 版本：

```bash
python --version
```

### 4.3 安装 PyTorch

先通过以下命令检查 GPU 和驱动：

```bash
nvidia-smi
```

然后根据驱动和运行环境，从 PyTorch 官方网站选择匹配的安装命令：

https://pytorch.org/get-started/locally/

安装完成后验证：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

如果最后输出 `True`，说明 PyTorch 可以识别 CUDA。

### 4.4 安装 LLaMA-Factory

下载项目：

```bash
mkdir -p vendor
git clone --depth 1 --branch v0.9.3 https://github.com/hiyouga/LLaMA-Factory.git vendor/LLaMA-Factory
```

建议按官方文档选择兼容的版本组合，避免将不同发行版本的依赖随意混装。

安装：

```bash
python -m pip install -c constraints.txt -e vendor/LLaMA-Factory
```

检查：

```bash
llamafactory-cli version
```

如需图形界面，可以运行：

```bash
llamafactory-cli webui
```

本仓库固定使用 v0.9.3 的 `qwen3` 模板，并通过 `enable_thinking: false` 关闭思考模式。不要混用其他版本的模板名和参数。

### 4.5 下载 Qwen3-8B

官方下载地址：[Hugging Face](https://huggingface.co/Qwen/Qwen3-8B) / [ModelScope](https://modelscope.cn/models/Qwen/Qwen3-8B)。

模型统一保存到 `models/Qwen3-8B`。安装好训练依赖后，可使用已有的 Hugging Face 下载接口：

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen3-8B', local_dir='models/Qwen3-8B')"
```

ModelScope 的替代下载命令见 [README](README.md)。如果服务器无法联网，可以在本机下载，再上传完整模型目录。确认配置、Tokenizer、权重索引和全部 5 个权重分片齐全；本例模型文件约 16.4 GB，不提交到 GitHub。

---

## 五、构建自己的微调数据集

数据质量通常比单纯增加数据数量更重要。

本项目希望模型学会编程教学行为，因此数据不仅要包含正确答案，还应该体现教学过程。

### 5.1 Alpaca 数据格式

LLaMA-Factory 支持 Alpaca 等常见数据格式。

一个典型的 Alpaca 样本包含：

```json
{
  "instruction": "为什么列表下标越界？",
  "input": "items = [1, 2, 3]\nprint(items[3])",
  "output": "列表的合法下标为 0、1、2。items[3] 访问了不存在的第四个元素，因此触发 IndexError。如果想访问第三个元素，应该使用 items[2]。",
  "history": []
}
```

字段解释：

| 字段 | 作用 |
| --- | --- |
| instruction | 用户提出的问题 |
| input | 补充信息，例如代码片段 |
| output | 期望模型生成的回答 |
| history | 之前的多轮问答 |

### 5.2 编写示例数据集

本仓库已提供 `data/code_tutor_seed.json`，包含 32 条样例。下面三个样本仅用于解释格式，无需用它们覆盖现有数据。相同题目的不同阶段需使用相同的 `metadata.group_id`：

```json
[
  {
    "instruction": "Python 中列表和元组有什么区别？",
    "input": "",
    "output": "列表（list）可以修改，元组（tuple）通常不能修改。\n\n例如：\nnums = [1, 2, 3]\nnums[0] = 10\n\n列表允许修改元素。\n\n而对于元组：\nnums = (1, 2, 3)\nnums[0] = 10\n\n则会触发 TypeError。\n\n你可以简单记住：需要修改数据时通常使用列表，不需要修改时可以考虑元组。",
    "history": [],
    "metadata": {
      "group_id": "py_list_tuple"
    }
  },
  {
    "instruction": "帮我看看下面的代码为什么报错。",
    "input": "numbers = [10, 20, 30]\nprint(numbers[3])",
    "output": "问题出在 numbers[3]。Python 的列表下标从 0 开始，这个列表只有三个元素，合法下标分别为 0、1、2。你访问的下标 3 超出了范围。\n\n如果想获取最后一个元素，可以使用 numbers[2]，或者使用 numbers[-1]。",
    "history": [],
    "metadata": {
      "group_id": "py_list_index"
    }
  },
  {
    "instruction": "这是我的作业，帮我写一个冒泡排序程序。",
    "input": "",
    "output": "我们可以先分析冒泡排序的基本思路。\n\n它的核心是比较相邻的两个元素，如果顺序不正确，就交换它们。每完成一轮，相应的较大元素就会逐步移动到后面。\n\n你可以先尝试回答两个问题：\n1. 需要几层循环？\n2. 每一轮应该比较哪些相邻元素？\n\n如果你已经写了一部分代码，可以发给我，我帮你检查。",
    "history": [],
    "metadata": {
      "group_id": "homework_bubble_sort"
    }
  }
]
```

以上仅用于展示数据格式。正式训练应准备更丰富的数据，覆盖不同语言、难度、错误类型及教学阶段。

### 5.3 多轮对话数据

编程助教不能只回答单轮问题，还应该根据学生的尝试进行反馈。

例如：

```json
{
  "instruction": "我写了这个循环，但最后一个元素没有被打印。",
  "input": "for i in range(len(nums) - 1):\n    print(nums[i])",
  "output": "你可以重点检查 range 的结束位置。range(n) 实际生成的是从 0 到 n-1 的整数。你现在写的是 range(len(nums)-1)，所以循环少执行了一次。如果希望访问列表中的每个元素，应该怎样调整结束位置呢？",
  "history": [
    [
      "怎样遍历 Python 列表？",
      "可以通过 for 循环遍历，例如 for x in nums，也可以使用 range(len(nums)) 根据下标访问元素。"
    ]
  ]
}
```

通过这类数据，可以让模型学习不同对话阶段的回答方式。

注意，`history` 必须与所使用的数据模板匹配，训练时是否对历史回答计算损失还取决于相关配置。

### 5.4 数据质量要求

建议重点检查以下内容：

1. **技术正确性**：示例代码应能正常运行，错误解释应准确。
2. **教学一致性**：不同样本遵循相同的教学原则。
3. **任务多样性**：包含概念解释、调试、审查、算法和作业指导。
4. **多轮连贯性**：根据学生已尝试的内容调整提示。
5. **数据去重**：避免同一道题的大量重复样本主导训练。
6. **数据隔离**：同一道题的改写版本不应同时出现在训练集和测试集。

建议按照题目或任务进行分组划分，而不是简单地随机切分所有对话记录。

### 5.5 准备并注册数据集

本仓库通过脚本按题目分组划分训练集和验证集，并自动生成数据注册文件：

```bash
python scripts/prepare_data.py
python -m unittest discover -s tests -v
```

默认得到 25 条训练记录、7 条验证记录，保存到 `data/processed/`。该目录的 `dataset_info.json` 自动注册 `code_tutor_train` 和 `code_tutor_validation`，不需要手工修改框架自带的注册表。

`metadata` 仅用于分组，不进入训练文本；系统提示统一从 `prompts/code_tutor_system.txt` 补入。自定义多轮样本也应设置相应的 `metadata.group_id`。

`data/code_tutor_eval.json` 是 16 个固定生成评测场景，不是训练时计算 eval_loss 的验证集，不应注册为训练或验证数据。

---

## 六、配置 LoRA 微调

### 6.1 创建训练配置文件

本仓库已提供训练配置：

```text
configs/train_lora.yaml
```

写入：

```yaml
# Linux、单张 48GB GPU；所有命令在项目根目录执行。
model_name_or_path: models/Qwen3-8B
flash_attn: sdpa
stage: sft
do_train: true
do_eval: true
finetuning_type: lora
lora_target: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.05

dataset_dir: data/processed
dataset: code_tutor_train
eval_dataset: code_tutor_validation
template: qwen3
enable_thinking: false
train_on_prompt: false
mask_history: true
cutoff_len: 2048
preprocessing_num_workers: 1
dataloader_num_workers: 0

output_dir: outputs/code-tutor/lora
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
per_device_eval_batch_size: 1
num_train_epochs: 3
learning_rate: 0.00005
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
gradient_checkpointing: true
logging_steps: 1
eval_strategy: epoch
save_strategy: epoch
save_total_limit: 2
report_to: none
plot_loss: true
seed: 42
```

如果使用不同版本的 LLaMA-Factory，应先检查当前版本的 Qwen3 示例配置。

`qwen3` 是对话模板，非思考模式通过 `enable_thinking: false` 指定。训练和推理应保持相同设置。

### 6.2 关键参数解释

| 参数 | 含义 |
| --- | --- |
| stage: sft | 使用监督微调 |
| finetuning_type: lora | 使用 LoRA 微调 |
| lora_rank: 8 | 低秩矩阵的秩 |
| lora_alpha: 16 | LoRA 缩放系数 |
| lora_target | 指定注意力与 MLP 的投影层作为 LoRA 目标 |
| cutoff_len: 2048 | 最大样本 Token 长度 |
| learning_rate | 学习率 |
| num_train_epochs | 数据集训练轮数 |
| gradient_accumulation_steps | 梯度累积次数 |
| bf16 | 使用 BF16 精度 |
| gradient_checkpointing | 通过重计算降低激活显存占用 |
| train_on_prompt: false | 不对用户提示部分计算训练损失 |
| mask_history: true | 不对历史回答计算训练损失 |

### 6.3 Batch Size 与梯度累积

例如：

```yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
```

单卡训练时，每处理 16 个小批次才执行一次优化器更新。

因此，其有效 Batch Size 可以近似理解为：

```text
Effective Batch Size
= Batch Size × Gradient Accumulation Steps × GPU 数量
```

在此例中：

```text
1 × 16 × 1 = 16
```

梯度累积可以在不显著增加单次前向传播显存需求的情况下，模拟更大的训练批次。

---

## 七、开始训练

### 7.1 选择 GPU

例如使用编号为 0 的 GPU：

```bash
export CUDA_VISIBLE_DEVICES=0
```

如果模型和数据都已经保存在本地，可以根据需要启用离线模式：

```bash
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

### 7.2 先进行小规模试跑

正式训练前建议先执行少量更新步骤：

```bash
llamafactory-cli train configs/train_lora.yaml \
  max_steps=3 \
  gradient_accumulation_steps=4 \
  output_dir=outputs/code-tutor/smoke
```

试跑主要用于检查：

- 模型能否正确加载。
- 数据集能否正常读取。
- Tokenizer 与模板是否兼容。
- GPU 能否完成反向传播。
- LoRA 适配器能否成功保存。

试跑正常只能说明训练流程基本可用，并不能证明微调有效。

### 7.3 正式训练

执行：

```bash
llamafactory-cli train configs/train_lora.yaml
```

训练过程中可以观察 Training Loss、Evaluation Loss（如果启用验证）、学习率以及 GPU 显存使用情况。

### 7.4 如何理解 Loss？

通常，训练 Loss 下降说明模型越来越容易预测训练数据中的参考回答。

但是：

- Training Loss 持续下降，可能只是模型记住了训练样本。
- Evaluation Loss 变差，可能意味着过拟合。
- Loss 改善，也不保证生成的代码更正确。
- 教学风格是否改善，需要通过实际问答评估。

因此，不应该仅凭 Loss 曲线判断微调成功与否。

### 7.5 训练产物

训练完成后，LoRA 适配器通常保存在：

```text
outputs/code-tutor/lora/
```

目录中可能包含：

```text
adapter_config.json
adapter_model.safetensors
training_args.yaml
checkpoint-xxx/
```

具体文件取决于框架版本及保存配置。

其中：

- `adapter_model.safetensors`：LoRA 适配器权重。
- `adapter_config.json`：LoRA 结构和配置。
- `checkpoint-xxx`：训练检查点，可用于恢复训练。

**LoRA 适配器不是完整的 Qwen3-8B 模型，推理时通常还需要原始基础模型。**

---

## 八、加载微调后的模型进行推理

### 8.1 启动 LoRA 对话

本仓库的对话脚本统一加载基础模型、可选适配器和助教提示，无需额外创建推理 YAML：

```bash
python scripts/chat_tutor.py
```

可以提问：“我刚开始学 Python，能帮我解释什么是 for 循环吗？”或提供具体报错。输入 `:clear` 清空历史，`:quit` 退出。

### 8.2 对照原始模型

退出当前对话、释放 GPU 显存后运行：

```bash
python scripts/chat_tutor.py --base
```

这里的 Base 是原始 Qwen3-8B 不加载本项目适配器，并非另一个名为 Qwen3-8B-Base 的模型。两边使用相同的系统提示，比较的是 LoRA 在提示词基础上带来的额外收益。

### 8.3 固定题目自动对照

```bash
python scripts/chat_tutor.py --compare
```

脚本逐题关闭、开启 LoRA，使用相同的系统提示、固定历史、采样参数和种子生成两份回答。结果保存到 `outputs/code-tutor/comparison_seed42.json` 和 `.md`；相同种子再次运行会覆盖对应报告，使用 `--seed 43` 可另存一组结果。

每题从其固定历史开始，不沿用上一题生成内容。达到生成长度上限的回答会标记，不能把截断内容当作完整答案评分。

---

## 九、如何评估微调效果？

评估应该从多个维度展开，而不是单纯比较回答长短。

### 9.1 建立测试集

可以设计以下几类题目：

| 类型 | 示例问题 | 评估目标 |
| --- | --- | --- |
| 语法解释 | Python 列表和元组有什么区别？ | 是否通俗易懂 |
| 错误定位 | 为什么出现 IndexError？ | 是否找到实际原因 |
| 代码审查 | 这段 C++ 代码有什么风险？ | 是否识别关键问题 |
| 算法辅导 | 如何实现二分查找？ | 是否先引导思路 |
| 作业指导 | 直接帮我完成排序作业 | 是否合理控制提示深度 |
| 多轮教学 | 我按照提示修改后仍然报错 | 是否结合历史继续指导 |

评估集应尽量选择训练中没有出现过的题目及相近改写。

### 9.2 比较两种模型

使用以下两个模型：

- **Base**：原始 Qwen3-8B。
- **LoRA**：原始 Qwen3-8B + 编程助教适配器。

保持相同的系统提示、生成配置以及测试数据。

建议记录两边的完整回答，而不仅仅是主观印象。

### 9.3 设计评价标准

对于编程助教，可以从五个维度进行评价。

| 维度 | 评价内容 |
| --- | --- |
| 技术正确性 | 代码、概念和错误分析是否正确 |
| 教学清晰度 | 是否适合初学者理解 |
| 具体帮助 | 是否针对实际问题提供可执行建议 |
| 教学策略 | 是否根据提问场景调整帮助深度 |
| 多轮连贯性 | 是否理解此前的尝试和反馈 |

本仓库当前按逐题 rubric 做定性判读，不自动打分。作为扩展，可以采用 1～5 分的人工评分，但需要在评测前为每个分值制定明确标准。

例如，代码调试场景中：

- 1 分：没有识别问题，或给出错误建议。
- 3 分：识别主要错误，但解释不充分。
- 5 分：正确定位错误，并给出清晰、具体的修复建议。

教学策略和技术正确性应分别评估，不能因为模型回答得更像老师，就忽略技术错误。

### 9.4 增加客观代码评测

对于可以执行的编程任务，还可以自动运行测试用例。

例如，对模型生成的函数检查：

```python
def add(a, b):
    return a + b

assert add(1, 2) == 3
assert add(-1, 1) == 0
assert add(0, 0) == 0
```

只有全部测试通过，才能认为这个示例满足对应测试要求。

执行模型生成代码时，应使用隔离的沙箱、权限限制和超时机制，避免直接在训练服务器的高权限环境中执行不可信代码。

### 9.5 如何判断微调是否值得？

建议同时考虑：

1. **质量收益**：技术正确性和教学体验是否改善。
2. **稳定性**：不同题目和多次生成是否具有一致性。
3. **泛化能力**：未见过的任务能否保持效果。
4. **训练成本**：微调需要多少显存和计算资源。
5. **部署成本**：加载适配器是否带来额外的管理和维护成本。

如果原始模型通过提示词已经能够稳定满足需求，那么微调的实际收益可能有限。

---

### 9.6 本次实验结果

项目使用 16 个固定评估场景，其中包含同一道括号作业的首轮提示、错误反馈和参考实现三个阶段，并非 16 道完全独立的题。起始种子 42 对应逐题 seed 42～57，两边逐题相同；它不是每道题都重复测试 16 次。

本轮由助手按检查要点逐题评阅，未经独立人工复核或盲审：

| 本轮判读 | 数量 |
| --- | --- |
| LoRA 局部更好 | 2 |
| 原始模型局部更好 | 1 |
| 相当 | 8 |
| 两边均未全覆盖且未见明确优劣 | 5 |

两边的全要点覆盖均为 9/16，这不是代码正确率。32 份回答均未截断。第 7 题 LoRA 对计数变化的解释更具体；第 8 题两边都漏掉迭代器失效；第 13 题两边都先口头拒绝、再给完整作业代码。第 11 题 LoRA 仅少了一个错误示例，边界条件仍未完全达标。

因此，本轮没有观察到明确的整体微调收益，也不足以给出统计显著性结论。当前 32 条种子样本主要用于演示流程，不能把“能够训练和对话”当作“助教能力稳定提升”。完整材料见[逐题评阅报告](reports/comparison_seed42.md)和[结构化结果](reports/comparison_seed42.json)。

---

## 十、可选步骤：合并 LoRA 权重

默认情况下，LoRA 推理需要加载：

```text
原始 Qwen3-8B + LoRA Adapter
```

本案例没有执行合并，仓库也未额外提供合并配置。以下是可选的手动示例；如需合并，可使用 LLaMA-Factory 的 Export 功能。

创建：

```text
configs/code_tutor_merge.yaml
```

配置如下：

```yaml
model_name_or_path: models/Qwen3-8B
adapter_name_or_path: outputs/code-tutor/lora
finetuning_type: lora
template: qwen3
enable_thinking: false
trust_remote_code: false

export_dir: outputs/code-tutor/merged
export_size: 5
export_device: cpu
export_legacy_format: false
```

执行：

```bash
llamafactory-cli export configs/code_tutor_merge.yaml
```

合并后可以将模型作为完整权重加载。

注意：

- 合并时应使用与适配器匹配的原始基础模型。
- 不应直接将量化后的基础模型当作普通全精度模型合并。
- 合并会产生额外的磁盘和内存开销。
- 合并后的模型应重新进行推理一致性检查。

如果只是本地实验，通常没有必要立即合并权重。

---

## 十一、常见问题

### 11.1 CUDA Out of Memory

显存不足时，可以优先尝试：

1. 降低 `cutoff_len`。
2. 减小 `per_device_train_batch_size`。
3. 启用 `gradient_checkpointing`。
4. 改用 QLoRA。
5. 检查其他进程是否占用了 GPU 显存。

其中，减小序列长度通常是降低训练显存占用的有效方法之一。

### 11.2 ModuleNotFoundError: llamafactory

可能原因包括：

- 没有激活正确的 Conda 环境。
- LLaMA-Factory 尚未正确安装。
- 可编辑安装对应的源码目录发生变化或被删除。
- 当前 Python 与 pip 不属于同一个环境。

建议检查：

```bash
which python
which pip
pip show llamafactory
```

如果使用 `pip install -e`，需要保证对应的源码目录一直存在。

### 11.3 Loss 下降，但回答没有明显变化

可能原因包括：

- 数据量不足。
- 训练数据与基础模型原有行为非常接近。
- 训练样本覆盖的任务较少。
- 系统提示词已经能够实现相同效果。
- 模型只记住了训练样本，没有泛化到新问题。

建议先检查数据质量，并使用固定测试集比较原模型和微调模型。

### 11.4 模型开始输出奇怪格式

优先检查：

- 数据模板是否与模型匹配。
- 训练和推理是否采用相同的 Chat Template。
- 是否混用了不同的思考模式。
- 训练数据是否存在不规范的格式。
- 是否使用了错误的 Adapter。

Qwen3 支持思考与非思考模式，使用时应注意模板和生成配置的一致性。

### 11.5 为什么不直接训练整个 8B 模型？

全参数微调能够更新更多参数，但不一定是小规模任务的最优选择。

如果目标主要是调整输出风格、教学流程和特定任务行为，LoRA 往往是更容易入门、训练成本也更低的选择。

是否需要全参数微调，应该由任务需求、数据规模、资源预算和实际评估结果共同决定。

---

## 十二、总结

通过本教程，我们梳理了使用 Qwen3-8B 进行 LoRA 监督微调的主要流程：

```text
选择基础模型
      ↓
明确任务目标
      ↓
设计训练样本
      ↓
构建训练集与验证集
      ↓
配置 LoRA 参数
      ↓
小规模试跑
      ↓
正式微调
      ↓
加载 Adapter 推理
      ↓
与原模型进行对照评估
      ↓
根据评测结果决定是否迭代
```

需要记住三个核心结论：

**第一，微调不一定是为了学习新知识。**

对于已经具备编程能力的通用大模型，微调可以更多地关注行为模式、教学方式和回答的一致性。

**第二，LoRA 的价值在于降低训练成本。**

通过冻结原模型，只训练少量低秩参数，可以在资源有限的情况下开展监督微调实验。

**第三，训练完成不等于微调有效。**

只有通过独立评估，才能判断模型是否真正改善了目标任务中的表现。

对于刚开始学习大模型训练的开发者而言，完整走通一次数据准备、LoRA 微调、模型推理和效果评估的流程，比单纯追求更低的训练 Loss 更有价值。

---

## 参考资料

1. [Qwen3-8B 官方模型卡](https://huggingface.co/Qwen/Qwen3-8B)
2. [LLaMA-Factory GitHub](https://github.com/hiyouga/LLaMA-Factory)
3. [LLaMA-Factory SFT 文档](https://llamafactory.readthedocs.io/en/latest/getting_started/sft.html)
4. [LLaMA-Factory 推理文档](https://llamafactory.readthedocs.io/en/latest/getting_started/inference.html)
5. [LoRA 原始论文](https://arxiv.org/abs/2106.09685)
6. [QLoRA 论文](https://arxiv.org/abs/2305.14314)
7. [PyTorch 官方安装指南](https://pytorch.org/get-started/locally/)

---

*本文主要用于学习和理解大语言模型微调的基本原理与工程流程。示例配置需要根据所使用的框架版本、硬件条件和实际数据集进行调整。*
