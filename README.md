# Qwen3-8B LoRA 编程助教

用 LLaMA-Factory 微调 Qwen3-8B，学习如何解释代码、定位报错、引导学生完成作业。面向新手，只走一条流程：**准备数据 → LoRA 训练 → 对话比较**。

运行环境：Linux、Python 3.10、单张支持 BF16 的 48GB GPU。已在服务器完成三步训练试跑与 LoRA 对话，并进行了 16 个固定场景的对照评阅；本轮未观察到明确的整体微调收益。

- [博客教学案例：原理、流程与评测](Qwen3-8B_LoRA微调教学案例.md)
- [本轮逐题评阅报告](reports/comparison_seed42.md) / [结构化结果](reports/comparison_seed42.json)

仓库只包含代码、配置、教学样本和小型评测报告。模型权重、适配器、第三方框架源码、缓存及离线安装包均不提交；模型下载方法见下文。

## 项目文件

```text
configs/train_lora.yaml         唯一的训练配置
data/code_tutor_seed.json       训练样例，包含多轮对话
data/code_tutor_eval.json       16 个固定评估场景及检查要点
prompts/code_tutor_system.txt   训练和对话共用的助教提示
scripts/prepare_data.py         按题目划分训练集、验证集
scripts/chat_tutor.py           基础模型或 LoRA 对话、固定题目对照
tests/                         数据分组、多轮记录和对照流程检查
constraints.txt                固定训练依赖版本
reports/                       本轮对照回答与助手逐题评阅
Qwen3-8B_LoRA微调教学案例.md     博客教学案例
```

先获取项目，后续命令均在项目根目录执行：

```bash
git clone https://github.com/yeashun/qwen3_8b_lora_tune.git
cd qwen3_8b_lora_tune
```

## 1. 安装环境并下载模型

```bash
conda create -n qwen3-lora python=3.10 -y
conda activate qwen3-lora
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
mkdir -p vendor
git clone --depth 1 --branch v0.9.3 https://github.com/hiyouga/LLaMA-Factory.git vendor/LLaMA-Factory
python -m pip install -c constraints.txt -e vendor/LLaMA-Factory
python -m pip check
```

### 下载模型（不包含在 GitHub 仓库中）

下载 **Qwen/Qwen3-8B**，将完整目录保存为 `models/Qwen3-8B`。本例权重约 16.4 GB，包含 5 个分片；已有完整模型目录时直接使用它。

官方下载地址：

- [ModelScope：Qwen/Qwen3-8B](https://modelscope.cn/models/Qwen/Qwen3-8B)
- [Hugging Face：Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)

方式一，使用 ModelScope：

```bash
python -m pip install -c constraints.txt modelscope==1.25.0 setuptools==69.5.1
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen3-8B', local_dir='models/Qwen3-8B')"
```

方式二，使用训练环境已有的 Hugging Face 下载工具：

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen3-8B', local_dir='models/Qwen3-8B')"
```

两种方式选一种即可。服务器无法联网时，在本机完成下载，再通过 MobaXterm/SFTP 将整个 `Qwen3-8B` 文件夹上传到服务器项目的 `models/` 下。目录需包含 `config.json`、tokenizer 文件、`model.safetensors.index.json` 和全部权重分片，不要只复制单个权重文件。

框架源码也可从[官方 v0.9.3 源码包](https://github.com/hiyouga/LLaMA-Factory/archive/refs/tags/v0.9.3.zip)下载、解压至 `vendor/LLaMA-Factory`，再按前面的命令安装。若使用可编辑安装，安装后须保留此源码目录。

模型已下载到本地时，可在当前终端启用离线模式，并将缓存留在项目内：

```bash
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export XDG_CACHE_HOME="$PWD/.cache"
export HF_HOME="$PWD/.cache/huggingface"
mkdir -p .cache/tmp
export TMPDIR="$PWD/.cache/tmp"
```

## 2. 准备数据

先读 `data/code_tutor_seed.json`。每条样本包括当前问题和期望回答：

```json
{
  "instruction": "为什么列表下标越界？",
  "input": "items = [1, 2, 3]\nprint(items[3])",
  "output": "列表有 3 个元素，合法下标是 0、1、2；这里的下标 3 越界了。",
  "history": [],
  "metadata": {"group_id": "python_list_index"}
}
```

`history` 存之前的 `[学生提问, 助教回答]` 对；没有历史时可省略。相同题目的多轮记录和改写题必须使用相同 `group_id`。`metadata` 不进入模型输入。

```bash
python scripts/prepare_data.py
```

脚本固定随机种子 42，按题目分组取约 20% 作为验证集，生成 `data/processed/` 下的 `train.json`、`validation.json`、`dataset_info.json`。修改样本或助教提示后，重新运行即可。

## 3. 训练

先选定实际分配给你的 GPU，做三步试跑：

```bash
export CUDA_VISIBLE_DEVICES=0  # 改为实际分配给你的 GPU 编号
llamafactory-cli train configs/train_lora.yaml max_steps=3 gradient_accumulation_steps=4 output_dir=outputs/code-tutor/smoke
```

确认能加载模型、输出 loss 并保存适配器，再运行正式训练：

```bash
llamafactory-cli train configs/train_lora.yaml
```

训练结果保存在 `outputs/code-tutor/lora`。LoRA 只保存适配器，对话时还需要原来的基础模型。新实验可在命令末尾加 `output_dir=outputs/code-tutor/run2` 指定另一个目录。

先理解这几个参数即可：

| 参数 | 含义 |
| --- | --- |
| `lora_rank: 8` | LoRA 适配器的秩 |
| `learning_rate: 0.00005` | 每次更新的学习率 |
| `num_train_epochs: 3` | 遍历训练集 3 轮 |
| `cutoff_len: 2048` | 样本总 token 长度上限，包含提示、历史和回答 |
| `gradient_accumulation_steps: 16` | batch 为 1 时，累计 16 次再更新参数 |

训练和对话统一关闭思考模式；`mask_history: true` 只对当前回答计算损失。新增长代码或长对话时，需检查是否超过 2048 token，超长样本会被截断。

## 4. 对话并比较效果

微调后的模型：

```bash
python scripts/chat_tutor.py
```

基础模型对照：

```bash
python scripts/chat_tutor.py --base
```

两个命令分别运行，用相同问题和对话历史比较结果。输入 `:clear` 开始新题，`:quit` 退出；不要让两份模型同时占用同一张卡。

自动运行固定题目对照：

```bash
python scripts/chat_tutor.py --compare
```

只加载一份基础模型与适配器，逐题关闭、开启 LoRA 生成两份回答。两边使用相同的系统提示、固定历史、采样参数和种子，每题从空会话开始拼接测试历史。16 道题包括同一道括号作业的首轮提示、错误尝试反馈、充分尝试后的参考实现三个阶段。

结果保存在 `outputs/code-tutor/comparison_seed42.json` 和 `comparison_seed42.md`。打开 Markdown，参考每题的 `rubric` 比较技术正确性、教学帮助和给代码的时机，填写人工结论。达到生成长度上限的回答会标记，需要补测，不能直接当作完整回答评分。

使用 `python scripts/chat_tutor.py --compare --seed 43` 可换一组种子复测，结果保存到对应的 `comparison_seed43.*`。同一个种子再次运行会覆盖该种子的报告。固定种子用于控制采样随机性，不保证跨设备、依赖版本得到完全一致的结果；先用一轮初筛，再对发现的差异复测。

助教行为由统一提示和训练样本引导，仍需要人工检查模型回答。32 条数据只能演示训练，不要仅凭 loss 下降判断教学效果。

仓库内的 `reports/` 是本次实验的评阅示例，由助手按检查要点逐题判读，未经独立人工复核或盲审；不作为统计显著提升的结论。重新运行对照只会生成 `outputs/` 下的报告，不会自动改写仓库内的示例报告。

本地无需 GPU 的数据检查：

```bash
python -m unittest discover -s tests -v
```
