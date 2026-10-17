# Qwen3-8B LoRA 编程助教

用 LLaMA-Factory 微调 Qwen3-8B，学习如何解释代码、定位报错、引导学生完成作业

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

##  安装环境并下载模型

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

使用 ModelScope：

```bash
python -m pip install -c constraints.txt modelscope==1.25.0 setuptools==69.5.1
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen3-8B', local_dir='models/Qwen3-8B')"
```

