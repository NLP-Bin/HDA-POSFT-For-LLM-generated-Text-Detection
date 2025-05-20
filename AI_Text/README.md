# LLaMA-3.1-8B-Instruct 微调与推理流程说明

本项目提供了一个完整的基于 [LLaMA-3.1-8B-Instruct](https://huggingface.co/) 模型的微调与推理流程，涵盖数据预处理、SFT 训练、DPO 训练以及模型预测过程。

任务为：NLPCC2025 共享任务 1: 大型语言模型生成文本检测

---

## 🛠 准备工作

### 1. 模型准备

请先下载并解压 `LLaMA-3.1-8B-Instruct` 模型权重文件，并将路径配置到以下两个脚本中：

- `run_llama.sh`
- `run_test_llama.sh`

修改内容如下所示：

```bash
--model_name_or_path /your/local/path/to/LLaMA-3.1-8B-Instruct
```
### 2.环境配置
>注意`main.py`中，需要将`WANDB_API_KEY`替换为个人用户实际值

确保使用 Python 3.10，并根据 requirements.txt 安装所需依赖：
```bash
conda create -n llama_env python=3.10
conda activate llama_env
pip install -r requirements.txt
```
🚀 使用步骤
### 第一步：数据预处理
运行数据处理脚本：

```bash
bash run_data_process.sh
```
该脚本依次执行以下操作：

数据增强：augment.py 会对原始数据进行增强，转换后的数据将保存在 ./data/augmented_data/augmented_train.json 中。

格式转换：transfer.py 将数据转换为以下两种训练格式，并处理验证集和测试集：

SFT（Supervised Fine-Tuning）格式；

DPO（Direct Preference Optimization）格式；

转换后的数据将保存在 ./data/augmented_inst_data 目录中，供后续训练使用。

### 第二步：模型训练
运行训练脚本：

```bash
bash run_llama.sh
```
该脚本包括两个阶段：

SFT（监督式微调）训练

DPO（直接偏好优化）训练

最终训练好的模型参数将保存在以下目录：

```bash
./checkpoints/sft_llama_dpo/
```

### 第三步：模型推理
运行推理脚本：

```bash
bash run_test_llama.sh
````
该脚本使用训练好的模型进行预测，结果将输出到：

```bash
./results/
```
你可以在该目录中查看模型生成的结果文本。

### 📁 项目结构说明
```bash
.
AI_Text/
├── data/
│   ├── augmented_data/
│   │   └── augmented_train.json # 数据增强后的训练数据
│   ├── augmented_inst_data/
│   │   ├── dpo_train.json       # DPO训练数据
│   │   ├── inst_dev.json        # 验证集
│   │   ├── inst_test.json       # 测试集
│   │   └── sft_train.json       # SFT训练数据
│   ├── dataset_info.json        
│   ├── dev.json                 # 原始验证集
│   ├── test.json                # 原始测试集
│   └── train.json               # 原始训练集
│
├── data_process/
│   ├── augment.py               # 数据增强脚本
│   └── transfer.py              # 数据格式转换脚本
│
├── src/
│   └── downloads.py             # 模型/数据下载脚本（如有使用）
│
├── README.md                    # 项目说明文档
├── requirements.txt             # 环境依赖文件
├── run_data_process.sh          # 数据预处理入口脚本
├── run_llama.sh                 # 模型训练脚本（包含SFT与DPO）
└── run_test_llama.sh            # 测试推理脚本
```
### ⚠️ 注意事项
建议使用具备至少 24GB 显存的 GPU；

数据预处理和训练过程可能占用较多磁盘空间，请确保磁盘剩余充足；

若模型路径或数据路径变化，请对应更新脚本参数。

### 如有问题欢迎提交 Issue 或联系维护者 🙌