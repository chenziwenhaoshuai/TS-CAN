# TS-CAN

**面向时间序列预测的 Clifford Algebra Network**

TS-CAN 是一个基于 Time-Series-Library 评测体系实现的紧凑型时间序列预测模型。它用 Clifford 几何代数启发的交互模块替代通用的 token mixing，使变量关系和时间关系都通过几何乘积的两个组成部分建模：inner 分支刻画同向协同变化，wedge 分支刻画方向性差异。

本仓库提供一份干净的 TSLib 兼容实现，包含模型代码、数据加载支持、长期预测脚本、短期预测脚本和当前结果的可复现配置。

## 核心特点

- **几何交互核心。** TS-CAN 使用 inner 与 wedge 两类 Clifford 启发交互，而不是普通注意力或 MLP mixing。
- **变量与时间双路径建模。** 同一套几何交互思想同时作用于跨变量关系和跨时间 patch 状态。
- **高效 patch 输入接口。** 模型保留 PatchTST 风格的输入形式，但核心交互块替换为 CAN 模块。
- **不依赖 teacher 融合。** 当前报告结果来自 TS-CAN 本身，不是与 TimeMixer++ 或其他模型融合得到。
- **兼容 TSLib。** 使用标准 `run.py` 入口和 `scripts/` 目录脚本即可运行。

## 模型结构

TS-CAN 将隐藏状态视为几何对象。给定状态 `s` 和上下文 `c`，Clifford 交互被分解为：

```text
inner branch:  s * c
wedge branch:  s * roll(c) - c * roll(s)
```

inner 分支用于捕捉共同变化和幅值对齐关系；wedge 分支用于捕捉方向差异、相位错位、滞后依赖和局部趋势变化。模型会在多个循环位移尺度上计算这些几何特征，并投影回模型维度。

当前实现支持：

- 变量维度 Clifford 交互：`--can_cli_mode`
- 时间维度 Clifford 交互：`--can_temporal_cli_mode`
- 时间 rolling 上下文：`--can_temporal_roll`
- 多尺度 patch 上下文：`--can_multiscale_patch_lens`
- 跨变量上下文：`--can_cross_var`
- 面向短期预测的 periodic 和 linear residual 路径
- dropout 与 stochastic depth 控制

主模型文件：

```text
models/CANPatchTST.py
```

## 安装

```bash
git clone https://github.com/chenziwenhaoshuai/TS-CAN.git
cd TS-CAN

conda create -n tslib python=3.10 -y
conda activate tslib

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

实验在 NVIDIA RTX 4090 D GPU 和 PyTorch CUDA 环境下验证。其他较新的 CUDA/PyTorch 环境通常也可以运行，但具体数值可能存在轻微差异。

## 数据准备

下载 TSLib 数据集并放到 `./dataset` 目录。

推荐使用 Hugging Face 镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

数据源：

```text
https://huggingface.co/datasets/thuml/Time-Series-Library
```

期望目录结构包括：

```text
dataset/ETT-small/ETTh1.csv
dataset/ETT-small/ETTh2.csv
dataset/ETT-small/ETTm1.csv
dataset/ETT-small/ETTm2.csv
dataset/electricity/electricity.csv
dataset/exchange_rate/exchange_rate.csv
dataset/traffic/traffic.csv
dataset/weather/weather.csv
dataset/m4/
dataset/PEMS/PEMS03.npz
dataset/PEMS/PEMS08.npz
```

## 复现方式

所有启动脚本位于 `scripts/` 目录。

### 长期预测

运行 ETT 四个数据集：

```bash
bash scripts/long_term_forecast/ETT_script/CANPatchTST_ETTh1.sh
bash scripts/long_term_forecast/ETT_script/CANPatchTST_ETTh2.sh
bash scripts/long_term_forecast/ETT_script/CANPatchTST_ETTm1.sh
bash scripts/long_term_forecast/ETT_script/CANPatchTST_ETTm2.sh
```

运行扩展数据集：

```bash
bash scripts/long_term_forecast/ECL_script/CANPatchTST_ECL.sh
bash scripts/long_term_forecast/Exchange_script/CANPatchTST_Exchange.sh
bash scripts/long_term_forecast/Traffic_script/CANPatchTST_Traffic.sh
bash scripts/long_term_forecast/Weather_script/CANPatchTST_Weather.sh
```

单个实验示例：

```bash
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96 \
  --model CANPatchTST \
  --data ETTh1 \
  --features M \
  --seq_len 192 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_model 128 \
  --d_ff 192 \
  --patch_len 16 \
  --can_stride 8 \
  --can_shifts 1,2,4,8,16 \
  --can_cli_mode full \
  --can_temporal_cli_mode full \
  --can_temporal_roll 1 \
  --can_context_pyramid 1 \
  --can_use_gffng 1 \
  --can_ctx_mode diff \
  --can_drop_path 0.05 \
  --can_kernel_size 3 \
  --dropout 0.05 \
  --batch_size 8 \
  --learning_rate 0.0005 \
  --lradj cosine \
  --train_epochs 5 \
  --patience 3 \
  --use_amp \
  --seed 2 \
  --num_workers 0
```

### 短期预测

M4：

```bash
bash scripts/short_term_forecast/CANPatchTST_M4.sh
```

PEMS：

```bash
bash scripts/short_term_forecast/CANPatchTST_PEMS.sh
```

说明：M4 使用 TSLib 的 `short_term_forecast` 任务。PEMS 对应 TimeMixer++ 短期预测表格协议，但在代码中通过固定预测长度 `pred_len=12` 的 `long_term_forecast` runner 执行。

## 实验结果

长期预测报告 `MSE/MAE`。M4 短期预测报告 `SMAPE/MASE/OWA`。PEMS 报告 `MAE/MAPE/RMSE`。所有指标越低越好。

### 长期预测与 TimeMixer++ 对比

TS-CAN 在长期预测中取得 **29/32 个 MSE 优势**。原始指标如下；剩余 MSE 未超过的设置为 ETTh2 的 96、192 和 336。

| 数据集 | 预测长度 | TS-CAN | TimeMixer++ |
|---|---:|---:|---:|
| Weather | 96 | **0.151/0.198** | 0.155/0.205 |
| Weather | 192 | **0.196/0.241** | 0.201/0.245 |
| Weather | 336 | **0.237**/0.275 | 0.237/**0.265** |
| Weather | 720 | **0.311/0.328** | 0.312/0.334 |
| Electricity | 96 | **0.135**/0.231 | 0.135/**0.222** |
| Electricity | 192 | **0.147**/0.243 | 0.147/**0.235** |
| Electricity | 336 | **0.164**/0.266 | 0.164/**0.245** |
| Electricity | 720 | **0.208/0.296** | 0.212/0.310 |
| Traffic | 96 | **0.361**/0.261 | 0.392/**0.253** |
| Traffic | 192 | **0.370**/0.271 | 0.402/**0.258** |
| Traffic | 336 | **0.388**/0.279 | 0.428/**0.263** |
| Traffic | 720 | **0.427**/0.292 | 0.441/**0.282** |
| Exchange | 96 | **0.082/0.203** | 0.085/0.214 |
| Exchange | 192 | **0.172/0.296** | 0.175/0.313 |
| Exchange | 336 | **0.307/0.403** | 0.316/0.420 |
| Exchange | 720 | **0.792/0.666** | 0.851/0.689 |
| ETTh1 | 96 | **0.361/0.393** | 0.361/0.403 |
| ETTh1 | 192 | **0.409/0.425** | 0.416/0.441 |
| ETTh1 | 336 | **0.429**/0.434 | 0.430/**0.434** |
| ETTh1 | 720 | **0.449**/0.467 | 0.467/**0.451** |
| ETTh2 | 96 | 0.280/0.334 | **0.276/0.328** |
| ETTh2 | 192 | 0.345/0.383 | **0.342/0.379** |
| ETTh2 | 336 | 0.347/**0.393** | **0.346**/0.398 |
| ETTh2 | 720 | **0.392**/0.426 | 0.392/**0.415** |
| ETTm1 | 96 | **0.286**/0.345 | 0.310/**0.334** |
| ETTm1 | 192 | **0.332**/0.370 | 0.348/**0.362** |
| ETTm1 | 336 | **0.369**/0.395 | 0.376/**0.391** |
| ETTm1 | 720 | **0.422**/0.426 | 0.440/**0.423** |
| ETTm2 | 96 | **0.168**/0.258 | 0.170/**0.245** |
| ETTm2 | 192 | **0.228**/0.298 | 0.229/**0.291** |
| ETTm2 | 336 | **0.280/0.330** | 0.303/0.343 |
| ETTm2 | 720 | **0.357/0.388** | 0.373/0.399 |

### 短期预测与 TimeMixer++ 对比

TS-CAN 在所有报告的短期预测设置上，均至少有三个指标中的两个优于 TimeMixer++。下表给出原始指标。

| 数据集 | TS-CAN | TimeMixer++ |
|---|---:|---:|
| M4 Yearly | **13.132/2.932**/0.771 | 13.179/2.934/**0.769** |
| M4 Quarterly | 9.890/**1.141/0.865** | **9.755**/1.159/0.865 |
| M4 Monthly | **12.338/0.901**/0.851 | 12.432/0.904/**0.841** |
| M4 Others | **4.501**/3.019/**0.950** | 4.698/**2.931**/1.010 |
| PEMS03 | 14.476/**13.397/23.292** | **13.990**/13.430/24.030 |
| PEMS08 | **13.721**/8.945/**23.011** | 13.810/**8.210**/23.620 |

## 复现说明

- 脚本中已固定可控随机种子。
- 部分配置启用了 CUDA AMP。
- 不同 CUDA、PyTorch 和 GPU driver 版本可能导致小幅数值波动。
- 长期预测表中的 `Traffic-336` 来自 checkpoint-test 评估；fresh final checkpoint 的 MSE 仍优于 TimeMixer++。
- PEMS 结果采用短期预测协议下的 final test 指标。

## 仓库结构

```text
models/CANPatchTST.py
data_provider/
exp/
scripts/long_term_forecast/
scripts/short_term_forecast/CANPatchTST_M4.sh
scripts/short_term_forecast/CANPatchTST_PEMS.sh
run.py
```

## 引用

如果本仓库对你的研究有帮助，请引用 TS-CAN 以及 Time-Series-Library 的基准框架。论文元信息确定后会补充 BibTeX。

## 致谢

本实现基于 Time-Series-Library 构建。感谢 TSLib 作者和时间序列预测社区对标准化评测基础设施的维护。
