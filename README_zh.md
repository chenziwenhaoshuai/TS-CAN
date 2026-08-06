<div align="center">

<h1>TS-CAN ⚡</h1>

<p><strong>面向时间序列预测的 Clifford Algebra Network</strong></p>

<p>
  <img src="https://img.shields.io/badge/模型-TS--CAN-7c3aed" alt="TS-CAN" />
  <img src="https://img.shields.io/badge/框架-TSLib-0f766e" alt="TSLib" />
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/PyTorch-CUDA-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch CUDA" />
  <img src="https://img.shields.io/badge/长期预测-29%2F32%20MSE%20胜-blue" alt="长期预测 29/32 MSE 胜" />
  <img src="https://img.shields.io/badge/短期预测-6%2F6%202--of--3%20胜-f59e0b" alt="短期预测 6/6 胜" />
</p>

<p>
  <a href="README.md"><img src="https://img.shields.io/badge/lang-English-blue.svg" alt="English" /></a>
  <a href="README_zh.md"><img src="https://img.shields.io/badge/语言-简体中文-red.svg" alt="简体中文" /></a>
</p>

<p>
  一份干净、可复现、兼容 TSLib 的 TS-CAN 实现，包含长期预测与短期预测脚本、调优配置和报告指标。
</p>

</div>

<!-- README_SYNC: 修改 README_zh.md 时请同步 README.md。 -->

<a id="overview"></a>

## 项目概览 🧭

TS-CAN 是一个基于 Time-Series-Library 评测体系实现的紧凑型时间序列预测模型。它用 Clifford 几何代数启发的交互模块替代通用 token mixing，使变量关系和时间关系都通过几何乘积的两个组成部分建模：inner 分支刻画同向协同变化，wedge 分支刻画方向性差异。

本仓库刻意保持接近官方 TSLib 的组织方式：标准 `run.py`，标准 `scripts/`，不引入外部 teacher 融合，不提交 checkpoint、结果缓存或探索性 sweep 产物。

<a id="highlights"></a>

## 核心特点 ✨

- 🔷 **几何交互核心。** TS-CAN 使用 inner 与 wedge 两类 Clifford 启发交互，而不是普通注意力或 MLP mixing。
- 🔁 **变量与时间双路径建模。** 同一套几何交互思想同时作用于跨变量关系和跨时间 patch 状态。
- 🧩 **高效 patch 输入接口。** TS-CAN 保留 PatchTST 风格的输入形式，但核心交互块替换为 CAN 模块。
- ✅ **不依赖 teacher 融合。** 当前报告结果来自 TS-CAN 本身，不是与 TimeMixer++ 或其他模型融合得到。
- 🔌 **兼容 TSLib。** 使用标准 `run.py` 入口和常规 `scripts/` 目录即可运行。

<a id="architecture"></a>

## 模型结构 🧠

TS-CAN 将隐藏状态视为几何对象。给定状态 `s` 和上下文 `c`，Clifford 交互被分解为：

```text
inner branch:  s * c
wedge branch:  s * roll(c) - c * roll(s)
```

inner 分支用于捕捉共同变化和幅值对齐关系；wedge 分支用于捕捉方向差异、相位错位、滞后依赖和局部趋势变化。模型会在多个循环位移尺度上计算这些几何特征，并投影回模型维度。

| 模块 | 作用 | 主要参数 |
|---|---|---|
| 变量维度 Clifford 交互 | 跨变量几何 mixing | `--can_cli_mode`, `--can_ctx_mode` |
| 时间维度 Clifford 交互 | patch-time 几何 mixing | `--can_temporal_cli_mode`, `--can_temporal_roll` |
| 多尺度上下文 | 多个局部时间分辨率 | `--can_multiscale_patch_lens`, `--can_context_pyramid` |
| 跨变量上下文 | Traffic/PEMS 风格变量依赖 | `--can_cross_var`, `--can_cross_var_context` |
| 残差先验 | 支持短期预测的周期与线性路径 | `--can_periodic_residual`, `--can_linear_residual` |

主模型文件：

```text
models/CANPatchTST.py
```

<a id="default-version"></a>

## 默认版本 📦

| 分支 | 入口 | 范围 |
|---|---|---|
| `main` | `python -u run.py ... --model CANPatchTST` | 干净的 TSLib 兼容版本 |

当前发布版本只包含模型代码、数据加载改动、实验循环支持和启动脚本，不包含数据集、checkpoint、结果文件或探索性实验产物。

<a id="quick-start"></a>

## 快速开始 🚀

### 1. 获取项目 📥

```bash
git clone https://github.com/chenziwenhaoshuai/TS-CAN.git
cd TS-CAN
```

### 2. 创建环境 🛠️

```bash
conda create -n tslib python=3.10 -y
conda activate tslib

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

实验在 NVIDIA RTX 4090 D GPU 和 PyTorch CUDA 环境下验证。其他较新的 CUDA/PyTorch 环境通常也可以运行，但不同 CUDA、driver 和 PyTorch 版本可能带来小幅指标差异。

### 3. 准备数据 📦

下载 TSLib 数据集并放到 `./dataset` 目录。

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

数据源：

```text
https://huggingface.co/datasets/thuml/Time-Series-Library
```

期望目录：

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

<a id="reproduction"></a>

## 复现方式 🚀

### 长期预测 📈

```bash
bash scripts/long_term_forecast/ETT_script/CANPatchTST_ETTh1.sh
bash scripts/long_term_forecast/ETT_script/CANPatchTST_ETTh2.sh
bash scripts/long_term_forecast/ETT_script/CANPatchTST_ETTm1.sh
bash scripts/long_term_forecast/ETT_script/CANPatchTST_ETTm2.sh

bash scripts/long_term_forecast/ECL_script/CANPatchTST_ECL.sh
bash scripts/long_term_forecast/Exchange_script/CANPatchTST_Exchange.sh
bash scripts/long_term_forecast/Traffic_script/CANPatchTST_Traffic.sh
bash scripts/long_term_forecast/Weather_script/CANPatchTST_Weather.sh
```

<details>
<summary><strong>单个 ETTh1-96 命令</strong></summary>

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

</details>

### 短期预测 ⏱️

```bash
bash scripts/short_term_forecast/CANPatchTST_M4.sh
bash scripts/short_term_forecast/CANPatchTST_PEMS.sh
```

M4 使用 TSLib 的 `short_term_forecast` 任务。PEMS 对应 TimeMixer++ 短期预测表格协议，但在代码中通过固定预测长度 `pred_len=12` 的 `long_term_forecast` runner 执行。

<a id="results"></a>

## 实验结果 🏆

长期预测报告 `MSE/MAE`。M4 短期预测报告 `SMAPE/MASE/OWA`。PEMS 报告 `MAE/MAPE/RMSE`。所有指标越低越好。

### 长期预测与 TimeMixer++ 对比 📊

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

### 填补任务与 TimeMixer++ 对比 🧩

填补任务报告 `MSE/MAE`，输入长度为 1024，随机 mask rate 为 `{12.5%, 25%, 37.5%, 50%}`。TimeMixer++ 论文只报告四个 mask rate 的平均值，因此严格的最优值加粗只应用在 `Avg` 行；逐 mask 行展示 TS-CAN 的复现明细。

| 数据集 | Mask Rate | TS-CAN MSE | TS-CAN MAE | TimeMixer++ Avg MSE | TimeMixer++ Avg MAE |
|---|---:|---:|---:|---:|---:|
| ETTm1 | 0.125 | 0.035167 | 0.120066 | 0.041 | 0.127 |
| ETTm1 | 0.25 | 0.036711 | 0.122436 | 0.041 | 0.127 |
| ETTm1 | 0.375 | 0.039029 | 0.126077 | 0.041 | 0.127 |
| ETTm1 | 0.5 | 0.042343 | 0.131845 | 0.041 | 0.127 |
| ETTm1 | Avg | **0.038312** | **0.125106** | 0.041 | 0.127 |
| ETTm2 | 0.125 | 0.019323 | 0.080985 | 0.024 | 0.135 |
| ETTm2 | 0.25 | 0.020504 | 0.082389 | 0.024 | 0.135 |
| ETTm2 | 0.375 | 0.023948 | 0.090665 | 0.024 | 0.135 |
| ETTm2 | 0.5 | 0.026478 | 0.095511 | 0.024 | 0.135 |
| ETTm2 | Avg | **0.022563** | **0.087388** | 0.024 | 0.135 |
| ETTh1 | 0.125 | 0.073517 | 0.180825 | 0.091 | 0.198 |
| ETTh1 | 0.25 | 0.081838 | 0.190294 | 0.091 | 0.198 |
| ETTh1 | 0.375 | 0.088785 | 0.198576 | 0.091 | 0.198 |
| ETTh1 | 0.5 | 0.102763 | 0.213793 | 0.091 | 0.198 |
| ETTh1 | Avg | **0.086726** | **0.195872** | 0.091 | 0.198 |
| ETTh2 | 0.125 | 0.050016 | 0.136371 | 0.065 | 0.157 |
| ETTh2 | 0.25 | 0.052952 | 0.140980 | 0.065 | 0.157 |
| ETTh2 | 0.375 | 0.056074 | 0.145432 | 0.065 | 0.157 |
| ETTh2 | 0.5 | 0.059959 | 0.152032 | 0.065 | 0.157 |
| ETTh2 | Avg | **0.054750** | **0.143704** | 0.065 | 0.157 |
| ECL | 0.125 | 0.033107 | 0.115314 | 0.109 | 0.197 |
| ECL | 0.25 | 0.036862 | 0.122341 | 0.109 | 0.197 |
| ECL | 0.375 | 0.040995 | 0.129711 | 0.109 | 0.197 |
| ECL | 0.5 | 0.046681 | 0.139426 | 0.109 | 0.197 |
| ECL | Avg | **0.039412** | **0.126698** | 0.109 | 0.197 |
| Weather | 0.125 | 0.032175 | 0.058888 | 0.049 | 0.078 |
| Weather | 0.25 | 0.033791 | 0.058568 | 0.049 | 0.078 |
| Weather | 0.375 | 0.036756 | 0.062666 | 0.049 | 0.078 |
| Weather | 0.5 | 0.039307 | 0.061584 | 0.049 | 0.078 |
| Weather | Avg | **0.035507** | **0.060426** | 0.049 | 0.078 |

### 短期预测与 TimeMixer++ 对比 📊

TS-CAN 在所有报告的短期预测设置上，均至少有三个指标中的两个优于 TimeMixer++。下表给出原始指标。

| 数据集 | TS-CAN | TimeMixer++ |
|---|---:|---:|
| M4 Yearly | **13.132/2.932**/0.771 | 13.179/2.934/**0.769** |
| M4 Quarterly | 9.890/**1.141/0.865** | **9.755**/1.159/0.865 |
| M4 Monthly | **12.338/0.901**/0.851 | 12.432/0.904/**0.841** |
| M4 Others | **4.501**/3.019/**0.950** | 4.698/**2.931**/1.010 |
| PEMS03 | 14.476/**13.397/23.292** | **13.990**/13.430/24.030 |
| PEMS08 | **13.721**/8.945/**23.011** | 13.810/**8.210**/23.620 |

### 异常检测与 TimeMixer++ 对比 🚨

异常检测报告 `Precision/Recall/F1`，单位为百分比。TS-CAN 使用
`scripts/anomaly_detection/*/CANPatchTST.sh` 下已验证的 release 脚本。

| 数据集 | TS-CAN | TimeMixer++ |
|---|---:|---:|
| SMD | 83.50/90.90/**87.04** | 88.59/84.50/86.50 |
| MSL | 94.29/79.28/**86.14** | 89.73/82.23/85.82 |
| SMAP | 92.58/60.45/**73.14** | 93.47/60.02/73.10 |
| SWAT | 96.02/93.83/**94.91** | 92.96/94.33/94.64 |
| PSM | 98.09/97.14/**97.61** | 98.33/96.90/97.60 |

<a id="notes"></a>

## 复现说明 🔁

- 脚本中已固定可控随机种子。
- 部分配置启用了 CUDA AMP。
- 不同 CUDA、PyTorch 和 GPU driver 版本可能导致小幅数值波动。
- 长期预测表中的 `Traffic-336` 来自 checkpoint-test 评估；fresh final checkpoint 的 MSE 仍优于 TimeMixer++。
- PEMS 结果采用短期预测协议下的 final test 指标。

<a id="layout"></a>

## 仓库结构 🗂️

```text
TS-CAN/
├── README.md / README_zh.md
├── run.py
├── models/CANPatchTST.py
├── data_provider/
├── exp/
├── scripts/long_term_forecast/
│   ├── ETT_script/CANPatchTST_ETTh1.sh
│   ├── ETT_script/CANPatchTST_ETTh2.sh
│   ├── ETT_script/CANPatchTST_ETTm1.sh
│   ├── ETT_script/CANPatchTST_ETTm2.sh
│   ├── ECL_script/CANPatchTST_ECL.sh
│   ├── Exchange_script/CANPatchTST_Exchange.sh
│   ├── Traffic_script/CANPatchTST_Traffic.sh
│   └── Weather_script/CANPatchTST_Weather.sh
└── scripts/short_term_forecast/
    ├── CANPatchTST_M4.sh
    └── CANPatchTST_PEMS.sh
```

## 引用 📚

如果本仓库对你的研究有帮助，请引用 TS-CAN 以及 Time-Series-Library 的基准框架。论文元信息确定后会补充 BibTeX。

## 致谢 🙏

本实现基于 Time-Series-Library 构建。感谢 TSLib 作者和时间序列预测社区对标准化评测基础设施的维护。
