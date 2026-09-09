# 多模型融合时间序列预测系统

## 云部署访问
[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit-FF4B4B)](https://time-series-forecast-dw8kazyz3pveab3znxfd4x.streamlit.app/)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://time-series-forecast-dw8kazyz3pveab3znxfd4x.streamlit.app/)

&gt; 基于 StatsForecast 与 Prophet 的多模型集成时间序列预测流水线，支持 per-SKU 自动选优、测试集评估与上线预测，配备 Streamlit 交互式可视化看板。

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 目录

1. [项目背景](#项目背景)
2. [系统架构](#系统架构)
3. [快速开始](#快速开始)
4. [项目结构](#项目结构)
5. [模型说明](#模型说明)
6. [关键结果](#关键结果)
7. [技术栈](#技术栈)
8. [许可证](#许可证)

---

## 项目背景

在零售与供应链场景中，准确预测每个 SKU 的未来销量是库存优化与业务决策的核心。不同 SKU 的销量模式差异巨大（有的强季节性、有的平稳、有的噪声大），单一模型难以在所有 SKU 上表现最优。

本项目解决的核心问题：

- 为每个 SKU 自动选择最合适的预测模型
- 支持多模型对比与集成评估
- 提供从离线评估到上线预测的完整闭环
- 具备可视化的预测结果展示能力

---

## 系统架构

整个系统分为 7 个模块，按流水线顺序执行：

```text
config.py                ← 全局配置（路径、参数、超参数搜索空间）
    │
data_loader.py           ← 数据加载与预处理
    │
    ├── statsforecast_model.py   ← StatsForecast 4 种统计模型（CV + 预测）
    │
    └── prophet_model.py         ← Prophet 模型（超参数搜索 + CV + 预测）
    │
ensemble.py              ← 合并模型 SMAPE、per-SKU 最优模型选择
    │
evaluate.py              ← 测试集评估 + 全量数据重训 + 上线预测
    │
pipeline.py              ← 主流程入口，串联所有模块（一键运行）
    │
streamlit_app.py         ← Streamlit 交互式可视化看板
```

---

## 快速开始

### 环境配置

推荐使用 Python 3.10 或以上版本，使用 conda 或 venv 创建虚拟环境：

```bash
# 克隆仓库
git clone https://github.com/yourusername/time-series-forecast.git
cd time-series-forecast

# 创建虚拟环境（推荐）
conda create -n ts_forecast python=3.10
conda activate ts_forecast

# 安装依赖
pip install -r requirements.txt
```

依赖清单（`requirements.txt`）：

```text
pandas>=1.5.0
numpy>=1.24.0
statsforecast>=1.7.0
prophet>=1.1.5
utilsforecast>=0.2.0
streamlit>=1.28.0
plotly>=5.15.0
tqdm>=4.65.0
```

### 数据准备

将数据文件放入 `data/` 目录，需包含以下两个文件：

- `data/train.csv` — 训练集
- `data/test.csv` — 测试集

数据格式要求：

| 列名 | 类型 | 说明 |
| ---- | ---- | ---- |
| unique_id | str | SKU 唯一标识 |
| ds | datetime | 日期 |
| y | float | 目标变量（如销量） |
| country | str | 国家（可选，Prophet 节假日用） |
| store | str | 商店（可选，看板展示用） |
| product | str | 产品（可选，看板展示用） |

### 运行流水线

一键运行完整预测流水线：

```bash
python pipeline.py
```

执行后将在 `output/` 目录下生成以下文件：

| 文件 | 说明 |
| ---- | ---- |
| final_model_comparison.csv | 各 SKU 在各模型上的 SMAPE 对比 |
| final_per_sku_best_forecast.csv | per-SKU 最优模型的预测结果 |
| test_set_evaluation.csv | 测试集评估结果 |
| retrain_final_forecast.csv | 全量数据重训后的最终上线预测 |
| sku_smape_detail.csv | 各 SKU 最优模型及 SMAPE 明细 |
| best_model_distribution.csv | 最优模型选择分布 |

### 启动可视化看板

```bash
streamlit run streamlit_app.py
```

看板功能：

- **全球地图** — 按国家展示 SKU 数量分布，圆点大小表示 SKU 密度
- **多层级 Bottom-Up 汇总** — 按国家 × 商店 × 产品逐级下钻，查看汇总销量曲线
- **时序对比** — 训练集 / 测试集 / 预测三阶段对比，含 95% 置信区间
- **交互式图表** — 所有图表支持 hover 查看详情、缩放、框选

---

## 项目结构

```text
time-series-forecast/
│
├── data/                          # 数据目录（需自行放入 train.csv / test.csv）
│
├── output/                        # 输出目录（运行后自动生成）
│
├── config.py                      # 全局配置参数
├── data_loader.py                 # 数据加载与预处理
├── statsforecast_model.py         # StatsForecast 统计模型
├── prophet_model.py               # Prophet 模型（超参数搜索）
├── ensemble.py                    # 模型集成与选优
├── evaluate.py                    # 测试集评估与上线预测
├── pipeline.py                    # 主流水线入口
├── streamlit_app.py               # Streamlit 可视化看板
│
├── requirements.txt               # 依赖清单
├── README.md                      # 本文件
└── LICENSE                        # 许可证
```

---

## 模型说明

### StatsForecast 统计模型

使用 `statsforecast` 库的 4 种自动模型，每种模型均设置 `season_length=7`（周季节性）：

| 模型 | 说明 |
| ---- | ---- |
| AutoETS | 指数平滑，自动选择趋势/季节成分 |
| AutoARIMA | 差分自回归移动平均，自动确定 p/d/q 阶数 |
| AutoTheta | Theta 方法，适合有趋势的时序数据 |
| AutoCES | 复数指数平滑，能够捕捉复杂季节模式 |

### Prophet 模型

Facebook 开源的时间序列预测工具，适用于含节假日效应和多周期性的数据。本项目训练了两个版本：

- **Prophet_tuned** — 含国家节假日特征的调优版本，通过网格搜索选取最优超参数
- **Prophet_plain** — 无节假日特征的基础版本，作为对比基线，用于评估节假日特征的实际增益

超参数搜索空间（16 种组合）：

| 参数 | 搜索值 |
| ---- | ---- |
| changepoint_prior_scale | [0.01, 0.1] |
| seasonality_prior_scale | [5.0, 10.0] |
| yearly_seasonality | [True, False] |
| weekly_seasonality | [True, False] |
| seasonality_mode | [additive, multiplicative] |

所有模型均使用**固定窗口交叉验证**（2 个窗口，步长 365 天），保证评估标准一致。

### per-SKU 自动选优

核心策略：不在全局层面选一个万能模型，而是为每个 SKU 各自选择最优模型。

流程：

1. 对每个 SKU，用全部 6 种模型做固定窗口交叉验证
2. 使用 `utilsforecast` 统一计算各模型的 SMAPE
3. 为每个 SKU 选择 SMAPE 最低的模型作为最优模型
4. 在测试集上独立验证最优模型的泛化能力
5. 验证通过后，合并训练集与测试集，用最优模型重训并输出最终上线预测

---

## 关键结果

- Kaggle 竞赛排名前 20%
- 集成 6 种时序模型，覆盖统计类与贝叶斯类方法
- 在数百个 SKU 上完成 per-SKU 自动选优
- 输出 365 天日级别预测，含 95% 置信区间
- 7 模块解耦流水线，支持一键全流程运行
- Streamlit 看板支持多层级 Bottom-Up 汇总与交互探索

---

## 技术栈

| 类别 | 技术 |
| ---- | ---- |
| 编程语言 | Python 3.10+ |
| 核心库 | pandas, numpy, itertools |
| 时序模型 | StatsForecast（AutoETS / AutoARIMA / AutoTheta / AutoCES）、Prophet |
| 模型评估 | utilsforecast（SMAPE）、固定窗口交叉验证 |
| 可视化 | Plotly, Streamlit |
| 工程化 | 模块化架构、一键全流程、CSV 中间结果管理 |

---

## 许可证

本项目基于 MIT 许可证开源。详见 [LICENSE](LICENSE) 文件。


