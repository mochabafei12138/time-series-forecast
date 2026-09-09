"""
config.py — 全局配置参数
"""
import os
from pathlib import Path

# ========== 路径配置 ==========
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== 数据文件路径 ==========
TRAIN_FILE = DATA_DIR / 'train.csv'
TEST_FILE = DATA_DIR / 'test.csv'

# ========== 预测参数（固定窗口） ==========
H = 365                       # 预测步长（天）
N_WINDOWS = 2                 # 固定 CV 窗口数
STEP_SIZE = 365               # 固定 CV 步长
LEVEL = 95                    # 置信区间水平

# ========== StatsForecast 模型参数 ==========
SF_SEASON_LENGTH = 7

# ========== Prophet 超参数搜索空间 ==========
PROPHET_PARAM_GRID = {
    'changepoint_prior_scale': [0.01, 0.1],
    'seasonality_prior_scale': [5.0, 10.0],
    'yearly_seasonality': [True, False],
    'weekly_seasonality': [True, False],
    'daily_seasonality': [False],
    'seasonality_mode': ['additive', 'multiplicative'],
}

# ========== 输出文件路径 ==========
MODEL_COMPARISON_FILE = OUTPUT_DIR / 'final_model_comparison.csv'
BEST_FORECAST_FILE = OUTPUT_DIR / 'final_per_sku_best_forecast.csv'
TEST_EVAL_FILE = OUTPUT_DIR / 'test_set_evaluation.csv'
RETRAIN_FORECAST_FILE = OUTPUT_DIR / 'retrain_final_forecast.csv'
SKU_SMAPE_DETAIL_FILE = OUTPUT_DIR / 'sku_smape_detail.csv'
BEST_MODEL_DIST_FILE = OUTPUT_DIR / 'best_model_distribution.csv'