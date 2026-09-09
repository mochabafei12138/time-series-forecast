"""
statsforecast_model.py — StatsForecast 统计模型
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
from statsforecast.models import AutoETS, AutoARIMA, AutoTheta, AutoCES
from statsforecast.core import StatsForecast
from config import H, N_WINDOWS, STEP_SIZE, LEVEL, SF_SEASON_LENGTH


def build_sf_models():
    """构建 StatsForecast 模型列表"""
    return [
        AutoETS(season_length=SF_SEASON_LENGTH),
        AutoARIMA(season_length=SF_SEASON_LENGTH),
        AutoTheta(season_length=SF_SEASON_LENGTH),
        AutoCES(season_length=SF_SEASON_LENGTH),
    ]


MODEL_NAMES = ['AutoETS', 'AutoARIMA', 'AutoTheta', 'AutoCES']


def run_sf_cv(sf_train, all_sku_ids):
    """
    对每个 SKU 用 StatsForecast 做固定窗口 CV。
    返回:
        sf_cv_results: {uid: cv_df} 或 None
        sf_cutoffs_map: {uid: [cutoff列表]} 或 None
    """
    sf_models = build_sf_models()
    sf_cv_results = {}
    sf_cutoffs_map = {}

    print("\n正在用 StatsForecast 进行交叉验证...")
    for uid in tqdm(all_sku_ids):
        sku_data = sf_train[sf_train['unique_id'] == uid].sort_values('ds').reset_index(drop=True)

        sf = StatsForecast(
            models=sf_models,
            freq='D',
            n_jobs=-1,
            verbose=False,
        )

        try:
            cv_df = sf.cross_validation(
                df=sku_data[['ds', 'unique_id', 'y']],
                h=H,
                step_size=STEP_SIZE,
                n_windows=N_WINDOWS,
            )
            cv_df = cv_df.reset_index()
            sf_cv_results[uid] = cv_df
            sf_cutoffs_map[uid] = sorted(cv_df['cutoff'].unique())
        except Exception:
            try:
                cv_df = sf.cross_validation(
                    df=sku_data[['ds', 'unique_id', 'y']],
                    h=H,
                    step_size=STEP_SIZE,
                    n_windows=max(N_WINDOWS - 1, 1),
                )
                cv_df = cv_df.reset_index()
                sf_cv_results[uid] = cv_df
                sf_cutoffs_map[uid] = sorted(cv_df['cutoff'].unique())
            except Exception:
                sf_cv_results[uid] = None
                sf_cutoffs_map[uid] = None

    print("StatsForecast CV 完成")
    return sf_cv_results, sf_cutoffs_map


def generate_sf_forecast(sf_train):
    """用 StatsForecast 生成全量预测"""
    sf_models = build_sf_models()
    print("\n正在用 StatsForecast 生成全量预测...")
    sf_all = StatsForecast(
        models=sf_models,
        freq='D',
        n_jobs=-1,
        verbose=False,
    )
    sf_forecasts_df = sf_all.forecast(
        df=sf_train[['ds', 'unique_id', 'y']],
        h=H,
        level=[LEVEL],
    )
    sf_forecasts_df = sf_forecasts_df.reset_index()
    sf_forecasts_df['ds'] = pd.to_datetime(sf_forecasts_df['ds'])
    print(f"StatsForecast 预测完成，共 {len(sf_forecasts_df)} 行")
    return sf_forecasts_df