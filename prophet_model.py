"""
prophet_model.py — Prophet 模型（含超参数搜索，跟随 SF cutoff）
"""
import itertools
import pandas as pd
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from prophet import Prophet
from prophet.diagnostics import cross_validation
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import smape as uf_smape
from config import H, PROPHET_PARAM_GRID


def build_param_combinations():
    """生成 Prophet 超参数搜索空间的所有组合"""
    param_combinations = [
        dict(zip(PROPHET_PARAM_GRID.keys(), values))
        for values in itertools.product(*PROPHET_PARAM_GRID.values())
    ]
    print(f"Prophet 超参数搜索空间: {len(param_combinations)} 种组合")
    return param_combinations


def safe_add_country_holidays(model, country):
    try:
        model.add_country_holidays(country_name=country)
    except Exception:
        pass


def build_prophet_model(params, use_holidays=True, country=None):
    """构建 Prophet 模型"""
    model = Prophet(
        weekly_seasonality=params['weekly_seasonality'],
        yearly_seasonality=params['yearly_seasonality'],
        daily_seasonality=params['daily_seasonality'],
        changepoint_prior_scale=params['changepoint_prior_scale'],
        seasonality_prior_scale=params['seasonality_prior_scale'],
        seasonality_mode=params['seasonality_mode'],
        interval_width=0.95,
    )
    if use_holidays and country is not None:
        safe_add_country_holidays(model, country)
    return model


def prophet_cv_with_cutoffs(train_df, uid, params, cutoffs, use_holidays=True):
    """用 StatsForecast 传入的 cutoffs 对 Prophet 做 CV"""
    sku_data = train_df[train_df['unique_id'] == uid].sort_values('ds').reset_index(drop=True)
    country = sku_data['country'].iloc[0] if use_holidays else None

    model = build_prophet_model(params, use_holidays=use_holidays, country=country)
    try:
        model.fit(sku_data[['ds', 'y']])
    except Exception:
        return None

    try:
        cv_df = cross_validation(
            model,
            cutoffs=cutoffs,
            horizon=f'{H} days',
            disable_tqdm=True,
        )
    except Exception:
        valid_cutoffs = [c for c in cutoffs if c < sku_data['ds'].max()]
        if len(valid_cutoffs) < len(cutoffs):
            try:
                cv_df = cross_validation(
                    model,
                    cutoffs=valid_cutoffs,
                    horizon=f'{H} days',
                    disable_tqdm=True,
                )
            except Exception:
                return None
        else:
            return None

    return cv_df


def prophet_cv_metric(cv_df):
    """基于 Prophet 的 cv_df 计算 SMAPE"""
    if cv_df is None or cv_df.empty:
        return np.nan
    df_pred = cv_df[['cutoff', 'ds', 'y', 'yhat']].copy()
    df_pred['unique_id'] = 'sku'
    df_pred = df_pred.rename(columns={'yhat': 'model_col'})
    try:
        evals = evaluate(df_pred, metrics=[uf_smape], models=['model_col'])
        val = evals['model_col'].iloc[0]
        return float(val) if not np.isnan(val) else np.nan
    except Exception:
        return np.nan


def search_best_prophet_params(train_df, uid, param_combinations, cutoffs, use_holidays=True):
    """对单个 SKU 搜索最优 Prophet 超参数"""
    best_metric = np.inf
    best_params = None

    for params in param_combinations:
        try:
            cv_df = prophet_cv_with_cutoffs(
                train_df, uid, params, cutoffs, use_holidays=use_holidays,
            )
        except Exception:
            continue

        if cv_df is None or cv_df.empty:
            continue

        metric_val = prophet_cv_metric(cv_df)
        if np.isnan(metric_val):
            continue
        if metric_val < best_metric:
            best_metric = metric_val
            best_params = deepcopy(params)

    return best_params, best_metric


def search_all_sku_prophet(train_df, all_sku_ids, sf_cutoffs_map, use_holidays=True):
    """对所有 SKU 搜索最优 Prophet 超参数"""
    param_combinations = build_param_combinations()
    desc = "有节假日" if use_holidays else "无节假日"
    print(f"\n正在搜索最优 Prophet 超参数（{desc}）...")

    best_params_per_sku = {}
    smape_per_sku = {}

    for uid in tqdm(all_sku_ids):
        cutoffs = sf_cutoffs_map.get(uid)
        if cutoffs is None:
            best_params_per_sku[uid] = None
            smape_per_sku[uid] = np.nan
            continue

        best_params, best_metric = search_best_prophet_params(
            train_df, uid, param_combinations, cutoffs, use_holidays=use_holidays,
        )
        best_params_per_sku[uid] = best_params
        smape_per_sku[uid] = best_metric

        if best_params is not None:
            print(f"  [{desc}] SKU {uid}: 最优 SMAPE={best_metric:.2f}%")
        else:
            print(f"  [{desc}] SKU {uid}: 未找到有效超参数")

    return best_params_per_sku, smape_per_sku


def prophet_forecast_with_best_params(train_df, uid, best_params, use_holidays=True, h=H):
    """用最优参数生成 Prophet 预测"""
    if best_params is None:
        return None

    sku_data = train_df[train_df['unique_id'] == uid].sort_values('ds')
    country = sku_data['country'].iloc[0] if use_holidays else None

    model = build_prophet_model(best_params, use_holidays=use_holidays, country=country)
    model.fit(sku_data[['ds', 'y']])

    future = model.make_future_dataframe(periods=h, freq='D')
    forecast = model.predict(future)

    last_date = sku_data['ds'].max()
    forecast_future = forecast[forecast['ds'] > last_date].head(h)

    return pd.DataFrame({
        'unique_id': uid,
        'ds': forecast_future['ds'].values,
        'yhat': forecast_future['yhat'].values,
        'yhat_lower': forecast_future['yhat_lower'].values,
        'yhat_upper': forecast_future['yhat_upper'].values,
    })


def generate_all_prophet_forecasts(train_df, all_sku_ids, best_params_per_sku, use_holidays=True):
    """对所有 SKU 用最优参数生成 Prophet 预测"""
    model_tag = 'Prophet_tuned' if use_holidays else 'Prophet_plain'
    desc = "有节假日" if use_holidays else "无节假日"
    print(f"\n正在生成 Prophet 预测（{desc}）...")

    all_results = []
    for uid in tqdm(all_sku_ids):
        best_params = best_params_per_sku[uid]
        pred = prophet_forecast_with_best_params(
            train_df, uid, best_params, use_holidays=use_holidays, h=H,
        )
        if pred is not None:
            pred.rename(columns={
                'yhat': model_tag,
                'yhat_lower': f'{model_tag}-lo-95',
                'yhat_upper': f'{model_tag}-hi-95',
            }, inplace=True)
            all_results.append(pred)

    prophet_forecasts_df = pd.concat(all_results, ignore_index=True)
    print(f"Prophet 预测（{desc}）完成，共 {len(prophet_forecasts_df)} 行")
    return prophet_forecasts_df


def generate_all_prophet_cv(train_df, all_sku_ids, best_params_per_sku,
                            sf_cutoffs_map, use_holidays=True):
    """对所有 SKU 用调参后的 Prophet 重跑 CV"""
    model_tag = 'Prophet_tuned' if use_holidays else 'Prophet_plain'
    desc = "有节假日" if use_holidays else "无节假日"
    print(f"\n正在用调参后的 Prophet 重跑 CV（{desc}）...")

    all_cv = []
    for uid in tqdm(all_sku_ids):
        best_params = best_params_per_sku[uid]
        cutoffs = sf_cutoffs_map.get(uid)
        if best_params is None or cutoffs is None:
            continue

        cv_df = prophet_cv_with_cutoffs(
            train_df, uid, best_params, cutoffs, use_holidays=use_holidays,
        )
        if cv_df is not None and len(cv_df) > 0:
            cv_df = cv_df.copy()
            cv_df['unique_id'] = uid
            cv_df = cv_df.rename(columns={'yhat': model_tag})
            all_cv.append(cv_df[['unique_id', 'ds', 'cutoff', 'y', model_tag]])

    prophet_cv_df = pd.concat(all_cv, ignore_index=True)
    print(f"Prophet CV（{desc}）完成，共 {len(prophet_cv_df)} 行")
    return prophet_cv_df