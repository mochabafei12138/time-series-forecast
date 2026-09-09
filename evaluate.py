"""
evaluate.py — 测试集评估 + 合并重训 + 上线预测
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
from statsforecast.core import StatsForecast
from statsforecast.models import AutoETS, AutoARIMA, AutoTheta, AutoCES
from prophet import Prophet
from utilsforecast.evaluation import evaluate       # ⚡ 统一标准：新增 import
from utilsforecast.losses import smape as uf_smape   # ⚡ 统一标准：新增 import
from config import H, LEVEL, SF_SEASON_LENGTH, TEST_EVAL_FILE, RETRAIN_FORECAST_FILE


# ============================================================
# 第 1 部分：在测试集上评估
# ============================================================

def evaluate_on_test_set(train_df, test_df, merge_eval,
                         best_params_per_sku_holiday,
                         best_params_per_sku_plain):
    """
    在测试集上评估 per-SKU 最优模型的预测表现。
    """
    print("\n" + "=" * 60)
    print("在测试集上评估 per-SKU 最优模型")
    print("=" * 60)

    all_sku_ids = merge_eval['unique_id'].unique()
    test_results = []

    for uid in tqdm(all_sku_ids):
        best_model_name = merge_eval.loc[
            merge_eval['unique_id'] == uid, 'best_model'
        ].values[0]

        # 测试集真实值
        test_true = test_df[test_df['unique_id'] == uid][['ds', 'y']].sort_values('ds')
        test_true = test_true.rename(columns={'y': 'y_true'})

        # 根据最优模型选择预测方法
        if best_model_name in ['AutoETS', 'AutoARIMA', 'AutoTheta', 'CES']:
            pred = _predict_statsforecast(train_df, uid, best_model_name, h=H)
        elif best_model_name == 'Prophet_tuned':
            pred = _predict_prophet(train_df, uid, best_params_per_sku_holiday[uid],
                                    use_holidays=True, h=H)
        elif best_model_name == 'Prophet_plain':
            pred = _predict_prophet(train_df, uid, best_params_per_sku_plain[uid],
                                    use_holidays=False, h=H)
        else:
            continue

        if pred is None or pred.empty:
            continue

        # 与测试集对齐
        merged = test_true.merge(pred, on='ds', how='inner')
        if merged.empty:
            continue

        # ⚡ 统一标准：使用 utilsforecast.evaluate 计算 SMAPE（与 CV 阶段一致）
        eval_df = pd.DataFrame({
            'unique_id': uid,
            'ds': merged['ds'],
            'y': merged['y_true'].values,
            'best_model_pred': merged['best_forecast'].values,
        })
        try:
            evals = evaluate(eval_df, metrics=[uf_smape], models=['best_model_pred'])
            smape_val = float(evals['best_model_pred'].iloc[0])
        except Exception:
            smape_val = np.nan

        test_results.append({
            'unique_id': uid,
            'best_model': best_model_name,
            'test_smape': smape_val,
            'test_n': len(merged),
        })

    test_eval_df = pd.DataFrame(test_results)

    print(f"\n测试集评估完成，共 {len(test_eval_df)} 个 SKU")
    if not test_eval_df.empty:
        print(f"测试集平均 SMAPE = {test_eval_df['test_smape'].mean():.4f}%")
        print(f"\n测试集 SMAPE 分布:")
        print(test_eval_df['test_smape'].describe())

        print("\n各模型在测试集上的平均 SMAPE:")
        print(test_eval_df.groupby('best_model')['test_smape'].agg(['mean', 'count', 'std']))

    test_eval_df.to_csv(TEST_EVAL_FILE, index=False)
    print(f"\n已保存测试集评估结果到: {TEST_EVAL_FILE}")

    return test_eval_df


# ---------- 辅助预测函数 ----------

def _predict_statsforecast(train_df, uid, model_name, h=H):
    """用 StatsForecast 模型对单个 SKU 做预测"""
    sku_data = train_df[train_df['unique_id'] == uid][['ds', 'unique_id', 'y']].sort_values('ds')
    sku_data['ds'] = pd.to_datetime(sku_data['ds'])

    if len(sku_data) < h:
        return None

    model_map = {
        'AutoETS': AutoETS(season_length=SF_SEASON_LENGTH),
        'AutoARIMA': AutoARIMA(season_length=SF_SEASON_LENGTH),
        'AutoTheta': AutoTheta(season_length=SF_SEASON_LENGTH),
        'CES': AutoCES(season_length=SF_SEASON_LENGTH),
    }

    sf = StatsForecast(
        models=[model_map[model_name]],
        freq='D',
        n_jobs=1,
        verbose=False,
    )

    try:
        forecast = sf.forecast(df=sku_data, h=h, level=[LEVEL])
        forecast = forecast.reset_index()
        forecast['ds'] = pd.to_datetime(forecast['ds'])
        forecast = forecast.rename(columns={model_name: 'best_forecast'})
        return forecast[['ds', 'best_forecast']]
    except Exception:
        return None


def _predict_prophet(train_df, uid, best_params, use_holidays=True, h=H):
    """用 Prophet 对单个 SKU 做预测"""
    if best_params is None:
        return None

    sku_data = train_df[train_df['unique_id'] == uid].sort_values('ds')
    if len(sku_data) < h:
        return None

    country = sku_data['country'].iloc[0] if use_holidays else None

    model = Prophet(
        weekly_seasonality=best_params['weekly_seasonality'],
        yearly_seasonality=best_params['yearly_seasonality'],
        daily_seasonality=best_params['daily_seasonality'],
        changepoint_prior_scale=best_params['changepoint_prior_scale'],
        seasonality_prior_scale=best_params['seasonality_prior_scale'],
        seasonality_mode=best_params['seasonality_mode'],
        interval_width=0.95,
    )
    if use_holidays and country is not None:
        try:
            model.add_country_holidays(country_name=country)
        except Exception:
            pass

    try:
        model.fit(sku_data[['ds', 'y']])
    except Exception:
        return None

    future = model.make_future_dataframe(periods=h, freq='D')
    forecast = model.predict(future)
    last_date = sku_data['ds'].max()
    forecast_future = forecast[forecast['ds'] > last_date].head(h)

    return pd.DataFrame({
        'ds': forecast_future['ds'].values,
        'best_forecast': forecast_future['yhat'].values,
    })


# ============================================================
# 第 2 部分：合并训练集 + 测试集，用最优模型重训
# ============================================================

def retrain_with_full_data(train_df, test_df, merge_eval,
                           best_params_per_sku_holiday,
                           best_params_per_sku_plain):
    """
    合并训练集 + 测试集，用 per-SKU 最优模型重训，输出未来 H 天预测。
    """
    print("\n" + "=" * 60)
    print("合并训练集 + 测试集，用最优模型重训")
    print("=" * 60)

    full_df = pd.concat([train_df, test_df], ignore_index=True)
    full_df = full_df.sort_values(['unique_id', 'ds']).reset_index(drop=True)
    print(f"合并后全量数据: {full_df.shape}")
    print(f"日期范围: {full_df['ds'].min()} ~ {full_df['ds'].max()}")
    print(f"SKU 数量: {full_df['unique_id'].nunique()}")

    all_sku_ids = merge_eval['unique_id'].unique()
    retrain_results = []
    skipped = 0

    for uid in tqdm(all_sku_ids):
        best_model_name = merge_eval.loc[
            merge_eval['unique_id'] == uid, 'best_model'
        ].values[0]

        if best_model_name in ['AutoETS', 'AutoARIMA', 'AutoTheta', 'CES']:
            pred = _retrain_statsforecast(full_df, uid, best_model_name, h=H)
        elif best_model_name == 'Prophet_tuned':
            pred = _retrain_prophet(full_df, uid, best_params_per_sku_holiday[uid],
                                    use_holidays=True, h=H)
        elif best_model_name == 'Prophet_plain':
            pred = _retrain_prophet(full_df, uid, best_params_per_sku_plain[uid],
                                    use_holidays=False, h=H)
        else:
            skipped += 1
            continue

        if pred is not None and not pred.empty:
            pred['unique_id'] = uid
            pred['best_model'] = best_model_name
            retrain_results.append(pred)
        else:
            skipped += 1

    if retrain_results:
        final_forecast = pd.concat(retrain_results, ignore_index=True)

        final_forecast.to_csv(RETRAIN_FORECAST_FILE, index=False)
        print(f"\n最终上线预测完成！")
        print(f"  预测形状: {final_forecast.shape}")
        print(f"  SKU 数量: {final_forecast['unique_id'].nunique()}")
        print(f"  预测日期范围: {final_forecast['ds'].min()} ~ {final_forecast['ds'].max()}")
        if skipped > 0:
            print(f"  跳过 {skipped} 个预测失败的 SKU")
        print(f"已保存到: {RETRAIN_FORECAST_FILE}")
        print(final_forecast.head())

        return final_forecast
    else:
        print("警告: 未生成任何预测结果")
        return None


def _retrain_statsforecast(full_df, uid, model_name, h=H):
    """用全量数据重训 StatsForecast"""
    sku_data = full_df[full_df['unique_id'] == uid][['ds', 'unique_id', 'y']].sort_values('ds')
    sku_data['ds'] = pd.to_datetime(sku_data['ds'])

    if len(sku_data) < h:
        return None

    model_map = {
        'AutoETS': AutoETS(season_length=SF_SEASON_LENGTH),
        'AutoARIMA': AutoARIMA(season_length=SF_SEASON_LENGTH),
        'AutoTheta': AutoTheta(season_length=SF_SEASON_LENGTH),
        'CES': AutoCES(season_length=SF_SEASON_LENGTH),
    }

    sf = StatsForecast(
        models=[model_map[model_name]],
        freq='D',
        n_jobs=1,
        verbose=False,
    )

    try:
        forecast = sf.forecast(df=sku_data, h=h, level=[LEVEL])
        forecast = forecast.reset_index()
        forecast['ds'] = pd.to_datetime(forecast['ds'])
        forecast = forecast.rename(columns={model_name: 'best_forecast'})

        for suffix in ('-lo-95', '-hi-95'):
            col = f'best_forecast{suffix}'
            src = f'{model_name}{suffix}'
            forecast[col] = forecast.get(src, np.nan)

        return forecast[['ds', 'best_forecast', 'best_forecast-lo-95', 'best_forecast-hi-95']]
    except Exception:
        return None


def _retrain_prophet(full_df, uid, best_params, use_holidays=True, h=H):
    """用全量数据重训 Prophet"""
    if best_params is None:
        return None

    sku_data = full_df[full_df['unique_id'] == uid].sort_values('ds')
    if len(sku_data) < h:
        return None

    country = sku_data['country'].iloc[0] if use_holidays else None

    model = Prophet(
        weekly_seasonality=best_params['weekly_seasonality'],
        yearly_seasonality=best_params['yearly_seasonality'],
        daily_seasonality=best_params['daily_seasonality'],
        changepoint_prior_scale=best_params['changepoint_prior_scale'],
        seasonality_prior_scale=best_params['seasonality_prior_scale'],
        seasonality_mode=best_params['seasonality_mode'],
        interval_width=0.95,
    )
    if use_holidays and country is not None:
        try:
            model.add_country_holidays(country_name=country)
        except Exception:
            pass

    try:
        model.fit(sku_data[['ds', 'y']])
    except Exception:
        return None

    future = model.make_future_dataframe(periods=h, freq='D')
    forecast = model.predict(future)
    last_date = sku_data['ds'].max()
    forecast_future = forecast[forecast['ds'] > last_date].head(h)

    return pd.DataFrame({
        'ds': forecast_future['ds'].values,
        'best_forecast': forecast_future['yhat'].values,
        'best_forecast-lo-95': forecast_future['yhat_lower'].values,
        'best_forecast-hi-95': forecast_future['yhat_upper'].values,
    })