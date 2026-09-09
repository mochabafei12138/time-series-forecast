"""
pipeline.py — 主流程入口，串联所有模块
"""
import time
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from config import OUTPUT_DIR
from data_loader import load_data, prepare_sf_data, get_all_sku_ids
from statsforecast_model import run_sf_cv, generate_sf_forecast, MODEL_NAMES
from prophet_model import (
    search_all_sku_prophet, generate_all_prophet_forecasts,
    generate_all_prophet_cv,
)
from ensemble import (
    merge_and_select_best_model, merge_all_forecasts, get_best_forecast,
)
from evaluate import evaluate_on_test_set, retrain_with_full_data
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import smape as uf_smape


def run_pipeline():
    """运行完整预测流水线"""
    overall_start = time.time()
    print("=" * 60)
    print("时间序列预测流水线 — StatsForecast + Prophet")
    print("=" * 60)

    # ============================================================
    # Step 1: 加载数据
    # ============================================================
    print("\n" + "=" * 60)
    print("Step 1: 加载数据")
    print("=" * 60)
    train_df, test_df = load_data()
    sf_train = prepare_sf_data(train_df)
    all_sku_ids = get_all_sku_ids(train_df)

    # ============================================================
    # Step 2: StatsForecast 统计模型
    # ============================================================
    print("\n" + "=" * 60)
    print("Step 2: StatsForecast 统计模型（固定窗口 CV）")
    print("=" * 60)
    t0 = time.time()

    sf_cv_results, sf_cutoffs_map = run_sf_cv(sf_train, all_sku_ids)
    sf_forecasts_df = generate_sf_forecast(sf_train)

    print(f"StatsForecast 耗时: {time.time() - t0:.1f}s")

    # ============================================================
    # Step 3: Prophet 模型
    # ============================================================
    print("\n" + "=" * 60)
    print("Step 3: Prophet 模型（含超参数搜索）")
    print("=" * 60)
    t0 = time.time()

    # 3a. 有节假日
    best_params_holiday, smape_holiday = search_all_sku_prophet(
        train_df, all_sku_ids, sf_cutoffs_map, use_holidays=True,
    )
    prophet_tuned_df = generate_all_prophet_forecasts(
        train_df, all_sku_ids, best_params_holiday, use_holidays=True,
    )
    prophet_tuned_cv = generate_all_prophet_cv(
        train_df, all_sku_ids, best_params_holiday,
        sf_cutoffs_map, use_holidays=True,
    )

    # 3b. 无节假日
    best_params_plain, smape_plain = search_all_sku_prophet(
        train_df, all_sku_ids, sf_cutoffs_map, use_holidays=False,
    )
    prophet_plain_df = generate_all_prophet_forecasts(
        train_df, all_sku_ids, best_params_plain, use_holidays=False,
    )
    prophet_plain_cv = generate_all_prophet_cv(
        train_df, all_sku_ids, best_params_plain,
        sf_cutoffs_map, use_holidays=False,
    )

    print(f"Prophet 耗时: {time.time() - t0:.1f}s")

    # ============================================================
    # Step 4: 合并 CV 结果，统一评估
    # ============================================================
    print("\n" + "=" * 60)
    print("Step 4: 合并 CV 结果，统一评估")
    print("=" * 60)

    # 合并 StatsForecast CV
    sf_cv_list = [df for df in sf_cv_results.values()
                  if df is not None and len(df) > 0]
    sf_cv_merged = pd.concat(sf_cv_list, ignore_index=True) if sf_cv_list else pd.DataFrame()

    # 合并 Prophet CV
    if not prophet_tuned_cv.empty and not prophet_plain_cv.empty:
        prophet_cv_merged = prophet_tuned_cv.merge(
            prophet_plain_cv[['unique_id', 'ds', 'cutoff', 'Prophet_plain']],
            on=['unique_id', 'ds', 'cutoff'], how='outer',
        )
    elif not prophet_tuned_cv.empty:
        prophet_cv_merged = prophet_tuned_cv.copy()
        prophet_cv_merged['Prophet_plain'] = np.nan
    else:
        prophet_cv_merged = pd.DataFrame()

    # 合并所有 CV
    if not sf_cv_merged.empty and not prophet_cv_merged.empty:
        cv_df_with_prophet = sf_cv_merged.merge(
            prophet_cv_merged[['unique_id', 'ds', 'cutoff', 'Prophet_tuned', 'Prophet_plain']],
            on=['unique_id', 'ds', 'cutoff'], how='left',
        )
    elif not sf_cv_merged.empty:
        cv_df_with_prophet = sf_cv_merged.copy()
    else:
        print("错误: 没有可用的 CV 结果")
        return

    # 统一评估
    def evaluate_cv(df, metric):
        models = [c for c in df.columns if c not in ['unique_id', 'ds', 'y', 'cutoff']]
        if len(models) < 2:
            print(f"警告: 可用模型列不足: {models}")
            return pd.DataFrame()
        evals = evaluate(df, metrics=[metric], models=models)
        evals = evals.drop(columns=['metric'])
        evals['best_model'] = evals[models].idxmin(axis=1)
        return evals

    evaluation_df = evaluate_cv(cv_df_with_prophet, uf_smape)

    if evaluation_df.empty:
        print("评估失败，无法继续")
        return

    print("\n各 SKU 最优模型分布（CV 阶段）:")
    print(evaluation_df['best_model'].value_counts())

    merge_eval = merge_and_select_best_model(evaluation_df)

    # ============================================================
    # Step 5: 合并所有预测，按最优模型输出
    # ============================================================
    print("\n" + "=" * 60)
    print("Step 5: 合并所有模型预测，按最优模型输出")
    print("=" * 60)

    combined_forecasts = merge_all_forecasts(
        sf_forecasts_df, prophet_tuned_df, prophet_plain_df,
    )
    best_forecast = get_best_forecast(combined_forecasts, merge_eval)

    # ============================================================
    # Step 6: 在测试集上评估
    # ============================================================
    print("\n" + "=" * 60)
    print("Step 6: 在测试集上评估最优模型")
    print("=" * 60)

    test_eval_df = evaluate_on_test_set(
        train_df, test_df, merge_eval,
        best_params_holiday, best_params_plain,
    )

    # ============================================================
    # Step 7: 合并训练+测试集重训，输出最终上线预测
    # ============================================================
    print("\n" + "=" * 60)
    print("Step 7: 合并全量数据重训，输出最终上线预测")
    print("=" * 60)

    final_forecast = retrain_with_full_data(
        train_df, test_df, merge_eval,
        best_params_holiday, best_params_plain,
    )

    # ============================================================
    # 汇总输出
    # ============================================================
    total_time = time.time() - overall_start
    print("\n" + "=" * 60)
    print("流水线执行完成")
    print("=" * 60)
    print(f"总耗时: {total_time:.1f}s")
    print(f"\n输出文件列表（{OUTPUT_DIR}）:")
    import os
    for f in sorted(os.listdir(OUTPUT_DIR)):
        fpath = OUTPUT_DIR / f
        size_kb = fpath.stat().st_size / 1024
        print(f"  {f:<45s} ({size_kb:.1f} KB)")

    print("\n" + "=" * 60)
    print("关键结果摘要")
    print("=" * 60)

    if not merge_eval.empty:
        dist = merge_eval['best_model'].value_counts()
        print("\n最优模型选择分布:")
        for model_name, count in dist.items():
            pct = count / len(merge_eval) * 100
            print(f"  {model_name:<25s}: {count:>4d} 个 SKU ({pct:.1f}%)")

    if 'best_smape' in merge_eval.columns:
        print(f"\nCV 平均 SMAPE（per-SKU 最优）: {merge_eval['best_smape'].mean():.4f}%")

    if test_eval_df is not None and not test_eval_df.empty:
        print(f"测试集平均 SMAPE: {test_eval_df['test_smape'].mean():.4f}%")

    if final_forecast is not None:
        print(f"\n最终上线预测: {len(final_forecast)} 行")
        print(f"  预测日期范围: {final_forecast['ds'].min()} ~ {final_forecast['ds'].max()}")

    return {
        'train_df': train_df,
        'test_df': test_df,
        'evaluation_df': evaluation_df,
        'merge_eval': merge_eval,
        'best_forecast': best_forecast,
        'test_eval_df': test_eval_df,
        'final_forecast': final_forecast,
    }


if __name__ == '__main__':
    results = run_pipeline()