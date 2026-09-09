"""
ensemble.py — 合并模型 SMAPE、选择 per-SKU 最优模型、输出预测
"""
import pandas as pd
import numpy as np
from config import (
    MODEL_COMPARISON_FILE, BEST_FORECAST_FILE,
    SKU_SMAPE_DETAIL_FILE, BEST_MODEL_DIST_FILE,
)

MODEL_COLS = [
    'AutoETS', 'AutoARIMA', 'AutoTheta', 'CES',
    'Prophet_tuned', 'Prophet_plain',
]


def merge_and_select_best_model(evaluation_df):
    """
    合并所有模型 SMAPE，选出 per-SKU 最优模型。
    """
    merge_eval = evaluation_df.copy()
    available_cols = [c for c in MODEL_COLS if c in merge_eval.columns]

    if len(available_cols) < 2:
        print(f"警告: 可用模型列不足: {available_cols}")
        return merge_eval

    # 选最优模型
    merge_eval['best_model'] = merge_eval[available_cols].idxmin(axis=1)
    merge_eval['best_smape'] = merge_eval.apply(
        lambda r: r[r['best_model']], axis=1
    )

    # 打印模型级对比
    print("\n" + "=" * 60)
    print("模型级对比：各模型全部 SKU 平均 SMAPE")
    print("=" * 60)
    summary = merge_eval[available_cols].mean().sort_values()
    for m, v in summary.items():
        print(f"  {m:<25s} {v:.4f}%")

    # 最优模型分布
    print("\n各 SKU 最优模型分布:")
    dist = merge_eval['best_model'].value_counts()
    print(dist)
    dist.to_csv(BEST_MODEL_DIST_FILE)
    print(f"已保存最优模型分布到: {BEST_MODEL_DIST_FILE}")

    # 整体平均 SMAPE
    overall_mean = merge_eval['best_smape'].mean()
    print(f"\n整体平均 SMAPE（per-SKU 各自最优）= {overall_mean:.4f}%")

    # 各 SKU 明细
    sku_detail = merge_eval[['unique_id', 'best_model', 'best_smape']].copy()
    sku_detail = sku_detail.sort_values('best_smape', ascending=False)

    print(f"\n各 SKU 最优模型及 SMAPE 明细（前 10 行 / 共 {len(sku_detail)} 行）:")
    print(f"{'SKU':<20s} {'最优模型':<25s} {'SMAPE(%)':<10s}")
    print("-" * 55)
    for _, row in sku_detail.head(10).iterrows():
        print(f"{str(row['unique_id']):<20s} {row['best_model']:<25s} {row['best_smape']:.4f}")
    if len(sku_detail) > 10:
        print(f"  ... 共 {len(sku_detail)} 个 SKU，详见保存文件")

    sku_detail.to_csv(SKU_SMAPE_DETAIL_FILE, index=False)
    print(f"已保存各 SKU 明细到: {SKU_SMAPE_DETAIL_FILE}")

    # 保存完整对比表
    compare_df = merge_eval[['unique_id', 'best_model', 'best_smape'] + available_cols]
    compare_df.to_csv(MODEL_COMPARISON_FILE, index=False)
    print(f"已保存完整对比表到: {MODEL_COMPARISON_FILE}")

    return merge_eval


def get_best_forecast(combined_forecasts, merge_eval):
    """
    按 per-SKU 最优模型提取预测（含置信区间）。
    """
    with_best = combined_forecasts.merge(
        merge_eval[['unique_id', 'best_model']],
        on='unique_id',
    )

    result = with_best[['unique_id', 'ds']].copy()
    result['best_model_name'] = with_best['best_model']

    # 点预测
    result['best_forecast'] = with_best.apply(
        lambda row: row.get(row['best_model'], np.nan), axis=1,
    )

    # 动态探测上下界列名
    def _find_interval_col(df, model_name, kind):
        for col in df.columns:
            if col.startswith(f"{model_name}-{kind}-") or \
               col.startswith(f"{model_name}-{kind}-95"):
                return col
        return None

    for kind, suffix in [('lo', '-lo-95'), ('hi', '-hi-95')]:
        col_name = f'best_forecast{suffix}'
        result[col_name] = with_best.apply(
            lambda row, k=kind: (
                row.get(_find_interval_col(with_best, row['best_model'], k), np.nan)
            ),
            axis=1,
        )

    # 检查缺失
    missing = result['best_forecast'].isna().sum()
    total = len(result)
    print(f"\n最佳预测结果统计:")
    print(f"  总行数: {total}")
    print(f"  缺失点预测: {missing} ({missing/total*100:.1f}%)")
    print(f"  缺失下界: {result['best_forecast-lo-95'].isna().sum()}")
    print(f"  缺失上界: {result['best_forecast-hi-95'].isna().sum()}")

    result.to_csv(BEST_FORECAST_FILE, index=False)
    print(f"已保存 per-SKU 最优模型预测到: {BEST_FORECAST_FILE}")

    return result


def merge_all_forecasts(sf_forecasts_df, prophet_tuned_df, prophet_plain_df):
    """合并所有模型预测结果成宽表"""
    for df in [sf_forecasts_df, prophet_tuned_df, prophet_plain_df]:
        if not df.empty:
            df['ds'] = pd.to_datetime(df['ds'])

    combined = sf_forecasts_df.copy()
    if not prophet_tuned_df.empty:
        combined = combined.merge(
            prophet_tuned_df, on=['unique_id', 'ds'], how='left',
        )
    if not prophet_plain_df.empty:
        combined = combined.merge(
            prophet_plain_df, on=['unique_id', 'ds'], how='left',
        )

    print(f"\n合并后预测宽表形状: {combined.shape}")
    return combined