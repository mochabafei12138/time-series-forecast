"""
data_loader.py — 数据加载与预处理
"""
import pandas as pd
from config import TRAIN_FILE, TEST_FILE


def load_data():
    """加载训练集和测试集"""
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    train_df['ds'] = pd.to_datetime(train_df['ds'])
    test_df['ds'] = pd.to_datetime(test_df['ds'])

    print(f"训练集: {train_df.shape}, 日期范围: {train_df['ds'].min()} ~ {train_df['ds'].max()}")
    print(f"测试集: {test_df.shape}, 日期范围: {test_df['ds'].min()} ~ {test_df['ds'].max()}")
    print(f"训练集 SKU 数量: {train_df['unique_id'].nunique()}")

    return train_df, test_df


def prepare_sf_data(train_df):
    """为 StatsForecast 准备数据（只需 ds, unique_id, y）"""
    sf_train = train_df[['ds', 'unique_id', 'y']].copy()
    sf_train['ds'] = pd.to_datetime(sf_train['ds'])
    return sf_train


def get_all_sku_ids(train_df):
    """获取所有 SKU ID 列表"""
    return train_df['unique_id'].unique()