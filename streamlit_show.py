"""
streamlit_app.py — 时间序列预测结果交互看板
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ========== 页面配置 ==========
st.set_page_config(
    page_title="时间序列预测看板",
    page_icon="🔮",
    layout="wide",
)

st.title("🔮 时间序列预测结果看板")
st.markdown("---")

# ========== 数据路径配置 ==========
DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")

TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"
RETRAIN_FORECAST_FILE = OUTPUT_DIR / "retrain_final_forecast.csv"


# ========== 国家坐标字典 ==========
COUNTRY_COORDS = {
    "Canada": (56.13, -106.35),
    "United States": (37.09, -95.71),
    "Mexico": (23.63, -102.55),
    "Brazil": (-14.24, -51.93),
    "Argentina": (-38.42, -63.62),
    "Chile": (-35.68, -71.54),
    "Colombia": (4.57, -74.30),
    "Peru": (-9.19, -75.02),
    "United Kingdom": (55.38, -3.44),
    "France": (46.23, 2.21),
    "Germany": (51.17, 10.45),
    "Italy": (41.87, 12.57),
    "Spain": (40.42, -3.70),
    "Portugal": (39.52, -8.39),
    "Netherlands": (52.13, 5.29),
    "Belgium": (50.50, 4.47),
    "Switzerland": (46.62, 7.67),
    "Sweden": (60.47, 8.47),
    "Norway": (60.47, 8.47),
    "Denmark": (56.26, 9.50),
    "Finland": (64.32, 26.81),
    "Poland": (52.33, 19.40),
    "Czech Republic": (49.82, 15.47),
    "Greece": (39.07, 21.82),
    "Turkey": (38.96, 35.24),
    "Russia": (62.10, 96.68),
    "Ukraine": (48.78, 31.51),
    "Romania": (46.61, 25.15),
    "Hungary": (47.16, 19.50),
    "Japan": (36.20, 138.25),
    "South Korea": (37.00, 127.50),
    "China": (35.86, 104.20),
    "India": (20.59, 78.96),
    "Australia": (-25.27, 133.78),
    "South Africa": (-29.00, 24.00),
    "Nigeria": (9.08, 8.68),
    "Kenya": (-0.02, 37.91),
    "Egypt": (26.82, 30.80),
    "Morocco": (31.79, -7.09),
}


# ========== 数据加载函数 ==========
@st.cache_data
def load_all_data():
    """加载并合并训练集、测试集和预测结果"""
    train_df = pd.read_csv(TRAIN_FILE)
    train_df["ds"] = pd.to_datetime(train_df["ds"])

    test_df = pd.read_csv(TEST_FILE)
    test_df["ds"] = pd.to_datetime(test_df["ds"])

    forecast_df = pd.read_csv(RETRAIN_FORECAST_FILE)
    forecast_df["ds"] = pd.to_datetime(forecast_df["ds"])

    # 标记数据来源
    train_df["data_type"] = "训练集"
    test_df["data_type"] = "测试集"
    forecast_df["data_type"] = "预测"

    # 训练集和测试集添加预测相关列（空值）
    train_df["best_forecast"] = np.nan
    train_df["best_forecast-lo-95"] = np.nan
    train_df["best_forecast-hi-95"] = np.nan
    train_df["best_model"] = ""

    test_df["best_forecast"] = np.nan
    test_df["best_forecast-lo-95"] = np.nan
    test_df["best_forecast-hi-95"] = np.nan
    test_df["best_model"] = ""

    # 预测集添加 y 列（空值）
    forecast_df["y"] = np.nan
    forecast_df["country"] = ""
    forecast_df["store"] = ""
    forecast_df["product"] = ""

    # 从 train/test 中获取 country, store, product 映射
    meta_df = pd.concat([train_df, test_df], ignore_index=True)
    meta_map = meta_df[["unique_id", "country", "store", "product"]].drop_duplicates(
        subset="unique_id"
    )
    meta_dict = {
        row["unique_id"]: {
            "country": row["country"],
            "store": row["store"],
            "product": row["product"],
        }
        for _, row in meta_map.iterrows()
    }

    # 填充预测集的元信息
    for uid in meta_dict:
        mask = forecast_df["unique_id"] == uid
        forecast_df.loc[mask, "country"] = meta_dict[uid]["country"]
        forecast_df.loc[mask, "store"] = meta_dict[uid]["store"]
        forecast_df.loc[mask, "product"] = meta_dict[uid]["product"]

    # 统一列顺序并合并
    common_cols = [
        "ds", "unique_id", "country", "store", "product",
        "y", "best_forecast", "best_forecast-lo-95", "best_forecast-hi-95",
        "best_model", "data_type",
    ]
    train_aligned = train_df[common_cols].copy()
    test_aligned = test_df[common_cols].copy()
    forecast_aligned = forecast_df[common_cols].copy()

    full_df = pd.concat([train_aligned, test_aligned, forecast_aligned], ignore_index=True)
    full_df = full_df.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    return full_df, train_df, test_df, forecast_df, meta_dict


# ========== 加载数据 ==========
with st.spinner("正在加载数据..."):
    full_df, train_df, test_df, forecast_df, meta_dict = load_all_data()

# 获取国家列表
countries = sorted(full_df["country"].dropna().unique().tolist())
country_sku_counts = (
    full_df[full_df["data_type"] == "训练集"]
    .groupby("country")["unique_id"]
    .nunique()
    .to_dict()
)

# 为每个国家准备地图数据
country_sku_df = pd.DataFrame(
    [
        {
            "country": c,
            "sku_count": country_sku_counts.get(c, 0),
        }
        for c in countries
        if c in COUNTRY_COORDS
    ]
)

# 添加经纬度
country_sku_df["lat"] = country_sku_df["country"].apply(
    lambda c: COUNTRY_COORDS.get(c, (0, 0))[0]
)
country_sku_df["lon"] = country_sku_df["country"].apply(
    lambda c: COUNTRY_COORDS.get(c, (0, 0))[1]
)

# ========== 侧边栏 ==========
st.sidebar.header("🌍 选择国家")

selected_country = st.sidebar.selectbox(
    "选择一个国家:",
    [""] + countries,
    index=0,
    key="sidebar_country",
)

if selected_country:
    st.session_state.selected_country = selected_country
else:
    st.session_state.selected_country = None


# ===== 地图 =====
st.subheader("🌎 国家分布")

hover_texts = country_sku_df.apply(
    lambda r: f"<b>{r['country']}</b><br>SKU 数量: {r['sku_count']}",
    axis=1,
)

fig_map = go.Figure()

fig_map.add_trace(
    go.Scattergeo(
        lon=country_sku_df["lon"],
        lat=country_sku_df["lat"],
        hovertext=hover_texts,
        hoverinfo="text",
        mode="markers+text",
        marker=dict(
            size=country_sku_df["sku_count"].apply(lambda x: max(10, min(40, x * 2))),
            color=country_sku_df["sku_count"],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="SKU 数量"),
            line=dict(width=1, color="white"),
        ),
        text=country_sku_df["country"],
        textposition="top center",
        textfont=dict(size=10, color="black"),
    )
)

fig_map.update_layout(
    geo=dict(
        showland=True,
        landcolor="rgb(243, 243, 243)",
        countrycolor="rgb(204, 204, 204)",
        showocean=True,
        oceancolor="rgb(230, 240, 255)",
        projection_type="natural earth",
        coastlinecolor="rgb(204, 204, 204)",
    ),
    height=520,
    margin=dict(l=0, r=0, t=0, b=0),
)

st.plotly_chart(fig_map, use_container_width=True)

st.markdown("---")

# ============================================================
# 多层级 Bottom-Up 汇总（替换了原来的商店与产品详情）
# ============================================================
st.subheader("📊 多层级 Bottom-Up 汇总")

if selected_country:
    country_data = full_df[full_df["country"] == selected_country].copy()
else:
    country_data = full_df.copy()

# 层级组合选择器
level_options = {
    "国家总览": "country",
    "国家 × 商店": "country_store",
    "国家 × 产品": "country_product",
    "国家 × 商店 × 产品": "country_store_product",
}
selected_level = st.selectbox(
    "选择汇总层级组合:",
    list(level_options.keys()),
    index=0,
    key="level_selector",
)

# 根据选择的层级做汇总
subset = country_data.copy()
level_label = selected_level

if selected_level == "国家总览":
    pass

elif selected_level == "国家 × 商店":
    stores = sorted(country_data["store"].dropna().unique())
    sel = st.selectbox("选择商店:", stores, index=0, key="store_lv")
    subset = country_data[country_data["store"] == sel]
    level_label = f"{selected_country} × {sel}"

elif selected_level == "国家 × 产品":
    products = sorted(country_data["product"].dropna().unique())
    sel = st.selectbox("选择产品:", products, index=0, key="prod_lv")
    subset = country_data[country_data["product"] == sel]
    level_label = f"{selected_country} × {sel}"

else:  # 国家 × 商店 × 产品
    stores = sorted(country_data["store"].dropna().unique())
    sel_store = st.selectbox("选择商店:", stores, index=0, key="store_lv2")
    sub_s = country_data[country_data["store"] == sel_store]
    products = sorted(sub_s["product"].dropna().unique())
    sel_prod = st.selectbox("选择产品:", products, index=0, key="prod_lv2")
    subset = sub_s[sub_s["product"] == sel_prod]
    level_label = f"{selected_country} × {sel_store} × {sel_prod}"

# 按日期汇总
agg_train = (
    subset[subset["data_type"] == "训练集"]
    .groupby("ds")["y"].sum().reset_index()
)
agg_test = (
    subset[subset["data_type"] == "测试集"]
    .groupby("ds")["y"].sum().reset_index()
) if len(subset[subset["data_type"] == "测试集"]) > 0 else pd.DataFrame()
agg_forecast = (
    subset[subset["data_type"] == "预测"]
    .groupby("ds")[["best_forecast", "best_forecast-lo-95", "best_forecast-hi-95"]]
    .sum().reset_index()
)

# 绘制汇总图
fig_agg = go.Figure()

if len(agg_train) > 0:
    fig_agg.add_trace(go.Scatter(
        x=agg_train["ds"], y=agg_train["y"],
        mode="lines", name="训练集",
        line=dict(color="royalblue", width=1.5), opacity=0.8,
    ))

if len(agg_test) > 0:
    fig_agg.add_trace(go.Scatter(
        x=agg_test["ds"], y=agg_test["y"],
        mode="lines+markers", name="测试集真实值",
        line=dict(color="orange", width=1.5), marker=dict(size=3),
    ))

if len(agg_forecast) > 0:
    # 国家×商店×产品层级显示置信区间
    if selected_level == "国家 × 商店 × 产品":
        lo = agg_forecast["best_forecast-lo-95"]
        hi = agg_forecast["best_forecast-hi-95"]
        fig_agg.add_trace(go.Scatter(
            x=agg_forecast["ds"], y=hi,
            mode="lines", line=dict(width=0), showlegend=False,
        ))
        fig_agg.add_trace(go.Scatter(
            x=agg_forecast["ds"], y=lo,
            mode="lines", line=dict(width=0),
            fillcolor="rgba(0,200,100,0.2)", fill="tonexty",
            name="95% 置信区间",
        ))

    fig_agg.add_trace(go.Scatter(
        x=agg_forecast["ds"], y=agg_forecast["best_forecast"],
        mode="lines", name="预测值",
        line=dict(color="green", width=2.5),
    ))

# 分界线
if len(agg_train) > 0:
    train_end = agg_train["ds"].max()
    fig_agg.add_vline(x=train_end, line_dash="dash", line_color="gray", opacity=0.5)
    y_top = max(
        agg_forecast["best_forecast"].max() if len(agg_forecast) > 0 else 0,
        agg_train["y"].max() if len(agg_train) > 0 else 0,
    )
    fig_agg.add_annotation(
        x=train_end, y=y_top * 0.95,
        text="训练/预测分界", showarrow=False,
        font=dict(color="gray", size=10),
    )

fig_agg.update_layout(
    title=f"{level_label} — 日销量汇总",
    xaxis_title="日期",
    yaxis_title="总销量",
    hovermode="x unified",
    height=450,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

st.plotly_chart(fig_agg, use_container_width=True)


# ===== 未选国家时的默认展示 =====
if not selected_country:
    st.info("👈 在左侧边栏选择一个国家，查看多层级 Bottom-Up 汇总")

    col1, col2 = st.columns(2)

    with col1:
        type_counts = full_df["data_type"].value_counts()
        fig_pie = px.pie(
            values=type_counts.values, names=type_counts.index,
            title="数据量分布 (行数)",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        country_bar = (
            full_df[full_df["data_type"] == "训练集"]
            .groupby("country")["unique_id"]
            .nunique()
            .sort_values(ascending=True)
            .reset_index()
        )
        country_bar.columns = ["国家", "SKU 数量"]
        fig_bar = px.bar(
            country_bar.tail(20), x="SKU 数量", y="国家",
            orientation="h", title="Top 20 国家 SKU 数量",
            color="SKU 数量", color_continuous_scale="Viridis",
        )
        fig_bar.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_bar, use_container_width=True)