import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# PC大画面向けワイドレイアウト設定
st.set_page_config(
    page_title="米国レバレッジETF ポートフォリオダッシュボード (TQQQ / SOXL)",
    page_icon="📈",
    layout="wide"
)

# -------------------------------------------------------------
# データ取得 & 指標計算関数 (キャッシュ機能付き)
# -------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_and_calculate(ticker: str, hv_window: int = 20):
    df = yf.download(ticker, period="3y", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df.dropna()
    # 200日・50日 指数平滑移動平均 (EMA)
    df["EMA_200"] = df["Close"].ewm(span=200, adjust=False).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
    
    # 乖離率 (%)
    df["EMA_200_Bias"] = ((df["Close"] - df["EMA_200"]) / df["EMA_200"]) * 100
    
    # 20日 ヒストリカル・ボラティリティ (HV20)
    df["Log_Ret"] = np.log(df["Close"] / df["Close"].shift(1))
    df["HV20"] = df["Log_Ret"].rolling(window=hv_window).std() * np.sqrt(252) * 100
    
    # RSI (14)
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    
    return df

# レジーム判定ロジック
def evaluate_regime(close_price, ema_200, hv20, hv_threshold):
    if close_price > ema_200:
        if hv20 < hv_threshold:
            return 1, "レジーム1：強気巡航", "🟢", 1.0, "安定上昇中。100%全力投資・継続保有"
        else:
            return 2, "レジーム2：波乱警戒", "🟡", 0.5, "高ボラティリティ。保有比率を50%に半減"
    else:
        return 3, "レジーム3：弱気防衛", "🔴", 0.0, "下落トレンド。全売却してキャッシュ（MMF）退避"

# -------------------------------------------------------------
# データ読み込み
# -------------------------------------------------------------
with st.spinner("最新の市場データを取得中..."):
    qqq_df = fetch_and_calculate("QQQ", hv_window=20)
    soxx_df = fetch_and_calculate("SOXX", hv_window=20)
    tqqq_df = yf.download("TQQQ", period="1y", interval="1d", progress=False)
    soxl_df = yf.download("SOXL", period="1y", interval="1d", progress=False)

if isinstance(tqqq_df.columns, pd.MultiIndex):
    tqqq_df.columns = tqqq_df.columns.get_level_values(0)
if isinstance(soxl_df.columns, pd.MultiIndex):
    soxl_df.columns = soxl_df.columns.get_level_values(0)

# 最新指標値
q_latest = qqq_df.iloc[-1]
q_prev = qqq_df.iloc[-2]
s_latest = soxx_df.iloc[-1]
s_prev = soxx_df.iloc[-2]

tqqq_latest_price = tqqq_df["Close"].iloc[-1]
soxl_latest_price = soxl_df["Close"].iloc[-1]

# レジーム計算 (QQQ: 28%しきい値, SOXX: 40%しきい値)
q_reg, q_title, q_icon, q_alloc, q_desc = evaluate_regime(q_latest["Close"], q_latest["EMA_200"], q_latest["HV20"], 28.0)
s_reg, s_title, s_icon, s_alloc, s_desc = evaluate_regime(s_latest["Close"], s_latest["EMA_200"], s_latest["HV20"], 40.0)

# -------------------------------------------------------------
# ダッシュボード UI
# -------------------------------------------------------------
st.title("🛡️ 米国レバレッジETF 統合投資ダッシュボード")
st.caption(f"最終更新基準日: {q_latest.name.strftime('%Y-%m-%d')} ｜ 運用ポートフォリオ：TQQQ（70%）+ SOXL（30%）")

# --- タブ構成 ---
tab1, tab2, tab3, tab4 = st.tabs([
    "🏛️ ポートフォリオ統合シグナル & 発注計算", 
    "📊 QQQ / TQQQ 詳細テクニカル", 
    "⚡ SOXX / SOXL 詳細テクニカル",
    "📈 パフォーマンス比較"
])

# =============================================================
# TAB 1: 統合シグナル & 発注株数シミュレーター
# =============================================================
with tab1:
    st.subheader("現在の市場レジームと推奨ポジション")
    
    col1, col2, col3 = st.columns([1.2, 1.2, 1.6])
    
    with col1:
        st.markdown(f"### {q_icon} TQQQ（配分 70%）")
        st.metric("QQQ 終値", f"${q_latest['Close']:.2f}", f"{(q_latest['Close']-q_prev['Close']):+.2f}")
        st.write(f"**判定**: {q_title}")
        st.progress(q_alloc)
        st.write(f"推奨比率: **{int(q_alloc*100)}%** ({q_desc})")
        
    with col2:
        st.markdown(f"### {s_icon} SOXL（配分 30%）")
        st.metric("SOXX 終値", f"${s_latest['Close']:.2f}", f"{(s_latest['Close']-s_prev['Close']):+.2f}")
        st.write(f"**判定**: {s_title}")
        st.progress(s_alloc)
        st.write(f"推奨比率: **{int(s_alloc*100)}%** ({s_desc})")
        
    with col3:
        total_market_exposure = (0.70 * q_alloc) + (0.30 * s_alloc)
        cash_ratio = 1.0 - total_market_exposure
        st.markdown("### 💼 ポートフォリオ総合配分")
        st.metric("株式エクスポージャー", f"{total_market_exposure*100:.1f} %", f"待機キャッシュ: {cash_ratio*100:.1f}%")
        
        # 配分パイチャート
        fig_pie = go.Figure(data=[go.Pie(
            labels=['TQQQ (株式)', 'SOXL (株式)', '米ドル現金 / MMF'],
            values=[0.70 * q_alloc * 100, 0.30 * s_alloc * 100, cash_ratio * 100],
            hole=.4,
            marker_colors=['#00ba38', '#619cff', '#b0b0b0']
        )])
        fig_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=180)
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")
    st.subheader("💡 楽天証券 寄り付き発注シミュレーター")
    st.write("運用予定の総資産を入力すると、今夜の米国市場寄り付きで保有すべき目標株数と金額を自動計算します。")
    
    calc_col1, calc_col2 = st.columns([1, 2])
    with calc_col1:
        total_funds = st.number_input("運用総資金額（米ドル：USD）", min_value=1000, value=30000, step=1000)
        st.caption(f"（参考: 1ドル=155円換算で 約 {total_funds * 155:,.0f} 円）")
        
    with calc_col2:
        target_tqqq_val = total_funds * 0.70 * q_alloc
        target_soxl_val = total_funds * 0.30 * s_alloc
        target_cash_val = total_funds * cash_ratio
        
        tqqq_shares = int(target_tqqq_val // tqqq_latest_price)
        soxl_shares = int(target_soxl_val // soxl_latest_price)
        
        res_data = {
            "対象銘柄": ["TQQQ (NASDAQ 3倍)", "SOXL (半導体 3倍)", "現金 / MMF"],
            "目標金額": [f"${target_tqqq_val:,.2f}", f"${target_soxl_val:,.2f}", f"${target_cash_val:,.2f}"],
            "参考現在価格": [f"${tqqq_latest_price:.2f}", f"${soxl_latest_price:.2f}", "-"],
            "目標保有株数": [f"{tqqq_shares} 株", f"{soxl_shares} 株", "-"],
            "今夜のアクション": [
                f"現在の保有数を {tqqq_shares} 株に合わせる" if q_alloc > 0 else "全売却（0株）",
                f"現在の保有数を {soxl_shares} 株に合わせる" if s_alloc > 0 else "全売却（0株）",
                "余剰分は米ドルMMF等で安全待機"
            ]
        }
        st.table(pd.DataFrame(res_data))

# =============================================================
# TAB 2: QQQ / TQQQ 詳細テクニカルチャート
# =============================================================
with tab2:
    st.subheader("QQQ (母体指数) 詳細テクニカル分析")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("200日 EMA", f"${q_latest['EMA_200']:.2f}")
    m2.metric("200日 EMA 乖離率", f"{q_latest['EMA_200_Bias']:+.2f} %")
    m3.metric("20日ボラティリティ (HV20)", f"{q_latest['HV20']:.1f} %", "しきい値: 28.0%")
    m4.metric("RSI (14)", f"{q_latest['RSI']:.1f}")
    
    # チャート作成 (2段サブプロット: 上段 価格/EMA, 下段 HV20)
    fig_q = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    
    fig_q.add_trace(go.Scatter(x=qqq_df.index, y=qqq_df['Close'], name='QQQ 終値', line=dict(color='#1f77b4', width=1.5)), row=1, col=1)
    fig_q.add_trace(go.Scatter(x=qqq_df.index, y=qqq_df['EMA_200'], name='200日 EMA', line=dict(color='#ff7f0e', width=2)), row=1, col=1)
    fig_q.add_trace(go.Scatter(x=qqq_df.index, y=qqq_df['EMA_50'], name='50日 EMA', line=dict(color='#2ca02c', width=1, dash='dash')), row=1, col=1)
    
    fig_q.add_trace(go.Scatter(x=qqq_df.index, y=qqq_df['HV20'], name='HV20 (ボラティリティ)', line=dict(color='#d62728', width=1.5)), row=2, col=1)
    fig_q.add_hline(y=28.0, line_dash="dash", line_color="orange", annotation_text="波乱しきい値 (28%)", row=2, col=1)
    
    fig_q.update_layout(height=550, margin=dict(t=20, b=20, l=10, r=10), hovermode="x unified")
    st.plotly_chart(fig_q, use_container_width=True)

# =============================================================
# TAB 3: SOXX / SOXL 詳細テクニカルチャート
# =============================================================
with tab3:
    st.subheader("SOXX (半導体母体指数) 詳細テクニカル分析")
    
    sm1, sm2, sm3, sm4 = st.columns(4)
    sm1.metric("200日 EMA", f"${s_latest['EMA_200']:.2f}")
    sm2.metric("200日 EMA 乖離率", f"{s_latest['EMA_200_Bias']:+.2f} %")
    sm3.metric("20日ボラティリティ (HV20)", f"{s_latest['HV20']:.1f} %", "半導体しきい値: 40.0%")
    sm4.metric("RSI (14)", f"{s_latest['RSI']:.1f}")
    
    fig_s = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    
    fig_s.add_trace(go.Scatter(x=soxx_df.index, y=soxx_df['Close'], name='SOXX 終値', line=dict(color='#9467bd', width=1.5)), row=1, col=1)
    fig_s.add_trace(go.Scatter(x=soxx_df.index, y=soxx_df['EMA_200'], name='200日 EMA', line=dict(color='#ff7f0e', width=2)), row=1, col=1)
    fig_s.add_trace(go.Scatter(x=soxx_df.index, y=soxx_df['EMA_50'], name='50日 EMA', line=dict(color='#2ca02c', width=1, dash='dash')), row=1, col=1)
    
    fig_s.add_trace(go.Scatter(x=soxx_df.index, y=soxx_df['HV20'], name='HV20 (ボラティリティ)', line=dict(color='#d62728', width=1.5)), row=2, col=1)
    fig_s.add_hline(y=40.0, line_dash="dash", line_color="red", annotation_text="波乱警戒ライン (40%)", row=2, col=1)
    
    fig_s.update_layout(height=550, margin=dict(t=20, b=20, l=10, r=10), hovermode="x unified")
    st.plotly_chart(fig_s, use_container_width=True)

# =============================================================
# TAB 4: パフォーマンス比較 & ドローダウン
# =============================================================
with tab4:
    st.subheader("QQQ vs SOXX 相対パフォーマンス比較 (直近3年)")
    
    # 規格化（初日を100とする）
    norm_q = (qqq_df['Close'] / qqq_df['Close'].iloc[0]) * 100
    norm_s = (soxx_df['Close'] / soxx_df['Close'].iloc[0]) * 100
    
    fig_perf = go.Figure()
    fig_perf.add_trace(go.Scatter(x=norm_q.index, y=norm_q, name='QQQ (NASDAQ100)', line=dict(color='#1f77b4', width=2)))
    fig_perf.add_trace(go.Scatter(x=norm_s.index, y=norm_s, name='SOXX (半導体)', line=dict(color='#9467bd', width=2)))
    fig_perf.update_layout(height=450, margin=dict(t=20, b=20, l=10, r=10), yaxis_title="基準値 (初日=100)")
    st.plotly_chart(fig_perf, use_container_width=True)
