import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# PCワイド画面設定
st.set_page_config(
    page_title="米国マルチレバレッジETF 全天候型統合ダッシュボード",
    page_icon="🏛️",
    layout="wide"
)

# -------------------------------------------------------------
# 1. アセット定義
# -------------------------------------------------------------
PORTFOLIO_CONFIG = {
    "TQQQ": {"name": "TQQQ (NASDAQ 3倍)", "underlying": "QQQ", "type": "株式(Tech)", "default_w": 35, "hv_thresh": 28.0},
    "SPXL": {"name": "SPXL (S&P500 3倍)", "underlying": "SPY", "type": "株式(全体)", "default_w": 25, "hv_thresh": 22.0},
    "SOXL": {"name": "SOXL (半導体 3倍)", "underlying": "SOXX", "type": "株式(半導体)", "default_w": 15, "hv_thresh": 40.0},
    "FAS":  {"name": "FAS (金融株 3倍)",   "underlying": "XLF", "type": "株式(金融)", "default_w": 15, "hv_thresh": 25.0},
    "UGL":  {"name": "UGL (ゴールド 2倍)",  "underlying": "GLD", "type": "コモディティ", "default_w": 10, "hv_thresh": 20.0},
}

# -------------------------------------------------------------
# 2. データ取得 & キャッシュ
# -------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_all_market_data():
    all_tickers = ["QQQ", "SPY", "SOXX", "XLF", "GLD", "TQQQ", "SPXL", "SOXL", "FAS", "UGL"]
    data = {}
    for t in all_tickers:
        df = yf.download(t, period="5y", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        data[t] = df.dropna()
    return data

@st.cache_data(ttl=3600)
def get_usdjpy_rate():
    try:
        fx = yf.download("USDJPY=X", period="5d", interval="1d", progress=False)
        if isinstance(fx.columns, pd.MultiIndex):
            fx.columns = fx.columns.get_level_values(0)
        return round(float(fx["Close"].iloc[-1]), 2)
    except:
        return 155.0

with st.spinner("5大アセットの市場データおよび為替レートを読み込み中..."):
    market_data = load_all_market_data()
    latest_fx_rate = get_usdjpy_rate()

# 共通取引日インデックスの抽出
common_idx = market_data["QQQ"].index
for k in market_data.keys():
    common_idx = common_idx.intersection(market_data[k].index)

# -------------------------------------------------------------
# 3. サイドバー：配分比率 & 検証設定
# -------------------------------------------------------------
st.sidebar.title("🎛️ ポートフォリオ配分設定")

st.sidebar.markdown("### ① 資産クラス別 目標配分比率 (%)")
w_tqqq = st.sidebar.slider("TQQQ (NASDAQ 3倍)", 0, 100, 35, step=5)
w_spxl = st.sidebar.slider("SPXL (S&P500 3倍)", 0, 100, 25, step=5)
w_soxl = st.sidebar.slider("SOXL (半導体 3倍)", 0, 100, 15, step=5)
w_fas  = st.sidebar.slider("FAS (金融株 3倍)", 0, 100, 15, step=5)
w_ugl  = st.sidebar.slider("UGL (ゴールド 2倍)", 0, 100, 10, step=5)

total_w = w_tqqq + w_spxl + w_soxl + w_fas + w_ugl
if total_w != 100:
    st.sidebar.warning(f"⚠️ 合計比率が **{total_w}%** です（100%に調整してください）。")
    norm_factor = 100.0 / total_w if total_w > 0 else 1.0
else:
    st.sidebar.success("✅ 合計比率: **100%**")
    norm_factor = 1.0

# 正規化ウェイト (合計1.0)
weights = {
    "TQQQ": (w_tqqq * norm_factor) / 100.0,
    "SPXL": (w_spxl * norm_factor) / 100.0,
    "SOXL": (w_soxl * norm_factor) / 100.0,
    "FAS":  (w_fas * norm_factor) / 100.0,
    "UGL":  (w_ugl * norm_factor) / 100.0,
}

st.sidebar.markdown("---")
st.sidebar.markdown("### ② 戦略ロジック設定")
ema_span = st.sidebar.slider("移動平均日数 (EMA)", 100, 250, 200, step=10)
regime2_alloc = st.sidebar.slider("レジーム2 (波乱警戒時) の保有比率 (%)", 0, 100, 50, step=10) / 100.0

# -------------------------------------------------------------
# 4. 指標計算 & レジーム判定
# -------------------------------------------------------------
asset_signals = {}
strat_daily_rets = pd.Series(0.0, index=common_idx)
bm_daily_rets = pd.Series(0.0, index=common_idx)

for sym, cfg in PORTFOLIO_CONFIG.items():
    u_sym = cfg["underlying"]
    df_u = market_data[u_sym].loc[common_idx].copy()
    df_etf = market_data[sym].loc[common_idx].copy()
    
    # 指標計算
    alpha = 2.0 / (ema_span + 1.0)
    ema_series = df_u["Close"].ewm(alpha=alpha, adjust=False).mean()
    hv_series = (np.log(df_u["Close"] / df_u["Close"].shift(1)).rolling(20).std() * np.sqrt(252) * 100).fillna(0)
    
    # レジーム判定 (1.0 / r2 / 0.0)
    cond_bull = (df_u["Close"] > ema_series) & (hv_series < cfg["hv_thresh"])
    cond_warn = (df_u["Close"] > ema_series) & (hv_series >= cfg["hv_thresh"])
    pos_series = np.where(cond_bull, 1.0, np.where(cond_warn, regime2_alloc, 0.0))
    
    # 翌日執行 (shift 1)
    exec_pos = pd.Series(pos_series, index=common_idx).shift(1).fillna(0)
    etf_ret = df_etf["Close"].pct_change().fillna(0)
    
    w = weights[sym]
    strat_daily_rets += w * exec_pos * etf_ret
    bm_daily_rets += w * etf_ret
    
    # 最新値の保存
    last_c = df_u["Close"].iloc[-1]
    last_ema = ema_series.iloc[-1]
    last_hv = hv_series.iloc[-1]
    last_pos = pos_series[-1]
    
    if last_pos == 1.0:
        reg_name, reg_icon = "強気巡航", "🟢"
    elif last_pos > 0.0:
        reg_name, reg_icon = "波乱警戒", "🟡"
    else:
        reg_name, reg_icon = "弱気防衛", "🔴"
        
    asset_signals[sym] = {
        "name": cfg["name"],
        "underlying": u_sym,
        "u_close": last_c,
        "u_prev": df_u["Close"].iloc[-2],
        "ema": last_ema,
        "hv": last_hv,
        "thresh": cfg["hv_thresh"],
        "pos": last_pos,
        "reg_name": reg_name,
        "reg_icon": reg_icon,
        "etf_price": df_etf["Close"].iloc[-1]
    }

# -------------------------------------------------------------
# 5. バックテスト統計指標計算
# -------------------------------------------------------------
equity_strat = (1.0 + strat_daily_rets).cumprod()
equity_bm = (1.0 + bm_daily_rets).cumprod()

n_days = len(common_idx)
yrs = n_days / 252.0

cagr_strat = equity_strat.iloc[-1] ** (1.0 / yrs) - 1.0
cagr_bm = equity_bm.iloc[-1] ** (1.0 / yrs) - 1.0

vol_strat = strat_daily_rets.std() * np.sqrt(252)
vol_bm = bm_daily_rets.std() * np.sqrt(252)

peak_strat = equity_strat.cummax()
dd_strat = (equity_strat - peak_strat) / peak_strat
mdd_strat = dd_strat.min()

peak_bm = equity_bm.cummax()
dd_bm = (equity_bm - peak_bm) / peak_bm
mdd_bm = dd_bm.min()

calmar_strat = cagr_strat / abs(mdd_strat) if mdd_strat != 0 else 0
gains = strat_daily_rets[strat_daily_rets > 0].sum()
losses = abs(strat_daily_rets[strat_daily_rets < 0].sum())
pf_strat = gains / losses if losses != 0 else 0
t_stat_strat = (strat_daily_rets.mean() / (strat_daily_rets.std() / np.sqrt(n_days))) if strat_daily_rets.std() != 0 else 0

# -------------------------------------------------------------
# 6. メイン画面レイアウト
# -------------------------------------------------------------
st.title("🏛️ 米国マルチレバレッジ 全天候型統合ダッシュボード")
st.caption(f"対象銘柄: TQQQ / SPXL / SOXL / FAS / UGL ｜ 判定更新基準日: {common_idx[-1].strftime('%Y/%m/%d')}")

tab1, tab2, tab3 = st.tabs([
    "🏛️ 総合シグナル & 楽天証券 発注計算",
    "🧪 ポートフォリオ バックテスト検証 & 統計指標", 
    "📊 5大アセット テクニカル分析"
])

# =============================================================
# TAB 1: 総合シグナル & 発注計算
# =============================================================
with tab1:
    st.subheader("現在の各アセット市場レジーム判定")
    
    # 5銘柄のシグナルカード
    cols = st.columns(5)
    total_effective_exposure = 0.0
    for idx_c, (sym, sig) in enumerate(asset_signals.items()):
        w = weights[sym]
        effective_alloc = w * sig["pos"]
        total_effective_exposure += effective_alloc
        
        with cols[idx_c]:
            st.markdown(f"#### {sig['reg_icon']} {sym}")
            st.caption(f"{PORTFOLIO_CONFIG[sym]['type']} (基本枠: {int(w*100)}%)")
            st.metric(f"{sig['underlying']} 終値", f"${sig['u_close']:.2f}", f"{(sig['u_close']-sig['u_prev']):+.2f}")
            st.write(f"判定: **{sig['reg_name']}**")
            st.write(f"推奨投資枠: **{int(sig['pos']*100)}%**")
            st.progress(sig["pos"])
            st.caption(f"EMA: ${sig['ema']:.2f} ｜ HV: {sig['hv']:.1f}%")

    # 全体配分サマリー & ドーナツチャート
    cash_ratio = 1.0 - total_effective_exposure
    st.markdown("---")
    
    c_sum1, c_sum2 = st.columns([1.5, 1])
    with c_sum1:
        st.markdown("### 💼 ポートフォリオ総合エクスポージャー")
        m_col1, m_col2 = st.columns(2)
        m_col1.metric("総株式・ゴールド投資比率", f"{total_effective_exposure*100:.1f} %")
        m_col2.metric("安全待機キャッシュ比率 (MMF等)", f"{cash_ratio*100:.1f} %")
        st.info(f"💡 現在は全5資産のうち、市場環境が良いアセットだけを合計 **{total_effective_exposure*100:.1f}%** 保有し、残りの **{cash_ratio*100:.1f}%** は米ドルMMF（年利約4〜5%）で安全待機する指示となっています。")
        
    with c_sum2:
        pie_labels = [PORTFOLIO_CONFIG[s]["name"] for s in PORTFOLIO_CONFIG.keys()] + ["米ドル現金 / MMF"]
        pie_values = [weights[s] * asset_signals[s]["pos"] * 100 for s in PORTFOLIO_CONFIG.keys()] + [cash_ratio * 100]
        fig_pie = go.Figure(data=[go.Pie(
            labels=pie_labels,
            values=pie_values,
            hole=.45,
            marker_colors=['#00ba38', '#619cff', '#f5b041', '#9b59b6', '#f1c40f', '#b0b0b0']
        )])
        fig_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=200)
        st.plotly_chart(fig_pie, use_container_width=True)

    # 楽天証券 発注シミュレーター
    st.markdown("---")
    st.subheader("💡 楽天証券 寄り付き発注シミュレーター（日本円・米ドル両対応）")
    
    curr_col1, curr_col2, curr_col3 = st.columns([1.2, 1.5, 1.3])
    with curr_col1:
        input_currency = st.radio("入力通貨単位", ["日本円 (万円)", "米ドル (USD)"], horizontal=True)
    with curr_col2:
        if input_currency == "日本円 (万円)":
            funds_jpy_man = st.number_input("運用総資金額（万円）", min_value=10, value=500, step=10)
        else:
            funds_usd_in = st.number_input("運用総資金額（米ドル：USD）", min_value=1000, value=30000, step=1000)
    with curr_col3:
        fx_rate = st.number_input("適用為替レート (USD/JPY)", value=latest_fx_rate, step=0.5, format="%.2f")

    if input_currency == "日本円 (万円)":
        funds_total_jpy = funds_jpy_man * 10000.0
        funds_total_usd = funds_total_jpy / fx_rate
    else:
        funds_total_usd = float(funds_usd_in)
        funds_total_jpy = funds_total_usd * fx_rate

    st.success(f"💰 **運用総資産**: **${funds_total_usd:,.2f}** ＝ **約 {funds_total_jpy:,.0f} 円** （{funds_total_jpy/10000:,.1f} 万円）")

    sim_rows = []
    for sym in PORTFOLIO_CONFIG.keys():
        sig = asset_signals[sym]
        w = weights[sym]
        target_val_usd = funds_total_usd * w * sig["pos"]
        target_val_jpy = target_val_usd * fx_rate
        p = sig["etf_price"]
        shares = int(target_val_usd // p)
        
        sim_rows.append({
            "銘柄": sig["name"],
            "資産クラス": PORTFOLIO_CONFIG[sym]["type"],
            "目標金額 (USD)": f"${target_val_usd:,.2f}",
            "目標金額 (日本円)": f"約 {target_val_jpy:,.0f} 円 ({target_val_jpy/10000:,.1f}万)",
            "参考現在価格": f"${p:.2f}",
            "目標保有株数": f"{shares} 株",
            "今夜のアクション": f"{shares} 株に調整" if target_val_usd > 0 else "全売却 (0株)"
        })
        
    cash_val_usd = funds_total_usd * cash_ratio
    cash_val_jpy = cash_val_usd * fx_rate
    sim_rows.append({
        "銘柄": "米ドル現金 / MMF",
        "資産クラス": "安全待機キャッシュ",
        "目標金額 (USD)": f"${cash_val_usd:,.2f}",
        "目標金額 (日本円)": f"約 {cash_val_jpy:,.0f} 円 ({cash_val_jpy/10000:,.1f}万)",
        "参考現在価格": "-",
        "目標保有株数": "-",
        "今夜のアクション": "米ドルMMF等で安全待機 (利息年4〜5%)"
    })
    st.table(pd.DataFrame(sim_rows))

# =============================================================
# TAB 2: バックテスト検証 & 統計指標
# =============================================================
with tab2:
    st.subheader("🧪 5大アセット 全天候型ポートフォリオ 統計パフォーマンス")
    st.caption("5つの資産クラスを分散し、レジーム判定により下落相場を回避した場合の過去5年間バックテスト結果です。")
    
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("期待年利 (CAGR)", f"{cagr_strat*100:.1f} %", f"バイ＆ホールド: {cagr_bm*100:.1f}%")
    m2.metric("年率リスク (Vol)", f"{vol_strat*100:.1f} %", f"単主持: {vol_bm*100:.1f}%")
    m3.metric("最大下落率 (MDD)", f"{mdd_strat*100:.1f} %", f"単主持: {mdd_bm*100:.1f}%")
    m4.metric("プロフィットファクター (PF)", f"{pf_strat:.2f}")
    m5.metric("カルマーレシオ", f"{calmar_strat:.2f}")
    m6.metric("t値 (有意性)", f"{t_stat_strat:.2f}")

    st.markdown("---")
    
    # 資産推移 & ドローダウン
    fig_bt = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.7, 0.3])
    fig_bt.add_trace(go.Scatter(x=common_idx, y=equity_strat, name="全天候型レジーム運用", line=dict(color="#00ba38", width=2.5)), row=1, col=1)
    fig_bt.add_trace(go.Scatter(x=common_idx, y=equity_bm, name="5銘柄バイ＆ホールド (放置)", line=dict(color="#888888", width=1.5, dash="dot")), row=1, col=1)
    fig_bt.add_trace(go.Scatter(x=common_idx, y=dd_strat * 100, name="ドローダウン (%)", fill="tozeroy", line=dict(color="#d62728", width=1)), row=2, col=1)
    fig_bt.update_layout(height=500, margin=dict(t=20, b=20, l=10, r=10), hovermode="x unified")
    fig_bt.update_yaxes(title_text="資産倍率", row=1, col=1)
    fig_bt.update_yaxes(title_text="下落率 (%)", row=2, col=1)
    st.plotly_chart(fig_bt, use_container_width=True)

# =============================================================
# TAB 3: テクニカルチャート
# =============================================================
with tab3:
    st.subheader("📈 5大母体指数のテクニカル動向 (200日EMA比較)")
    chart_cols = st.columns(2)
    
    underlying_list = [("QQQ", "#1f77b4"), ("SPY", "#2ca02c"), ("SOXX", "#9467bd"), ("XLF", "#d62728"), ("GLD", "#bcbd22")]
    for i, (u_sym, color) in enumerate(underlying_list):
        col_target = chart_cols[i % 2]
        with col_target:
            st.markdown(f"**{u_sym} ＆ {ema_span}日EMA**")
            df_plot = market_data[u_sym].loc[common_idx]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=common_idx, y=df_plot["Close"], name=f"{u_sym} 終値", line=dict(color=color)))
            fig.add_trace(go.Scatter(x=common_idx, y=df_plot["Close"].ewm(span=ema_span, adjust=False).mean(), name=f"{ema_span}日EMA", line=dict(color="#ff7f0e", width=1.5)))
            fig.update_layout(height=300, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)
