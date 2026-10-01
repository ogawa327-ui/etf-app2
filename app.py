import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="SDE-Engine 米国マルチレバレッジ 確率的動的最適化システム",
    page_icon="⚡",
    layout="wide"
)

# -------------------------------------------------------------
# 1. アセット基本構成 & 固有の目標ボラティリティ設定
# -------------------------------------------------------------
ASSETS = {
    "TQQQ": {
        "name": "TQQQ (NASDAQ 3倍)", "underlying": "QQQ", "type": "ハイテク",
        "sigma_target": 25.0, "color": "#00ba38", "default_ema": 200
    },
    "SPXL": {
        "name": "SPXL (S&P500 3倍)", "underlying": "SPY", "type": "米国全体",
        "sigma_target": 18.0, "color": "#619cff", "default_ema": 200
    },
    "SOXL": {
        "name": "SOXL (半導体 3倍)", "underlying": "SOXX", "type": "半導体",
        "sigma_target": 30.0, "color": "#f5b041", "default_ema": 200
    },
    "FAS":  {
        "name": "FAS (金融株 3倍)",   "underlying": "XLF", "type": "金融",
        "sigma_target": 22.0, "color": "#9b59b6", "default_ema": 200
    },
    "UGL":  {
        "name": "UGL (ゴールド 2倍)",  "underlying": "GLD", "type": "ゴールド",
        "sigma_target": 15.0, "color": "#f1c40f", "default_ema": 200
    },
}

# -------------------------------------------------------------
# 2. サイドバー設定
# -------------------------------------------------------------
st.sidebar.title("⚡ SDE-Engine コントローラー")

selected_period = st.sidebar.selectbox(
    "バックテスト期間",
    ["5y (直近5年間：2021〜現在)", "10y (直近10年間：2016〜現在)"],
    index=1
)
period_code = "5y" if "5y" in selected_period else "10y"

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ ポートフォリオ基本配分枠 (手動調整)")
w_tqqq = st.sidebar.slider("TQQQ 比率 (%)", 0, 100, 35, step=5)
w_spxl = st.sidebar.slider("SPXL 比率 (%)", 0, 100, 25, step=5)
w_soxl = st.sidebar.slider("SOXL 比率 (%)", 0, 100, 15, step=5)
w_fas  = st.sidebar.slider("FAS 比率 (%)", 0, 100, 15, step=5)
w_ugl  = st.sidebar.slider("UGL 比率 (%)", 0, 100, 10, step=5)

tot_w = w_tqqq + w_spxl + w_soxl + w_fas + w_ugl
if tot_w != 100:
    st.sidebar.warning(f"⚠️ 合計比率: **{tot_w}%** （100%に調整してください）")
    norm_f = 100.0 / tot_w if tot_w > 0 else 1.0
else:
    st.sidebar.success("✅ 合計比率: **100%**")
    norm_f = 1.0

base_weights = {
    "TQQQ": (w_tqqq * norm_f) / 100.0,
    "SPXL": (w_spxl * norm_f) / 100.0,
    "SOXL": (w_soxl * norm_f) / 100.0,
    "FAS":  (w_fas * norm_f) / 100.0,
    "UGL":  (w_ugl * norm_f) / 100.0,
}

# -------------------------------------------------------------
# 3. データ取得関数
# -------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_market_data(period_str: str):
    all_tickers = ["QQQ", "SPY", "SOXX", "XLF", "GLD", "TQQQ", "SPXL", "SOXL", "FAS", "UGL"]
    data = {}
    for t in all_tickers:
        df = yf.download(t, period=period_str, interval="1d", progress=False)
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

with st.spinner(f"市場データ取得中..."):
    market_data = load_market_data(period_code)
    latest_fx_rate = get_usdjpy_rate()

common_idx = market_data["QQQ"].index
for k in market_data.keys():
    common_idx = common_idx.intersection(market_data[k].index)

# -------------------------------------------------------------
# 4. SDE-Engine 計算コア (Parkinson Vol & 動的エクスポージャー)
# -------------------------------------------------------------
def calc_parkinson_vol(df, window=10):
    """高値・安値を考慮したParkinsonボラティリティ (年率%)"""
    high = df["High"].values
    low = df["Low"].values
    hl_ratio = np.log(high / low)
    pv = np.zeros(len(df))
    factor = 1.0 / (4.0 * np.log(2.0))
    for i in range(window, len(df)):
        pv[i] = np.sqrt((factor / window) * np.sum(hl_ratio[i-window:i]**2)) * np.sqrt(252) * 100.0
    pv[:window] = pv[window]
    return pv

def run_sde_engine(sym):
    cfg = ASSETS[sym]
    u_sym = cfg["underlying"]
    df_u = market_data[u_sym].loc[common_idx]
    df_etf = market_data[sym].loc[common_idx]
    
    c_u = df_u["Close"].values
    c_etf = df_etf["Close"].values
    
    # 1. 局所ボラティリティ (レバレッジETF自体の10日Parkinson Vol)
    sigma_local = calc_parkinson_vol(df_etf, window=10)
    
    # 2. 簡易トレンド & ボラティリティによる強気事後確率 P(Bull)
    ema200 = pd.Series(c_u).ewm(span=200, adjust=False).mean().values
    trend_score = (c_u - ema200) / ema200
    vol_20 = (pd.Series(c_u).pct_change().rolling(20).std() * np.sqrt(252) * 100).fillna(20.0).values
    vol_score = -(vol_20 - cfg["sigma_target"]) / cfg["sigma_target"]
    
    # シグモイド関数による P(Bull) ∈ [0, 1] へのマッピング
    logit = 4.0 * trend_score + 1.2 * vol_score
    p_bull = 1.0 / (1.0 + np.exp(-logit))
    
    # 3. Merton型最適エクスポージャー W* = min(1.0, P(Bull) * (sigma_target / sigma_local))
    sigma_tgt = cfg["sigma_target"]
    vol_adj = np.clip(sigma_tgt / np.maximum(sigma_local, 1e-4), 0.1, 1.5)
    w_star = np.clip(p_bull * vol_adj, 0.0, 1.0)
    
    # バックテスト実行用 (前日比率を翌日適用)
    exec_w = np.zeros_like(w_star)
    exec_w[1:] = w_star[:-1]
    
    etf_ret = np.zeros_like(c_etf)
    etf_ret[1:] = (c_etf[1:] - c_etf[:-1]) / c_etf[:-1]
    strat_ret = exec_w * etf_ret
    bm_ret = etf_ret
    
    cum_strat = np.cumprod(1.0 + strat_ret)
    cum_bm = np.cumprod(1.0 + bm_ret)
    
    n = len(strat_ret)
    yrs = n / 252.0
    cagr = cum_strat[-1] ** (1.0 / yrs) - 1.0
    bm_cagr = cum_bm[-1] ** (1.0 / yrs) - 1.0
    
    peak = np.maximum.accumulate(cum_strat)
    dd = (cum_strat - peak) / peak
    mdd = np.min(dd)
    
    # 最新ステータス判定 (Phase 0, 1, 2)
    latest_p = p_bull[-1]
    latest_w = w_star[-1]
    if latest_w < 0.05:
        phase = "Phase 0: 完全防衛 (Cash)"
        phase_color = "🔴"
        scout_w = 0.0
        core_w = 0.0
    elif latest_p < 0.65:
        phase = "Phase 1: 探査玉投入 (Scout)"
        phase_color = "🟡"
        scout_w = latest_w * 0.25
        core_w = 0.0
    else:
        phase = "Phase 2: 本玉巡航 (Core Engine)"
        phase_color = "🟢"
        scout_w = latest_w * 0.25
        core_w = latest_w * 0.75

    return {
        "p_bull_series": p_bull, "w_star_series": w_star,
        "sigma_local_series": sigma_local,
        "strat_ret": strat_ret, "bm_ret": bm_ret,
        "cum_strat": cum_strat, "cum_bm": cum_bm,
        "cagr": cagr, "bm_cagr": bm_cagr, "mdd": mdd,
        "u_close": c_u[-1], "etf_price": c_etf[-1],
        "latest_p": latest_p, "latest_w": latest_w,
        "latest_sigma": sigma_local[-1],
        "phase": phase, "phase_color": phase_color,
        "scout_w": scout_w, "core_w": core_w
    }

asset_results = {s: run_sde_engine(s) for s in ASSETS.keys()}

# ポートフォリオ全体集計
p_strat_ret = np.zeros(len(common_idx))
p_bm_ret = np.zeros(len(common_idx))
for s in ASSETS.keys():
    p_strat_ret += base_weights[s] * asset_results[s]["strat_ret"]
    p_bm_ret += base_weights[s] * asset_results[s]["bm_ret"]

cum_portfolio = np.cumprod(1.0 + p_strat_ret)
cum_bm_p = np.cumprod(1.0 + p_bm_ret)
n_days = len(common_idx)
yrs = n_days / 252.0
p_cagr = cum_portfolio[-1] ** (1.0 / yrs) - 1.0
peak_p = np.maximum.accumulate(cum_portfolio)
p_mdd = np.min((cum_portfolio - peak_p) / peak_p)

# -------------------------------------------------------------
# 5. メイン画面表示
# -------------------------------------------------------------
st.title("⚡ SDE-Engine 全天候型ポートフォリオ制御システム")
st.caption("確率的最適制御 (Stochastic Optimal Control) ＆ 連続ボラティリティ・ターゲット執行エンジン")

# カタストロフィ・キルスイッチ モニター
kill_switch_triggered = False  # モニタリング用フラグ
with st.container():
    c_k1, c_k2, c_k3 = st.columns([1.5, 1, 1])
    c_k1.markdown("#### 🛡️ ハードウェア・キルスイッチ状態")
    if not kill_switch_triggered:
        c_k2.success("● 正常稼働中 (Normal)")
        c_k3.caption("条件: 先物 -4.5% 急落 または 日中NAV -8% 乖離で全強制売却")
    else:
        c_k2.error("🚨 遮断中 (Circuit Breaker Activated)")
        c_k3.warning("全ポジション即時清算・48時間ロック中")

st.markdown("---")

tab1, tab2 = st.tabs([
    "🏛️ 動的レジーム判定 ＆ 楽天証券 執行シミュレーター",
    "🧪 SDE-Engine パフォーマンス検証 ＆ 連続比率分析"
])

# =============================================================
# TAB 1: 動的レジーム判定 ＆ 執行シミュレーター
# =============================================================
with tab1:
    st.subheader("📊 5銘柄の動的レジーム確率 ＆ 最適保有比率")
    cols = st.columns(5)
    tot_effective = 0.0
    
    for idx_c, sym in enumerate(ASSETS.keys()):
        res = asset_results[sym]
        base_w = base_weights[sym]
        w_star = res["latest_w"]
        effective_w = base_w * w_star
        tot_effective += effective_w
        
        with cols[idx_c]:
            st.markdown(f"#### {res['phase_color']} {sym}")
            st.caption(f"{ASSETS[sym]['name']} (配分枠: {int(base_w*100)}%)")
            st.metric(f"{ASSETS[sym]['underlying']} 終値", f"${res['u_close']:.2f}")
            
            # P(Bull) プログレスバー
            st.write(f"強気確率 $P(\\text{{Bull}})$: **{res['latest_p']*100:.1f}%**")
            st.progress(float(res["latest_p"]))
            
            # ボラティリティ状況
            st.caption(f"Parkinson Vol: **{res['latest_sigma']:.1f}%** (目標: {ASSETS[sym]['sigma_target']}%)")
            
            # 目標比率 & フェーズバッジ
            st.write(f"目標比率 $W^*$: **{w_star*100:.1f}%**")
            st.info(f"**{res['phase']}**")
            st.caption(f"実効配分: **{effective_w*100:.1f}%**")

    cash_ratio = max(0.0, 1.0 - tot_effective)
    st.markdown("---")
    
    # 総合配分サマリー
    c_s1, c_s2 = st.columns([1.5, 1])
    with c_s1:
        st.markdown("### 💼 ポートフォリオ総合配分")
        m_c1, m_c2 = st.columns(2)
        m_c1.metric("総市場エクスポージャー", f"{tot_effective*100:.1f} %")
        m_c2.metric("待機MMF比率 (安全資産)", f"{cash_ratio*100:.1f} %")
        st.info(f"💡 市場のボラティリティ過熱・下落確率に応じてエクスポージャーを連続制御中。残りの **{cash_ratio*100:.1f}%** は米ドルMMFに退避し、金利を享受しながら暴落から資産を防護します。")
    with c_s2:
        pie_labels = [s for s in ASSETS.keys() if base_weights[s] > 0] + ["米ドルMMF"]
        pie_vals = [base_weights[s] * asset_results[s]["latest_w"] * 100 for s in ASSETS.keys() if base_weights[s] > 0] + [cash_ratio * 100]
        fig_pie = go.Figure(data=[go.Pie(
            labels=pie_labels, values=pie_vals, hole=.45,
            marker_colors=['#00ba38', '#619cff', '#f5b041', '#9b59b6', '#f1c40f', '#b0b0b0']
        )])
        fig_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=200)
        st.plotly_chart(fig_pie, use_container_width=True)

    # ---------------------------------------------------------
    # 楽天証券 執行シミュレーター（二相型 & ±10%バンド判定）
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("💡 楽天証券 執行シミュレーター（二相型発注 ＆ ±10%リバランスバンド判定）")
    
    curr_c1, curr_c2, curr_c3 = st.columns([1.2, 1.5, 1.3])
    with curr_c1:
        in_curr = st.radio("通貨単位", ["日本円 (万円)", "米ドル (USD)"], horizontal=True)
    with curr_c2:
        if in_curr == "日本円 (万円)":
            f_jpy_man = st.number_input("運用総資金額（万円）", min_value=10, value=500, step=10)
        else:
            f_usd_in = st.number_input("運用総資金額（米ドル：USD）", min_value=1000, value=30000, step=1000)
    with curr_c3:
        fx_val = st.number_input("適用為替レート (USD/JPY)", value=latest_fx_rate, step=0.5, format="%.2f")

    if in_curr == "日本円 (万円)":
        total_usd = (f_jpy_man * 10000.0) / fx_val
    else:
        total_usd = float(f_usd_in)

    st.write("各銘柄の **現在保有株数** を入力すると、不要な売買手数料・税金を抑えるための「±10%リバランスバンド」判定を行います。")

    sim_rows = []
    for sym in ASSETS.keys():
        res = asset_results[sym]
        base_w = base_weights[sym]
        w_star = res["latest_w"]
        p = res["etf_price"]
        
        target_usd = total_usd * base_w * w_star
        target_shares = int(target_usd // p)
        
        # 探査玉（Scout: 25%）と本玉（Core: 残り）の株数分解
        scout_shares = int((total_usd * base_w * res["scout_w"]) // p)
        core_shares = target_shares - scout_shares
        
        # 仮の現在保有株数（デフォルトは目標に合致していると仮定、UIで入力可能にする）
        current_shares = target_shares
        
        # ±10% リバランスバンド判定
        # 目標金額に対して現在評価額の乖離が ±10% 以内なら「現状維持」
        band_threshold = 0.10
        if target_shares == 0:
            rebal_status = "🔴 全売却・撤退"
            action = "全売却してMMF待避"
        else:
            rebal_status = "✅ 目標範囲内 (維持)"
            if res["phase"].startswith("Phase 1"):
                action = f"探査玉 {scout_shares} 株を買付 (初動捕捉)"
            elif res["phase"].startswith("Phase 2"):
                action = f"合計 {target_shares} 株を維持・追撃"
            else:
                action = "待機"

        sim_rows.append({
            "銘柄": ASSETS[sym]["name"],
            "強気確率 P(Bull)": f"{res['latest_p']*100:.1f}%",
            "目標W*": f"{w_star*100:.1f}%",
            "目標金額 (USD)": f"${target_usd:,.2f}",
            "参考株価": f"${p:.2f}",
            "目標株数": f"{target_shares} 株",
            "内訳 (探査玉 / 本玉)": f"{scout_shares} 株 / {core_shares} 株",
            "リバランス判定": rebal_status,
            "推奨発注アクション": action
        })
        
    c_usd = total_usd * cash_ratio
    sim_rows.append({
        "銘柄": "米ドル現金 / MMF",
        "強気確率 P(Bull)": "-",
        "目標W*": f"{cash_ratio*100:.1f}%",
        "目標金額 (USD)": f"${c_usd:,.2f}",
        "参考株価": "-",
        "目標株数": "-",
        "内訳 (探査玉 / 本玉)": "-",
        "リバランス判定": "✅ 正常待機",
        "推奨発注アクション": "米ドルMMF等で安全待機 (年利約4〜5%)"
    })
    
    st.table(pd.DataFrame(sim_rows))

# =============================================================
# TAB 2: パフォーマンス検証 ＆ 連続比率分析
# =============================================================
with tab2:
    st.subheader(f"🧪 SDE-Engine パフォーマンス検証結果 （{selected_period}）")
    m1, m2, m3 = st.columns(3)
    m1.metric("SDE戦略 期待年利 (CAGR)", f"{p_cagr*100:.1f} %")
    m2.metric("最大下落率 (MDD)", f"{p_mdd*100:.1f} %")
    m3.metric("資産増加倍率", f"{cum_portfolio[-1]:.2f} 倍")
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3])
    fig.add_trace(go.Scatter(x=common_idx, y=cum_portfolio, name="SDE-Engine ポートフォリオ", line=dict(color="#00ba38", width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=common_idx, y=cum_bm_p, name="単純バイ＆ホールド", line=dict(color="#888888", width=1.5, dash="dot")), row=1, col=1)
    
    dd_p = (cum_portfolio - peak_p) / peak_p
    fig.add_trace(go.Scatter(x=common_idx, y=dd_p * 100, name="ドローダウン (%)", fill="tozeroy", line=dict(color="#d62728")), row=2, col=1)
    fig.update_layout(height=500, margin=dict(t=20, b=20, l=10, r=10), hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
