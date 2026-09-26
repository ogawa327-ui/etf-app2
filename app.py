import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# PCワイド画面設定
st.set_page_config(
    page_title="米国マルチレバレッジ 科学的ポートフォリオ最適化システム",
    page_icon="🏛️",
    layout="wide"
)

# -------------------------------------------------------------
# 1. アセット基本構成
# -------------------------------------------------------------
ASSETS = {
    "TQQQ": {"name": "TQQQ (NASDAQ 3倍)", "underlying": "QQQ", "type": "ハイテク", "default_ema": 200, "default_hv": 28.0, "hv_grid": [24.0, 28.0, 32.0]},
    "SPXL": {"name": "SPXL (S&P500 3倍)", "underlying": "SPY", "type": "米国全体", "default_ema": 200, "default_hv": 22.0, "hv_grid": [18.0, 22.0, 26.0]},
    "SOXL": {"name": "SOXL (半導体 3倍)", "underlying": "SOXX", "type": "半導体",   "default_ema": 200, "default_hv": 40.0, "hv_grid": [35.0, 40.0, 45.0]},
    "FAS":  {"name": "FAS (金融株 3倍)",   "underlying": "XLF", "type": "金融",     "default_ema": 200, "default_hv": 25.0, "hv_grid": [22.0, 26.0, 30.0]},
    "UGL":  {"name": "UGL (ゴールド 2倍)",  "underlying": "GLD", "type": "ゴールド", "default_ema": 200, "default_hv": 20.0, "hv_grid": [16.0, 20.0, 24.0]},
}

# -------------------------------------------------------------
# 2. サイドバー：データ期間 & 最適化コントローラー
# -------------------------------------------------------------
st.sidebar.title("🔬 科学的最適化ラボ")

st.sidebar.markdown("### 📅 データ検証期間の選択")
selected_period = st.sidebar.selectbox(
    "バックテスト期間",
    ["5y (直近5年間：2021〜現在)", "10y (直近10年間：2016〜現在)"],
    index=0
)
period_code = "5y" if "5y" in selected_period else "10y"

# -------------------------------------------------------------
# 3. データ取得 & 為替レート
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

with st.spinner(f"市場データ（{selected_period}）を取得・インデックス調整中..."):
    market_data = load_market_data(period_code)
    latest_fx_rate = get_usdjpy_rate()

common_idx = market_data["QQQ"].index
for k in market_data.keys():
    common_idx = common_idx.intersection(market_data[k].index)

# -------------------------------------------------------------
# 4. セッション状態の初期化
# -------------------------------------------------------------
for sym in ASSETS.keys():
    if f"ema_{sym}" not in st.session_state:
        st.session_state[f"ema_{sym}"] = ASSETS[sym]["default_ema"]
    if f"hv_{sym}" not in st.session_state:
        st.session_state[f"hv_{sym}"] = ASSETS[sym]["default_hv"]

if "w_TQQQ" not in st.session_state:
    st.session_state["w_TQQQ"] = 35
if "w_SPXL" not in st.session_state:
    st.session_state["w_SPXL"] = 25
if "w_SOXL" not in st.session_state:
    st.session_state["w_SOXL"] = 15
if "w_FAS" not in st.session_state:
    st.session_state["w_FAS"] = 15
if "w_UGL" not in st.session_state:
    st.session_state["w_UGL"] = 10
if "regime2_alloc" not in st.session_state:
    st.session_state["regime2_alloc"] = 50
if "opt_msg" not in st.session_state:
    st.session_state["opt_msg"] = None

# -------------------------------------------------------------
# 5. 高速単体バックテスト & 最適化関数
# -------------------------------------------------------------
def eval_single_asset(sym, ema_s, hv_t, r2_val):
    u_sym = ASSETS[sym]["underlying"]
    c_u = market_data[u_sym].loc[common_idx, "Close"].values
    c_etf = market_data[sym].loc[common_idx, "Close"].values
    
    alpha = 2.0 / (ema_s + 1.0)
    ema_vals = pd.Series(c_u).ewm(alpha=alpha, adjust=False).mean().values
    log_ret = np.log(pd.Series(c_u) / pd.Series(c_u).shift(1))
    hv_vals = (log_ret.rolling(20).std() * np.sqrt(252) * 100).fillna(0).values
    
    bull = (c_u > ema_vals) & (hv_vals < hv_t)
    warn = (c_u > ema_vals) & (hv_vals >= hv_t)
    pos = np.where(bull, 1.0, np.where(warn, r2_val, 0.0))
    
    # 翌日執行
    exec_pos = np.zeros_like(pos)
    exec_pos[1:] = pos[:-1]
    
    etf_ret = np.zeros_like(c_etf)
    etf_ret[1:] = (c_etf[1:] - c_etf[:-1]) / c_etf[:-1]
    
    strat_ret = exec_pos * etf_ret
    return strat_ret, pos[-1], ema_vals[-1], hv_vals[-1], c_u[-1], c_etf[-1]

# -------------------------------------------------------------
# 6. 自動最適化ルーチン（銘柄別パラメータ ＋ 最適配分比率）
# -------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 科学的自動最適化エンジン")
opt_target = st.sidebar.selectbox("最適化目標", ["カルマーレシオ最大化（下落抑制＆効率）", "プロフィットファクター最大化（利益損失比）"])

if st.sidebar.button("🚀 銘柄別パラメータ＆最適配分を一括自動算出", type="primary", use_container_width=True):
    with st.spinner("全銘柄のパラメータ空間＆配分比率をモンテカルロ解析中..."):
        target_mode = "Calmar" if "カルマー" in opt_target else "PF"
        r2_now = st.session_state["regime2_alloc"] / 100.0
        
        # Step 1: 各銘柄ごとの最適EMAとHVを探索
        best_ema = {}
        best_hv = {}
        single_rets = {}
        
        ema_test_candidates = [150, 175, 200, 220]
        for sym, cfg in ASSETS.items():
            best_s_score = -9999
            b_e, b_h = cfg["default_ema"], cfg["default_hv"]
            for e_candidate in ema_test_candidates:
                for h_candidate in cfg["hv_grid"]:
                    s_ret, _, _, _, _, _ = eval_single_asset(sym, e_candidate, h_candidate, r2_now)
                    cum = np.cumprod(1.0 + s_ret)
                    yrs = len(s_ret) / 252.0
                    cagr = cum[-1] ** (1.0 / yrs) - 1.0
                    peak = np.maximum.accumulate(cum)
                    dd = (cum - peak) / peak
                    mdd = np.min(dd)
                    gains = np.sum(s_ret[s_ret > 0])
                    losses = np.abs(np.sum(s_ret[s_ret < 0]))
                    
                    if target_mode == "Calmar":
                        score = cagr / abs(mdd) if mdd != 0 else 0
                    else:
                        score = gains / losses if losses != 0 else 0
                        
                    if score > best_s_score:
                        best_s_score = score
                        b_e = e_candidate
                        b_h = h_candidate
                        
            best_ema[sym] = b_e
            best_hv[sym] = b_h
            ret_opt, _, _, _, _, _ = eval_single_asset(sym, b_e, b_h, r2_now)
            single_rets[sym] = ret_opt
            
            st.session_state[f"ema_{sym}"] = b_e
            st.session_state[f"hv_{sym}"] = b_h

        # Step 2: 5銘柄の最適配分比率を探索（ディリクレ・モンテカルロ 4,000試行）
        ret_matrix = np.column_stack([single_rets[s] for s in ASSETS.keys()])
        n_days = len(common_idx)
        yrs = n_days / 252.0
        
        np.random.seed(42)
        n_sim = 4000
        sim_weights = np.random.dirichlet(np.ones(5), size=n_sim)
        
        best_p_score = -9999
        best_weights = None
        
        for w_cand in sim_weights:
            # 5%刻みに丸め
            w_round = np.round(w_cand * 20) / 20.0
            if np.sum(w_round) == 0:
                continue
            w_round = w_round / np.sum(w_round)
            
            p_ret = np.dot(ret_matrix, w_round)
            cum = np.cumprod(1.0 + p_ret)
            cagr = cum[-1] ** (1.0 / yrs) - 1.0
            peak = np.maximum.accumulate(cum)
            mdd = np.min((cum - peak) / peak)
            gains = np.sum(p_ret[p_ret > 0])
            losses = np.abs(np.sum(p_ret[p_ret < 0]))
            
            if target_mode == "Calmar":
                score = cagr / abs(mdd) if mdd != 0 else 0
            else:
                score = gains / losses if losses != 0 else 0
                
            if score > best_p_score:
                best_p_score = score
                best_weights = w_round
                
        # 配分をセッションに反映 (整数%)
        keys_list = list(ASSETS.keys())
        w_pct = [int(round(x * 100)) for x in best_weights]
        diff = 100 - sum(w_pct)
        w_pct[0] += diff  # 100%になるよう調整
        
        for i, sym in enumerate(keys_list):
            st.session_state[f"w_{sym}"] = max(0, w_pct[i])
            
        st.session_state["opt_msg"] = f"""
        **✅ 科学的最適化が完了しました！（期間: {selected_period} ｜ 目標: {target_mode}最大化）**
        * **最適配分比率**: TQQQ {st.session_state['w_TQQQ']}% ｜ SPXL {st.session_state['w_SPXL']}% ｜ SOXL {st.session_state['w_SOXL']}% ｜ FAS {st.session_state['w_FAS']}% ｜ UGL {st.session_state['w_UGL']}%
        * 各銘柄のEMA日数・HV閾値も個別最適値へ更新されました。
        """
        st.rerun()

# -------------------------------------------------------------
# 7. サイドバー：手動パラメータ微調整
# -------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ 5銘柄の配分比率（手動調整）")
w_tqqq = st.sidebar.slider("TQQQ 比率 (%)", 0, 100, step=5, key="w_TQQQ")
w_spxl = st.sidebar.slider("SPXL 比率 (%)", 0, 100, step=5, key="w_SPXL")
w_soxl = st.sidebar.slider("SOXL 比率 (%)", 0, 100, step=5, key="w_SOXL")
w_fas  = st.sidebar.slider("FAS 比率 (%)", 0, 100, step=5, key="w_FAS")
w_ugl  = st.sidebar.slider("UGL 比率 (%)", 0, 100, step=5, key="w_UGL")

tot_w = w_tqqq + w_spxl + w_soxl + w_fas + w_ugl
if tot_w != 100:
    st.sidebar.warning(f"⚠️ 合計比率: **{tot_w}%** （合計100%に調整してください）")
    norm_f = 100.0 / tot_w if tot_w > 0 else 1.0
else:
    st.sidebar.success("✅ 合計比率: **100%**")
    norm_f = 1.0

user_weights = {
    "TQQQ": (w_tqqq * norm_f) / 100.0,
    "SPXL": (w_spxl * norm_f) / 100.0,
    "SOXL": (w_soxl * norm_f) / 100.0,
    "FAS":  (w_fas * norm_f) / 100.0,
    "UGL":  (w_ugl * norm_f) / 100.0,
}

with st.sidebar.expander("🛠️ 銘柄別パラメータ個別調整"):
    for sym in ASSETS.keys():
        st.markdown(f"**{sym} ({ASSETS[sym]['underlying']}基準)**")
        st.slider(f"{sym} EMA日数", 100, 250, step=5, key=f"ema_{sym}")
        st.slider(f"{sym} HV閾値(%)", 15.0, 50.0, step=1.0, key=f"hv_{sym}")

st.sidebar.markdown("---")
r2_input = st.sidebar.slider("レジーム2 (警戒時) の保有比率 (%)", 0, 100, step=10, key="regime2_alloc") / 100.0

# -------------------------------------------------------------
# 8. ポートフォリオ統合計算
# -------------------------------------------------------------
p_strat_ret = np.zeros(len(common_idx))
p_bm_ret = np.zeros(len(common_idx))
signals_now = {}

for sym, cfg in ASSETS.items():
    s_ret, p_pos, p_ema, p_hv, u_c, etf_c = eval_single_asset(
        sym, st.session_state[f"ema_{sym}"], st.session_state[f"hv_{sym}"], r2_input
    )
    w = user_weights[sym]
    p_strat_ret += w * s_ret
    
    # ベンチマーク
    etf_arr = market_data[sym].loc[common_idx, "Close"].values
    raw_etf_ret = np.zeros_like(etf_arr)
    raw_etf_ret[1:] = (etf_arr[1:] - etf_arr[:-1]) / etf_arr[:-1]
    p_bm_ret += w * raw_etf_ret
    
    signals_now[sym] = {
        "pos": p_pos, "ema": p_ema, "hv": p_hv, "u_close": u_c, "etf_price": etf_c,
        "underlying": cfg["underlying"], "name": cfg["name"], "type": cfg["type"],
        "ema_param": st.session_state[f"ema_{sym}"], "hv_param": st.session_state[f"hv_{sym}"]
    }

# 統合パフォーマンス指標
cum_strat = np.cumprod(1.0 + p_strat_ret)
cum_bm = np.cumprod(1.0 + p_bm_ret)

n_days = len(common_idx)
yrs = n_days / 252.0

cagr_strat = cum_strat[-1] ** (1.0 / yrs) - 1.0
cagr_bm = cum_bm[-1] ** (1.0 / yrs) - 1.0

vol_strat = np.std(p_strat_ret) * np.sqrt(252)
vol_bm = np.std(p_bm_ret) * np.sqrt(252)

peak_s = np.maximum.accumulate(cum_strat)
dd_s = (cum_strat - peak_s) / peak_s
mdd_strat = np.min(dd_s)

peak_b = np.maximum.accumulate(cum_bm)
mdd_bm = np.min((cum_bm - peak_b) / peak_b)

calmar_strat = cagr_strat / abs(mdd_strat) if mdd_strat != 0 else 0
gains_s = np.sum(p_strat_ret[p_strat_ret > 0])
losses_s = np.abs(np.sum(p_strat_ret[p_strat_ret < 0]))
pf_strat = gains_s / losses_s if losses_s != 0 else 0
t_stat_strat = (np.mean(p_strat_ret) / (np.std(p_strat_ret) / np.sqrt(n_days))) if np.std(p_strat_ret) != 0 else 0
active_days = np.sum(p_strat_ret != 0)
win_rate = (np.sum(p_strat_ret > 0) / active_days * 100) if active_days > 0 else 0

# -------------------------------------------------------------
# 9. メイン画面レイアウト
# -------------------------------------------------------------
st.title("🏛️ 米国マルチレバレッジ 全天候型最適化ダッシュボード")
st.caption(f"検証データ期間: **{selected_period}** （{common_idx[0].strftime('%Y/%m/%d')} 〜 {common_idx[-1].strftime('%Y/%m/%d')} ｜ 計 {n_days} 営業日）")

if st.session_state["opt_msg"]:
    st.success(st.session_state["opt_msg"])

tab1, tab2, tab3 = st.tabs([
    "🏛️ 現在シグナル & 楽天証券発注計算",
    "🧪 統合パフォーマンス検証 & 統計指標",
    "🔬 銘柄別パラメータ & 個別分析"
])

# =============================================================
# TAB 1: 現在シグナル & 発注計算
# =============================================================
with tab1:
    st.subheader("現在の各アセット市場レジーム判定")
    
    cols = st.columns(5)
    tot_effective = 0.0
    for idx_c, sym in enumerate(ASSETS.keys()):
        sig = signals_now[sym]
        w = user_weights[sym]
        tot_effective += w * sig["pos"]
        
        with cols[idx_c]:
            icon = "🟢" if sig["pos"] == 1.0 else ("🟡" if sig["pos"] > 0 else "🔴")
            st.markdown(f"#### {icon} {sym}")
            st.caption(f"{sig['type']} (配分: {int(w*100)}%)")
            st.metric(f"{sig['underlying']} 終値", f"${sig['u_close']:.2f}")
            st.write(f"推奨投資比率: **{int(sig['pos']*100)}%**")
            st.progress(sig["pos"])
            st.caption(f"EMA({sig['ema_param']}): ${sig['ema']:.2f}\nHV20: {sig['hv']:.1f}% (閾値:{sig['hv_param']}%)")

    cash_ratio = 1.0 - tot_effective
    st.markdown("---")
    
    c_s1, c_s2 = st.columns([1.5, 1])
    with c_s1:
        st.markdown("### 💼 ポートフォリオ総合配分")
        m_c1, m_c2 = st.columns(2)
        m_c1.metric("総株式・ゴールド投資比率", f"{tot_effective*100:.1f} %")
        m_c2.metric("安全待機キャッシュ比率 (MMF等)", f"{cash_ratio*100:.1f} %")
        st.info(f"💡 現在のシグナルに基づき、資産の **{tot_effective*100:.1f}%** を市場に配分し、残りの **{cash_ratio*100:.1f}%** を米ドルMMF（年利約4〜5%）に待機させます。")
    with c_s2:
        pie_labels = [s for s in ASSETS.keys()] + ["米ドルMMF / キャッシュ"]
        pie_vals = [user_weights[s] * signals_now[s]["pos"] * 100 for s in ASSETS.keys()] + [cash_ratio * 100]
        fig_pie = go.Figure(data=[go.Pie(
            labels=pie_labels, values=pie_vals, hole=.45,
            marker_colors=['#00ba38', '#619cff', '#f5b041', '#9b59b6', '#f1c40f', '#b0b0b0']
        )])
        fig_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=200)
        st.plotly_chart(fig_pie, use_container_width=True)

    # 楽天証券 寄り付き発注シミュレーター
    st.markdown("---")
    st.subheader("💡 楽天証券 寄り付き発注シミュレーター（日本円・米ドル両対応）")
    
    curr_c1, curr_c2, curr_c3 = st.columns([1.2, 1.5, 1.3])
    with curr_c1:
        in_curr = st.radio("入力通貨単位", ["日本円 (万円)", "米ドル (USD)"], horizontal=True)
    with curr_c2:
        if in_curr == "日本円 (万円)":
            f_jpy_man = st.number_input("運用総資金額（万円）", min_value=10, value=500, step=10)
        else:
            f_usd_in = st.number_input("運用総資金額（米ドル：USD）", min_value=1000, value=30000, step=1000)
    with curr_c3:
        fx_val = st.number_input("適用為替レート (USD/JPY)", value=latest_fx_rate, step=0.5, format="%.2f")

    if in_curr == "日本円 (万円)":
        total_jpy = f_jpy_man * 10000.0
        total_usd = total_jpy / fx_val
    else:
        total_usd = float(f_usd_in)
        total_jpy = total_usd * fx_val

    st.success(f"💰 **運用総資産**: **${total_usd:,.2f}** ＝ **約 {total_jpy:,.0f} 円** （{total_jpy/10000:,.1f} 万円）")

    sim_rows = []
    for sym in ASSETS.keys():
        sig = signals_now[sym]
        w = user_weights[sym]
        t_usd = total_usd * w * sig["pos"]
        t_jpy = t_usd * fx_val
        p = sig["etf_price"]
        sh = int(t_usd // p)
        sim_rows.append({
            "銘柄": sig["name"],
            "資産タイプ": sig["type"],
            "目標金額 (USD)": f"${t_usd:,.2f}",
            "目標金額 (日本円)": f"約 {t_jpy:,.0f} 円 ({t_jpy/10000:,.1f}万)",
            "参考株価": f"${p:.2f}",
            "目標保有株数": f"{sh} 株",
            "今夜のアクション": f"{sh} 株に調整" if t_usd > 0 else "全売却 (0株)"
        })
    c_usd = total_usd * cash_ratio
    c_jpy = c_usd * fx_val
    sim_rows.append({
        "銘柄": "米ドル現金 / MMF",
        "資産タイプ": "安全待機資金",
        "目標金額 (USD)": f"${c_usd:,.2f}",
        "目標金額 (日本円)": f"約 {c_jpy:,.0f} 円 ({c_jpy/10000:,.1f}万)",
        "参考株価": "-",
        "目標保有株数": "-",
        "今夜のアクション": "米ドルMMF等で安全待機 (利息年4〜5%)"
    })
    st.table(pd.DataFrame(sim_rows))

# =============================================================
# TAB 2: パフォーマンス検証 & 統計指標
# =============================================================
with tab2:
    st.subheader(f"🧪 全天候型ポートフォリオ 統計パフォーマンス検証 （{selected_period}）")
    
    # 統計指標カード
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("期待年利 (CAGR)", f"{cagr_strat*100:.1f} %", f"単主持: {cagr_bm*100:.1f}%")
    m2.metric("年率リスク (Vol)", f"{vol_strat*100:.1f} %", f"単主持: {vol_bm*100:.1f}%")
    m3.metric("最大下落率 (MDD)", f"{mdd_strat*100:.1f} %", f"単主持: {mdd_bm*100:.1f}%")
    m4.metric("プロフィットファクター (PF)", f"{pf_strat:.2f}")
    m5.metric("カルマーレシオ", f"{calmar_strat:.2f}")
    m6.metric("t値 (有意性)", f"{t_stat_strat:.2f}")

    st.markdown("---")
    
    # 資産推移 & ドローダウン
    fig_bt = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.7, 0.3])
    fig_bt.add_trace(go.Scatter(x=common_idx, y=cum_strat, name="全天候型レジーム運用", line=dict(color="#00ba38", width=2.5)), row=1, col=1)
    fig_bt.add_trace(go.Scatter(x=common_idx, y=cum_bm, name="5銘柄バイ＆ホールド (放置)", line=dict(color="#888888", width=1.5, dash="dot")), row=1, col=1)
    fig_bt.add_trace(go.Scatter(x=common_idx, y=dd_s * 100, name="ドローダウン (%)", fill="tozeroy", line=dict(color="#d62728", width=1)), row=2, col=1)
    fig_bt.update_layout(height=480, margin=dict(t=20, b=20, l=10, r=10), hovermode="x unified")
    fig_bt.update_yaxes(title_text="資産倍率", row=1, col=1)
    fig_bt.update_yaxes(title_text="下落率 (%)", row=2, col=1)
    st.plotly_chart(fig_bt, use_container_width=True)

# =============================================================
# TAB 3: 銘柄別パラメータ & 個別分析
# =============================================================
with tab3:
    st.subheader("🔬 銘柄別 適用パラメータ & 単体パフォーマンス一覧")
    
    asset_perf_table = []
    for sym in ASSETS.keys():
        s_ret, _, _, _, _, _ = eval_single_asset(sym, st.session_state[f"ema_{sym}"], st.session_state[f"hv_{sym}"], r2_input)
        cum_single = np.cumprod(1.0 + s_ret)
        cagr_s = cum_single[-1] ** (1.0 / yrs) - 1.0
        mdd_s = np.min((cum_single - np.maximum.accumulate(cum_single)) / np.maximum.accumulate(cum_single))
        calm_s = cagr_s / abs(mdd_s) if mdd_s != 0 else 0
        g = np.sum(s_ret[s_ret > 0])
        l = np.abs(np.sum(s_ret[s_ret < 0]))
        pf_s = g / l if l != 0 else 0
        
        asset_perf_table.append({
            "銘柄": sym,
            "母体指数": ASSETS[sym]["underlying"],
            "設定EMA日数": f"{st.session_state[f'ema_{sym}']} 日",
            "設定HV閾値": f"{st.session_state[f'hv_{sym}']:.1f} %",
            "ポートフォリオ配分": f"{int(user_weights[sym]*100)} %",
            "単体CAGR": f"{cagr_s*100:.1f} %",
            "単体MDD": f"{mdd_s*100:.1f} %",
            "単体PF": f"{pf_s:.2f}",
            "単体カルマー": f"{calm_s:.2f}"
        })
    st.table(pd.DataFrame(asset_perf_table))
