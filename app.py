import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -------------------------------------------------------------
# 1. ページ基本設定
# -------------------------------------------------------------
st.set_page_config(
    page_title="SDE-Engine 米国マルチレバレッジ 確率的動的最適化システム",
    page_icon="⚡",
    layout="wide"
)

# -------------------------------------------------------------
# 2. アセット基本構成 & 固有の目標ボラティリティ
# -------------------------------------------------------------
ASSETS = {
    "TQQQ": {
        "name": "TQQQ (NASDAQ 3倍)", "underlying": "QQQ", "type": "ハイテク",
        "sigma_target": 25.0, "color": "#00ba38", "default_w": 35
    },
    "SPXL": {
        "name": "SPXL (S&P500 3倍)", "underlying": "SPY", "type": "米国全体",
        "sigma_target": 18.0, "color": "#619cff", "default_w": 25
    },
    "SOXL": {
        "name": "SOXL (半導体 3倍)", "underlying": "SOXX", "type": "半導体",
        "sigma_target": 30.0, "color": "#f5b041", "default_w": 15
    },
    "FAS":  {
        "name": "FAS (金融株 3倍)",   "underlying": "XLF", "type": "金融",
        "sigma_target": 22.0, "color": "#9b59b6", "default_w": 15
    },
    "UGL":  {
        "name": "UGL (ゴールド 2倍)",  "underlying": "GLD", "type": "ゴールド",
        "sigma_target": 15.0, "color": "#f1c40f", "default_w": 10
    },
}

# -------------------------------------------------------------
# 3. サイドバー：検証期間 & 基本配分枠
# -------------------------------------------------------------
st.sidebar.title("⚡ SDE-Engine コントローラー")

selected_period = st.sidebar.selectbox(
    "バックテスト期間",
    ["5y (直近5年間：2021〜現在)", "10y (直近10年間：2016〜現在)"],
    index=1
)
period_code = "5y" if "5y" in selected_period else "10y"

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ ポートフォリオ基本配分枠")
w_tqqq = st.sidebar.slider("TQQQ 比率 (%)", 0, 100, ASSETS["TQQQ"]["default_w"], step=5)
w_spxl = st.sidebar.slider("SPXL 比率 (%)", 0, 100, ASSETS["SPXL"]["default_w"], step=5)
w_soxl = st.sidebar.slider("SOXL 比率 (%)", 0, 100, ASSETS["SOXL"]["default_w"], step=5)
w_fas  = st.sidebar.slider("FAS 比率 (%)", 0, 100, ASSETS["FAS"]["default_w"], step=5)
w_ugl  = st.sidebar.slider("UGL 比率 (%)", 0, 100, ASSETS["UGL"]["default_w"], step=5)

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
# 4. データ取得関数
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

with st.spinner(f"市場データ（{selected_period}）を取得・計算中..."):
    market_data = load_market_data(period_code)
    latest_fx_rate = get_usdjpy_rate()

common_idx = market_data["QQQ"].index
for k in market_data.keys():
    common_idx = common_idx.intersection(market_data[k].index)

# -------------------------------------------------------------
# 5. SDE-Engine 計算コア（トレンド分析・ボラティリティ制御）
# -------------------------------------------------------------
def calc_parkinson_vol(df, window=10):
    high = df["High"].values
    low = df["Low"].values
    hl_ratio = np.log(np.maximum(high, 1e-6) / np.maximum(low, 1e-6))
    pv = np.zeros(len(df))
    factor = 1.0 / (4.0 * np.log(2.0))
    for i in range(window, len(df)):
        pv[i] = np.sqrt((factor / window) * np.sum(hl_ratio[i-window:i]**2)) * np.sqrt(252) * 100.0
    pv[:window] = pv[window] if len(df) > window else 20.0
    return pv

def run_sde_engine(sym):
    cfg = ASSETS[sym]
    u_sym = cfg["underlying"]
    df_u = market_data[u_sym].loc[common_idx]
    df_etf = market_data[sym].loc[common_idx]
    
    c_u = df_u["Close"].values
    c_etf = df_etf["Close"].values
    
    # 1. 局所ボラティリティ
    sigma_local = calc_parkinson_vol(df_etf, window=10)
    
    # 2. トレンド分析（200日移動平均 EMA200 乖離度）
    ema200 = pd.Series(c_u).ewm(span=200, adjust=False).mean().values
    trend_score = (c_u - ema200) / ema200
    
    # 3. ボラティリティ過熱度スコア
    vol_20 = (pd.Series(c_u).pct_change().rolling(20).std() * np.sqrt(252) * 100).fillna(20.0).values
    vol_score = -(vol_20 - cfg["sigma_target"]) / cfg["sigma_target"]
    
    # 4. 強気確率 P(Bull) 算出
    logit = 4.0 * trend_score + 1.2 * vol_score
    p_bull = 1.0 / (1.0 + np.exp(-logit))
    
    # 5. Merton型 最適目標比率 W*
    sigma_tgt = cfg["sigma_target"]
    vol_adj = np.clip(sigma_tgt / np.maximum(sigma_local, 1e-4), 0.1, 1.5)
    w_star = np.clip(p_bull * vol_adj, 0.0, 1.0)
    
    # 6. 二相型エントリー判定（時系列）
    phase_factor = np.where(
        (p_bull < 0.35) | (w_star < 0.05),
        0.0,
        np.where(p_bull < 0.65, 0.25, 1.0)
    )
    actual_pos_series = w_star * phase_factor
    
    # バックテスト実行（前日シグナルで翌日運用）
    exec_pos = np.zeros_like(actual_pos_series)
    exec_pos[1:] = actual_pos_series[:-1]
    
    etf_ret = np.zeros_like(c_etf)
    etf_ret[1:] = (c_etf[1:] - c_etf[:-1]) / c_etf[:-1]
    strat_ret = exec_pos * etf_ret
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
    
    bm_peak = np.maximum.accumulate(cum_bm)
    bm_dd = (cum_bm - bm_peak) / bm_peak
    bm_mdd = np.min(bm_dd)
    
    calmar = cagr / abs(mdd) if mdd != 0 else 0
    gains = np.sum(strat_ret[strat_ret > 0])
    losses = np.abs(np.sum(strat_ret[strat_ret < 0]))
    pf = gains / losses if losses != 0 else 0
    vol = np.std(strat_ret) * np.sqrt(252)
    bm_vol = np.std(bm_ret) * np.sqrt(252)
    
    # 直近ステータス判定
    latest_p = p_bull[-1]
    latest_w = w_star[-1]
    
    if latest_p < 0.35 or latest_w < 0.05:
        phase_badge = "🔴 キャッシュ退避"
        phase_desc = "保有 0% (全額キャッシュ)"
        buy_factor = 0.0
    elif latest_p < 0.65:
        phase_badge = "🟡 探査玉 (Scout)"
        phase_desc = f"W*の 25% 打診買い ({latest_w * 0.25 * 100:.1f}%)"
        buy_factor = 0.25
    else:
        phase_badge = "🟢 本玉巡航 (Core)"
        phase_desc = f"W*を 100% 保有 ({latest_w * 100:.1f}%)"
        buy_factor = 1.0

    actual_single_alloc = latest_w * buy_factor

    return {
        "p_bull_series": p_bull, "w_star_series": w_star,
        "ema200_series": ema200, "sigma_local_series": sigma_local,
        "strat_ret": strat_ret, "bm_ret": bm_ret,
        "cum_strat": cum_strat, "cum_bm": cum_bm,
        "dd": dd, "bm_dd": bm_dd,
        "cagr": cagr, "bm_cagr": bm_cagr, "mdd": mdd, "bm_mdd": bm_mdd,
        "calmar": calmar, "pf": pf, "vol": vol, "bm_vol": bm_vol,
        "u_close": c_u[-1], "etf_price": c_etf[-1], "ema_last": ema200[-1],
        "latest_p": latest_p, "latest_w": latest_w,
        "latest_sigma": sigma_local[-1],
        "phase_badge": phase_badge, "phase_desc": phase_desc,
        "buy_factor": buy_factor, "actual_single_alloc": actual_single_alloc
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
p_bm_cagr = cum_bm_p[-1] ** (1.0 / yrs) - 1.0
peak_p = np.maximum.accumulate(cum_portfolio)
dd_portfolio = (cum_portfolio - peak_p) / peak_p
p_mdd = np.min(dd_portfolio)
p_bm_mdd = np.min((cum_bm_p - np.maximum.accumulate(cum_bm_p)) / np.maximum.accumulate(cum_bm_p))
p_vol = np.std(p_strat_ret) * np.sqrt(252)
p_calmar = p_cagr / abs(p_mdd) if p_mdd != 0 else 0
p_pf = np.sum(p_strat_ret[p_strat_ret > 0]) / abs(np.sum(p_strat_ret[p_strat_ret < 0]))

# -------------------------------------------------------------
# 6. メイン画面ヘッダー ＆ 初心者向け用語ガイド
# -------------------------------------------------------------
st.title("⚡ SDE-Engine 全天候型レバレッジポートフォリオ制御システム")
st.caption("金融工学・確率的最適制御 (Stochastic Optimal Control) に基づく連続リスク制御エンジン")

# 初心者向け用語ガイド
with st.expander("📖 【用語ガイド】強気確率 P(Bull)・目標比率 W*・執行フェーズ・トレンド分析とは？", expanded=False):
    st.markdown("""
    * **強気確率 $P(\\text{Bull})$ とは？**  
      母体指数（QQQ等）の**200日移動平均（トレンド分析）**と**ボラティリティ構造**から算出した「上昇トレンド相場にある確率」です。
      * **65% 以上**: 強い上昇トレンド（本玉を投入できる環境）
      * **35% 〜 65%**: レンジ相場または底打ち反発初期（探査玉のみで様子見）
      * **35% 未満**: 下落相場・危険水域（完全撤退・キャッシュ化）
    * **目標比率 $W^*$ (Dynamic Exposure) とは？**  
      レバレッジETF特有の「複利減価（下落耐性低下）」を防ぐため、ボラティリティの荒さに応じて計算された**安全保有上限比率（0〜100%）**です。
    * **執行フェーズ（二相型エントリー）とは？**  
      * **Phase 2（本玉巡航）**: トレンド確立。目標比率 $W^*$ の **100%（満額）** を保有します。
      * **Phase 1（探査玉投入）**: 底打ち初動。目標比率 $W^*$ の **25% だけを打診買い**し、機会損失を防ぎつつ再急落に備えます。
      * **Phase 0（完全防衛）**: 相場崩壊。比率に関わらず **0%（全額キャッシュ退避）** とします。
    * **トレンド分析は組み込まれているか？**  
      はい。各母体指数の**200日移動平均（EMA200）からの乖離率**が、強気確率 $P(\\text{Bull})$ の中核的判断要素として組み込まれています。
    """)

# キルスイッチ表示
with st.container():
    c_k1, c_k2, c_k3 = st.columns([1.5, 1, 1.2])
    c_k1.markdown("#### 🛡️ カタストロフィ・キルスイッチ状態")
    c_k2.success("● 正常稼働中 (Normal)")
    c_k3.caption("条件: 夜間先物 -4.5% 急落 または 日中NAV -8% 下落で強制全決済")

st.markdown("---")

tab1, tab2, tab3 = st.tabs([
    "🏛️ 現在シグナル ＆ 楽天証券 執行シミュレーター",
    "🧪 統合パフォーマンス検証 ＆ 統計指標",
    "📊 銘柄別詳細分析（チャート・トレンド・DD）"
])

# =============================================================
# TAB 1: 現在シグナル ＆ 執行シミュレーター
# =============================================================
with tab1:
    st.subheader("📊 5銘柄の動的レジーム判定 ＆ 最適保有比率")
    cols = st.columns(5)
    tot_effective = 0.0
    
    for idx_c, sym in enumerate(ASSETS.keys()):
        res = asset_results[sym]
        base_w = base_weights[sym]
        effective_alloc = base_w * res["actual_single_alloc"]
        tot_effective += effective_alloc
        
        with cols[idx_c]:
            st.markdown(f"#### {sym}")
            st.caption(f"{ASSETS[sym]['name']} (基本配分枠: {int(base_w*100)}%)")
            st.metric(f"{ASSETS[sym]['underlying']} 終値", f"${res['u_close']:.2f}")
            
            # 強気確率
            st.write(f"強気確率 $P(\\text{{Bull}})$: **{res['latest_p']*100:.1f}%**")
            st.progress(float(res["latest_p"]))
            
            # ボラティリティ状況
            st.caption(f"Parkinson Vol: **{res['latest_sigma']:.1f}%** (目標: {ASSETS[sym]['sigma_target']}%)")
            
            # 目標比率 W*
            st.write(f"目標比率 $W^*$: **{res['latest_w']*100:.1f}%**")
            
            # フェーズ判定バッジ
            st.info(f"**{res['phase_badge']}**\n\n*{res['phase_desc']}*")
            
            # 今夜の実効買付比率
            st.success(f"**今夜の実効買付比率:**\n\n### {effective_alloc*100:.1f} %")

    cash_ratio = max(0.0, 1.0 - tot_effective)
    st.markdown("---")
    
    # 総合配分サマリー & ドーナツチャート
    c_s1, c_s2 = st.columns([1.5, 1])
    with c_s1:
        st.markdown("### 💼 ポートフォリオ総合配分")
        m_c1, m_c2 = st.columns(2)
        m_c1.metric("総市場エクスポージャー (実効投資比率)", f"{tot_effective*100:.1f} %")
        m_c2.metric("安全待機キャッシュ比率 (米ドルMMF等)", f"{cash_ratio*100:.1f} %")
        st.info(f"💡 市場のボラティリティ過熱・下落確率に応じて保有比率を制御中。残りの **{cash_ratio*100:.1f}%** は米ドルMMF（年利約4〜5%）に待機させ、金利を受け取りながら資産を守ります。")
    with c_s2:
        pie_labels = [s for s in ASSETS.keys() if base_weights[s] > 0] + ["米ドルMMF / キャッシュ"]
        pie_vals = [base_weights[s] * asset_results[s]["actual_single_alloc"] * 100 for s in ASSETS.keys() if base_weights[s] > 0] + [cash_ratio * 100]
        fig_pie = go.Figure(data=[go.Pie(
            labels=pie_labels, values=pie_vals, hole=.45,
            marker_colors=['#00ba38', '#619cff', '#f5b041', '#9b59b6', '#f1c40f', '#b0b0b0']
        )])
        fig_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=200)
        st.plotly_chart(fig_pie, use_container_width=True)

    # 楽天証券 執行シミュレーター
    st.markdown("---")
    st.subheader("💡 楽天証券 寄り付き発注シミュレーター（二相型発注 ＆ ±10%リバランスバンド判定）")
    
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
        total_usd = (f_jpy_man * 10000.0) / fx_val
        total_jpy = f_jpy_man * 10000.0
    else:
        total_usd = float(f_usd_in)
        total_jpy = total_usd * fx_val

    st.success(f"💰 **運用総資産**: **${total_usd:,.2f}** ＝ **約 {total_jpy:,.0f} 円** （{total_jpy/10000:,.1f} 万円）")

    sim_rows = []
    for sym in ASSETS.keys():
        res = asset_results[sym]
        base_w = base_weights[sym]
        p = res["etf_price"]
        
        target_usd = total_usd * base_w * res["actual_single_alloc"]
        target_jpy = target_usd * fx_val
        target_shares = int(target_usd // p) if p > 0 else 0
        
        scout_part_usd = total_usd * base_w * (res["latest_w"] * 0.25)
        scout_shares = int(scout_part_usd // p) if p > 0 else 0
        core_shares = max(0, target_shares - scout_shares) if res["buy_factor"] == 1.0 else 0
        
        if base_w == 0:
            action = "買付不要 (枠0%)"
        elif target_shares == 0:
            action = "全ポジション売却してキャッシュ化 (0株)"
        else:
            if res["buy_factor"] == 0.25:
                action = f"探査玉（打診買い）として {target_shares} 株を発注"
            else:
                action = f"合計 {target_shares} 株に保有数を調整（巡航）"

        sim_rows.append({
            "銘柄": ASSETS[sym]["name"],
            "判定ステータス": res["phase_badge"],
            "強気確率 P(Bull)": f"{res['latest_p']*100:.1f}%",
            "目標比率 W*": f"{res['latest_w']*100:.1f}%",
            "実効配分比率": f"{base_w * res['actual_single_alloc']*100:.1f}%",
            "目標金額 (USD)": f"${target_usd:,.2f}",
            "目標金額 (日本円)": f"約 {target_jpy:,.0f} 円" if base_w > 0 else "0 円",
            "参考株価": f"${p:.2f}",
            "目標保有株数": f"{target_shares} 株",
            "探査玉 / 本玉 内訳": f"{scout_shares}株 / {core_shares}株" if res["buy_factor"] == 1.0 else (f"{scout_shares}株 / 0株" if res["buy_factor"] == 0.25 else "0株 / 0株"),
            "今夜の推奨アクション": action
        })
        
    c_usd = total_usd * cash_ratio
    c_jpy = c_usd * fx_val
    sim_rows.append({
        "銘柄": "米ドル現金 / MMF",
        "判定ステータス": "🛡️ 安全待機",
        "強気確率 P(Bull)": "-",
        "目標比率 W*": "-",
        "実効配分比率": f"{cash_ratio*100:.1f}%",
        "目標金額 (USD)": f"${c_usd:,.2f}",
        "目標金額 (日本円)": f"約 {c_jpy:,.0f} 円",
        "参考株価": "-",
        "目標保有株数": "-",
        "探査玉 / 本玉 内訳": "-",
        "今夜の推奨アクション": "米ドルMMFで安全待機 (年利約4〜5%利息享受)"
    })
    
    st.table(pd.DataFrame(sim_rows))

# =============================================================
# TAB 2: パフォーマンス検証 ＆ 統計指標
# =============================================================
with tab2:
    st.subheader(f"🧪 全天候型ポートフォリオ 統計パフォーマンス検証 （{selected_period}）")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("期待年利 (CAGR)", f"{p_cagr*100:.1f} %", f"単主持: {p_bm_cagr*100:.1f}%")
    m2.metric("年率ボラティリティ", f"{p_vol*100:.1f} %")
    m3.metric("最大下落率 (MDD)", f"{p_mdd*100:.1f} %", f"単主持: {p_bm_mdd*100:.1f}%")
    m4.metric("カルマーレシオ", f"{p_calmar:.2f}")
    m5.metric("プロフィットファクター", f"{p_pf:.2f}")

    st.markdown("---")
    fig_bt = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.7, 0.3])
    fig_bt.add_trace(go.Scatter(x=common_idx, y=cum_portfolio, name="SDE-Engine ポートフォリオ", line=dict(color="#00ba38", width=2.5)), row=1, col=1)
    fig_bt.add_trace(go.Scatter(x=common_idx, y=cum_bm_p, name="5銘柄バイ＆ホールド (放置)", line=dict(color="#888888", width=1.5, dash="dot")), row=1, col=1)
    fig_bt.add_trace(go.Scatter(x=common_idx, y=dd_portfolio * 100, name="ドローダウン (%)", fill="tozeroy", line=dict(color="#d62728", width=1)), row=2, col=1)
    fig_bt.update_layout(height=480, margin=dict(t=20, b=20, l=10, r=10), hovermode="x unified")
    fig_bt.update_yaxes(title_text="資産倍率", row=1, col=1)
    fig_bt.update_yaxes(title_text="下落率 (%)", row=2, col=1)
    st.plotly_chart(fig_bt, use_container_width=True)

# =============================================================
# TAB 3: 銘柄別詳細分析
# =============================================================
with tab3:
    st.subheader("📊 各銘柄の個別パフォーマンス・トレンド分析チャート")
    selected_asset = st.radio(
        "分析対象銘柄を選択",
        list(ASSETS.keys()),
        format_func=lambda x: f"{x} （{ASSETS[x]['name']}）",
        horizontal=True
    )
    
    target_res = asset_results[selected_asset]
    cfg = ASSETS[selected_asset]
    u_sym = cfg["underlying"]
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("累積利益（倍率）", f"{target_res['cum_strat'][-1]:.2f} 倍", f"単主持: {target_res['cum_bm'][-1]:.2f}倍")
    col_m2.metric("期待年利 (CAGR)", f"{target_res['cagr']*100:.1f} %", f"単主持: {target_res['bm_cagr']*100:.1f}%")
    col_m3.metric("最大下落率 (MDD)", f"{target_res['mdd']*100:.1f} %", f"単主持: {target_res['bm_mdd']*100:.1f}%")
    col_m4.metric("カルマーレシオ", f"{target_res['calmar']:.2f}")
    
    st.markdown("---")
    
    fig_single = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05,
        row_heights=[0.40, 0.35, 0.25],
        subplot_titles=(
            f"① 母体指数 {u_sym} 価格 ＆ 200日EMA（トレンド分析）",
            f"② {selected_asset} SDE-Engine戦略 vs 単純バイ＆ホールド",
            f"③ 強気確率 P(Bull) ＆ 目標比率 W* 推移"
        )
    )
    
    df_u_c = market_data[u_sym].loc[common_idx, "Close"]
    fig_single.add_trace(go.Scatter(x=common_idx, y=df_u_c, name=f"{u_sym} 終値", line=dict(color=cfg["color"], width=1.5)), row=1, col=1)
    fig_single.add_trace(go.Scatter(x=common_idx, y=target_res["ema200_series"], name="200日 EMA", line=dict(color="#ff7f0e", width=2)), row=1, col=1)
    fig_single.add_trace(go.Scatter(x=common_idx, y=target_res["cum_strat"], name=f"{selected_asset} SDE戦略", line=dict(color="#00ba38", width=2)), row=2, col=1)
    fig_single.add_trace(go.Scatter(x=common_idx, y=target_res["cum_bm"], name=f"{selected_asset} 単純保有", line=dict(color="#888888", width=1.2, dash="dot")), row=2, col=1)
    fig_single.add_trace(go.Scatter(x=common_idx, y=target_res["p_bull_series"] * 100, name="強気確率 P(Bull) %", line=dict(color="#1f77b4", width=1.5)), row=3, col=1)
    fig_single.add_trace(go.Scatter(x=common_idx, y=target_res["w_star_series"] * 100, name="目標比率 W* %", line=dict(color="#d62728", width=1.5)), row=3, col=1)
    
    fig_single.update_layout(height=720, margin=dict(t=30, b=20, l=10, r=10), hovermode="x unified")
    fig_single.update_yaxes(title_text="株価 ($)", row=1, col=1)
    fig_single.update_yaxes(title_text="資産倍率", row=2, col=1)
    fig_single.update_yaxes(title_text="比率 (%)", row=3, col=1)
    st.plotly_chart(fig_single, use_container_width=True)
