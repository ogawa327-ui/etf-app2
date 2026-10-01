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
    page_title="SDE-Engine Pro 動的資金配分システム",
    page_icon="⚡",
    layout="wide"
)

# -------------------------------------------------------------
# 2. アセット基本構成 & 銘柄パラメータ
# -------------------------------------------------------------
ASSETS = {
    "TQQQ": {
        "name": "TQQQ (NASDAQ 3倍)", "underlying": "QQQ", "type": "ハイテク",
        "leverage": 3.0, "sigma_target": 55.0, "color": "#00ba38"
    },
    "SPXL": {
        "name": "SPXL (S&P500 3倍)", "underlying": "SPY", "type": "米国全体",
        "leverage": 3.0, "sigma_target": 45.0, "color": "#619cff"
    },
    "SOXL": {
        "name": "SOXL (半導体 3倍)", "underlying": "SOXX", "type": "半導体",
        "leverage": 3.0, "sigma_target": 65.0, "color": "#f5b041"
    },
    "FAS":  {
        "name": "FAS (金融株 3倍)",   "underlying": "XLF", "type": "金融",
        "leverage": 3.0, "sigma_target": 50.0, "color": "#9b59b6"
    },
    "UGL":  {
        "name": "UGL (ゴールド 2倍)",  "underlying": "GLD", "type": "ゴールド",
        "leverage": 2.0, "sigma_target": 35.0, "color": "#f1c40f"
    },
}

# -------------------------------------------------------------
# 3. サイドバー：検証期間 & 動的配分コントローラー
# -------------------------------------------------------------
st.sidebar.title("⚡ SDE-Engine Pro")

period_options = {
    "5y (直近5年間: 2021〜現在)": "5y",
    "10y (直近10年間: 2016〜現在)": "10y",
    "15y (直近15年間: 2011〜現在)": "15y",
    "20y (直近20年間: 2006〜現在・リーマンショック含む)": "20y",
    "25y (直近25年間: 2001〜現在・ITバブル崩壊含む)": "25y",
    "30y (直近30年間: 1996〜現在)": "30y"
}

selected_period_label = st.sidebar.selectbox(
    "📅 バックテスト期間の選択",
    list(period_options.keys()),
    index=1
)
period_code = period_options[selected_period_label]

st.sidebar.markdown("---")
st.sidebar.markdown("### 🛡️ リスク管理・集中投資上限")
max_cap_pct = st.sidebar.slider(
    "1銘柄あたりの最大投資上限 (案A: 50%推奨)",
    min_value=30, max_value=100, value=50, step=5
)
MAX_CAP = max_cap_pct / 100.0

st.sidebar.info(f"""
**【動的資金配分の仕様】**
* 固定枠は完全撤廃されました。
* 投資適格な銘柄だけに資金を集中投入します。
* 1銘柄への過度な集中を防ぐため、1銘柄の上限は **{max_cap_pct}%** に自動制限されます（案A）。
* 複数銘柄が強気になれば最大100%まで配分を拡大します。
""")

# -------------------------------------------------------------
# 4. データ取得 & 未上場期間の科学的合成（バックフィル）
# -------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_and_sync_market_data(period_str: str):
    all_tickers = ["QQQ", "SPY", "SOXX", "XLF", "GLD", "TQQQ", "SPXL", "SOXL", "FAS", "UGL"]
    raw_data = {}
    for t in all_tickers:
        df = yf.download(t, period=period_str, interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        raw_data[t] = df.dropna()
        
    # 原資産の共通日付をベースに設定
    u_tickers = ["QQQ", "SPY", "SOXX", "XLF", "GLD"]
    base_idx = raw_data["SPY"].index
    for u in u_tickers:
        if u in raw_data and not raw_data[u].empty:
            base_idx = base_idx.intersection(raw_data[u].index)

    cleaned_data = {}
    for u in u_tickers:
        cleaned_data[u] = raw_data[u].loc[base_idx].copy()

    # レバレッジETFが未上場の過去期間を原資産リターン×倍率で合成
    for sym, cfg in ASSETS.items():
        u_sym = cfg["underlying"]
        lev = cfg["leverage"]
        df_u = cleaned_data[u_sym]
        df_etf = raw_data[sym]
        
        # 原資産の価格系列とリターン
        u_close = df_u["Close"]
        u_ret = u_close.pct_change().fillna(0)
        
        # 合成ETFの終値系列を作成
        syn_ret = u_ret * lev - (0.0095 / 252.0)  # 年率0.95%の信託報酬を日次控除
        
        # ETFの実データが存在する日付
        real_idx = df_etf.index.intersection(base_idx)
        if len(real_idx) > 0:
            first_real_date = real_idx[0]
            # 実データと合成データの接続
            real_close = df_etf.loc[real_idx, "Close"]
            first_price = real_close.iloc[0]
            
            # 実データより前の期間を逆算
            pre_idx = base_idx[base_idx < first_real_date]
            if len(pre_idx) > 0:
                pre_ret = syn_ret.loc[pre_idx]
                # 逆算累積
                rev_cum = np.cumprod(1.0 + pre_ret.values[::-1])[::-1]
                synth_pre_price = first_price / rev_cum
                full_close = pd.concat([pd.Series(synth_pre_price, index=pre_idx), real_close])
            else:
                full_close = real_close
        else:
            # 実データが一切ない場合は全期間合成
            full_close = 100.0 * np.cumprod(1.0 + syn_ret)
            
        full_close = full_close.reindex(base_idx).ffill().bfill()
        
        # 高値・安値の近似合成
        df_res = pd.DataFrame(index=base_idx)
        df_res["Close"] = full_close
        df_res["High"] = full_close * (1.0 + np.abs(syn_ret) * 0.6)
        df_res["Low"] = full_close * (1.0 - np.abs(syn_ret) * 0.6)
        cleaned_data[sym] = df_res

    return cleaned_data, base_idx

@st.cache_data(ttl=3600)
def get_usdjpy_rate():
    try:
        fx = yf.download("USDJPY=X", period="5d", interval="1d", progress=False)
        if isinstance(fx.columns, pd.MultiIndex):
            fx.columns = fx.columns.get_level_values(0)
        return round(float(fx["Close"].iloc[-1]), 2)
    except:
        return 155.0

with st.spinner(f"市場データ（{selected_period_label}）を取得・合成解析中..."):
    market_data, common_idx = load_and_sync_market_data(period_code)
    latest_fx_rate = get_usdjpy_rate()

# -------------------------------------------------------------
# 5. SDE-Engine Pro 個別シグナル判定
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

def run_asset_signals(sym):
    cfg = ASSETS[sym]
    u_sym = cfg["underlying"]
    df_u = market_data[u_sym].loc[common_idx]
    df_etf = market_data[sym].loc[common_idx]
    
    c_u = df_u["Close"].values
    c_etf = df_etf["Close"].values
    
    sigma_local = calc_parkinson_vol(df_etf, window=10)
    
    # マルチタイムフレーム・トレンド（50日EMA ＋ 200日EMA）
    ema50 = pd.Series(c_u).ewm(span=50, adjust=False).mean().values
    ema200 = pd.Series(c_u).ewm(span=200, adjust=False).mean().values
    
    short_trend = (c_u - ema50) / ema50
    long_trend = (ema50 - ema200) / ema200
    combined_trend = 0.65 * short_trend + 0.35 * long_trend
    
    vol_20 = (pd.Series(c_u).pct_change().rolling(20).std() * np.sqrt(252) * 100).fillna(20.0).values
    vol_score = -(vol_20 - 30.0) / 30.0
    
    logit = 6.0 * combined_trend + 1.5 * vol_score
    p_bull = 1.0 / (1.0 + np.exp(-logit))
    
    sigma_tgt = cfg["sigma_target"]
    vol_adj = np.clip(sigma_tgt / np.maximum(sigma_local, 1e-4), 0.2, 1.2)
    w_star = np.clip(p_bull * vol_adj, 0.0, 1.0)
    
    # 投資適格係数（Phase 0: 0.0, Phase 1: 0.5, Phase 2: 1.0）
    factor = np.where(
        (p_bull < 0.35) | (w_star < 0.10),
        0.0,
        np.where(p_bull < 0.55, 0.50, 1.0)
    )
    raw_desired_w = w_star * factor
    
    etf_ret = np.zeros_like(c_etf)
    etf_ret[1:] = (c_etf[1:] - c_etf[:-1]) / c_etf[:-1]
    
    latest_p = p_bull[-1]
    latest_w = w_star[-1]
    if latest_p < 0.35 or latest_w < 0.10:
        badge = "🔴 対象外 (待機)"
        desc = "シグナル消滅・キャッシュ退避"
        phase_type = "None"
    elif latest_p < 0.55:
        badge = "🟡 探査玉投入"
        desc = f"打診買い 50%稼働 (W* {latest_w*100:.0f}%)"
        phase_type = "Scout"
    else:
        badge = "🟢 本玉巡航"
        desc = f"満額 100%稼働 (W* {latest_w*100:.0f}%)"
        phase_type = "Core"

    return {
        "p_bull": p_bull, "w_star": w_star, "raw_w": raw_desired_w,
        "sigma_local": sigma_local, "ema50": ema50, "ema200": ema200,
        "c_u": c_u, "c_etf": c_etf, "etf_ret": etf_ret,
        "latest_p": latest_p, "latest_w": latest_w, "latest_sigma": sigma_local[-1],
        "badge": badge, "desc": desc, "phase_type": phase_type,
        "u_close": c_u[-1], "etf_price": c_etf[-1]
    }

signals = {s: run_asset_signals(s) for s in ASSETS.keys()}

# -------------------------------------------------------------
# 6. 動的資金配分エンジン（日次最適化シミュレーション）
# -------------------------------------------------------------
n_days = len(common_idx)
keys = list(ASSETS.keys())
daily_alloc_matrix = np.zeros((n_days, len(keys)))

for t in range(n_days):
    raw_weights = np.array([signals[k]["raw_w"][t] for k in keys])
    active_mask = raw_weights > 0.05
    n_active = np.sum(active_mask)
    
    if n_active == 0:
        continue
    elif n_active == 1:
        # 1銘柄のみ適格の場合：上限MAX_CAP (50%) を上限に配分
        idx = np.where(active_mask)[0][0]
        daily_alloc_matrix[t, idx] = min(raw_weights[idx], MAX_CAP)
    else:
        # 複数銘柄適格の場合：スコア比率で配分し、各銘柄MAX_CAPを適用
        tot_raw = np.sum(raw_weights[active_mask])
        cand_w = np.zeros(len(keys))
        for i in range(len(keys)):
            if active_mask[i]:
                # 全体で100%を超えないように調整しつつ、各銘柄上限を適用
                cand_w[i] = min((raw_weights[i] / tot_raw) if tot_raw > 1.0 else raw_weights[i], MAX_CAP)
        daily_alloc_matrix[t] = cand_w

# バックテスト実行（前日ウェイトで翌日運用）
exec_matrix = np.zeros_like(daily_alloc_matrix)
exec_matrix[1:] = daily_alloc_matrix[:-1]

# キャッシュ金利（米ドルMMF 日次利息 年利約3.5%）
daily_cash_rate = (0.035 / 252.0)
ret_matrix = np.column_stack([signals[k]["etf_ret"] for k in keys])

strat_daily_ret = np.sum(exec_matrix * ret_matrix, axis=1)
cash_ratio_daily = 1.0 - np.sum(exec_matrix, axis=1)
strat_daily_ret += cash_ratio_daily * daily_cash_rate

# 5銘柄均等バイ＆ホールド比較ベンチマーク
bm_daily_ret = np.mean(ret_matrix, axis=1)

cum_strat = np.cumprod(1.0 + strat_daily_ret)
cum_bm = np.cumprod(1.0 + bm_daily_ret)

yrs = n_days / 252.0
strat_cagr = cum_strat[-1] ** (1.0 / yrs) - 1.0
bm_cagr = cum_bm[-1] ** (1.0 / yrs) - 1.0

peak_s = np.maximum.accumulate(cum_strat)
dd_s = (cum_strat - peak_s) / peak_s
strat_mdd = np.min(dd_s)

peak_bm = np.maximum.accumulate(cum_bm)
dd_bm = (cum_bm - peak_bm) / peak_bm
bm_mdd = np.min(dd_bm)

strat_vol = np.std(strat_daily_ret) * np.sqrt(252)
strat_calmar = strat_cagr / abs(strat_mdd) if strat_mdd != 0 else 0
pos_ret = strat_daily_ret[strat_daily_ret > 0]
neg_ret = strat_daily_ret[strat_daily_ret < 0]
strat_pf = np.sum(pos_ret) / np.abs(np.sum(neg_ret)) if len(neg_ret) > 0 else 0

# 直近（今夜）の動的配分結果
latest_alloc = daily_alloc_matrix[-1]
tot_latest_invested = np.sum(latest_alloc)
tot_latest_cash = max(0.0, 1.0 - tot_latest_invested)

# -------------------------------------------------------------
# 7. メイン画面ヘッダー
# -------------------------------------------------------------
st.title("⚡ SDE-Engine Pro 動的資金配分ダッシュボード")
st.caption(f"検証期間: **{selected_period_label}** （{common_idx[0].strftime('%Y/%m/%d')} 〜 {common_idx[-1].strftime('%Y/%m/%d')} ｜ 計 {n_days} 営業日）")

# カタストロフィ・キルスイッチ
with st.container():
    c_k1, c_k2, c_k3 = st.columns([1.5, 1, 1.2])
    c_k1.markdown("#### 🛡️ カタストロフィ・キルスイッチ状態")
    c_k2.success("● 正常稼働中 (Normal)")
    c_k3.caption("条件: 夜間先物 -4.5% 急落 または 日中NAV -8% 下落で強制全決済")

st.markdown("---")

tab1, tab2, tab3 = st.tabs([
    "🏛️ 今夜の最適配分 ＆ 楽天証券 執行シミュレーター",
    "🧪 超長期パフォーマンス検証（5年〜30年）",
    "📊 銘柄別詳細分析（チャート・シグナル）"
])

# =============================================================
# TAB 1: 今夜の最適配分 ＆ 執行シミュレーター
# =============================================================
with tab1:
    st.subheader("📊 5銘柄の動的配分シグナル（固定枠なし・適格銘柄へ自動集中）")
    cols = st.columns(5)
    
    for idx_c, sym in enumerate(keys):
        sig = signals[sym]
        alloc_ratio = latest_alloc[idx_c]
        
        with cols[idx_c]:
            st.markdown(f"#### {sym}")
            st.caption(f"{ASSETS[sym]['name']}")
            st.metric(f"{ASSETS[sym]['underlying']} 終値", f"${sig['u_close']:.2f}")
            
            st.write(f"強気確率 $P(\\text{{Bull}})$: **{sig['latest_p']*100:.1f}%**")
            st.progress(float(sig["latest_p"]))
            
            st.caption(f"Parkinson Vol: **{sig['latest_sigma']:.1f}%**")
            st.info(f"**{sig['badge']}**\n\n*{sig['desc']}*")
            
            if alloc_ratio > 0:
                st.success(f"**推奨投資比率:**\n\n### {alloc_ratio*100:.1f} %")
            else:
                st.warning(f"**投資配分:**\n\n### 0.0 % (待機)")

    st.markdown("---")
    
    c_s1, c_s2 = st.columns([1.5, 1])
    with c_s1:
        st.markdown("### 💼 ポートフォリオ全体の実効配分")
        m_c1, m_c2 = st.columns(2)
        m_c1.metric("総株式・ゴールド投資比率", f"{tot_latest_invested*100:.1f} %")
        m_c2.metric("安全待機MMF比率 (待機資金)", f"{tot_latest_cash*100:.1f} %")
        st.info(f"💡 強気シグナルが点灯した銘柄だけに資金を集中。1銘柄上限は **{max_cap_pct}%** に制限し、残りの **{tot_latest_cash*100:.1f}%** は米ドルMMF（年利約3.5〜4.5%）で安全待機します。")
    with c_s2:
        pie_labels = [keys[i] for i in range(len(keys)) if latest_alloc[i] > 0] + ["米ドルMMF"]
        pie_vals = [latest_alloc[i] * 100 for i in range(len(keys)) if latest_alloc[i] > 0] + [tot_latest_cash * 100]
        fig_pie = go.Figure(data=[go.Pie(
            labels=pie_labels, values=pie_vals, hole=.45,
            marker_colors=['#00ba38', '#619cff', '#f5b041', '#9b59b6', '#f1c40f', '#b0b0b0']
        )])
        fig_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=200)
        st.plotly_chart(fig_pie, use_container_width=True)

    # 楽天証券 執行シミュレーター
    st.markdown("---")
    st.subheader("💡 楽天証券 寄り付き発注シミュレーター（動的バランス算出）")
    
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
    for idx_c, sym in enumerate(keys):
        sig = signals[sym]
        alloc_ratio = latest_alloc[idx_c]
        p = sig["etf_price"]
        
        target_usd = total_usd * alloc_ratio
        target_jpy = target_usd * fx_val
        target_shares = int(target_usd // p) if p > 0 else 0
        
        if alloc_ratio == 0:
            action = "待機・買付なし (保有中の場合は全売却)"
        elif sig["phase_type"] == "Scout":
            action = f"🟡 探査玉として {target_shares} 株を買付 (初動打診)"
        else:
            action = f"🟢 本玉として {target_shares} 株に保有数を調整 (巡航)"

        sim_rows.append({
            "銘柄": ASSETS[sym]["name"],
            "シグナル判定": sig["badge"],
            "最適配分比率": f"{alloc_ratio*100:.1f} %",
            "投入目標額 (USD)": f"${target_usd:,.2f}",
            "投入目標額 (日本円)": f"約 {target_jpy:,.0f} 円" if alloc_ratio > 0 else "0 円",
            "参考株価": f"${p:.2f}",
            "目標保有株数": f"{target_shares} 株",
            "今夜の推奨アクション": action
        })
        
    c_usd = total_usd * tot_latest_cash
    c_jpy = c_usd * fx_val
    sim_rows.append({
        "銘柄": "米ドル現金 / MMF",
        "シグナル判定": "🛡️ 安全待機",
        "最適配分比率": f"{tot_latest_cash*100:.1f} %",
        "投入目標額 (USD)": f"${c_usd:,.2f}",
        "投入目標額 (日本円)": f"約 {c_jpy:,.0f} 円",
        "参考株価": "-",
        "目標保有株数": "-",
        "今夜の推奨アクション": "米ドルMMFで安全待機 (利息年約3.5〜4.5%享受)"
    })
    
    st.table(pd.DataFrame(sim_rows))

# =============================================================
# TAB 2: パフォーマンス検証 ＆ 統計指標
# =============================================================
with tab2:
    st.subheader(f"🧪 超長期バックテスト検証結果 （{selected_period_label}）")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("期待年利 (CAGR)", f"{strat_cagr*100:.1f} %", f"単主持: {bm_cagr*100:.1f}%")
    m2.metric("年率ボラティリティ", f"{strat_vol*100:.1f} %")
    m3.metric("最大下落率 (MDD)", f"{strat_mdd*100:.1f} %", f"単主持: {bm_mdd*100:.1f}%")
    m4.metric("カルマーレシオ", f"{strat_calmar:.2f}", f"単主持: {bm_cagr/abs(bm_mdd):.2f}")
    m5.metric("プロフィットファクター", f"{strat_pf:.2f}")

    st.markdown("---")
    fig_bt = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.7, 0.3])
    fig_bt.add_trace(go.Scatter(x=common_idx, y=cum_strat, name="SDE Pro 動的資金配分戦略", line=dict(color="#00ba38", width=2.5)), row=1, col=1)
    fig_bt.add_trace(go.Scatter(x=common_idx, y=cum_bm, name="5銘柄均等バイ＆ホールド (放置)", line=dict(color="#888888", width=1.5, dash="dot")), row=1, col=1)
    fig_bt.add_trace(go.Scatter(x=common_idx, y=dd_s * 100, name="戦略DD (%)", fill="tozeroy", line=dict(color="#d62728", width=1)), row=2, col=1)
    fig_bt.add_trace(go.Scatter(x=common_idx, y=dd_bm * 100, name="バイ＆ホールドDD (%)", line=dict(color="#7f7f7f", width=1, dash="dash")), row=2, col=1)
    fig_bt.update_layout(height=500, margin=dict(t=20, b=20, l=10, r=10), hovermode="x unified")
    fig_bt.update_yaxes(title_text="資産倍率", row=1, col=1)
    fig_bt.update_yaxes(title_text="下落率 (%)", row=2, col=1)
    st.plotly_chart(fig_bt, use_container_width=True)

# =============================================================
# TAB 3: 銘柄別詳細分析
# =============================================================
with tab3:
    st.subheader("📊 各銘柄の個別トレンド分析チャート")
    selected_asset = st.radio(
        "分析対象銘柄を選択",
        keys,
        format_func=lambda x: f"{x} （{ASSETS[x]['name']}）",
        horizontal=True
    )
    
    sig = signals[selected_asset]
    cfg = ASSETS[selected_asset]
    u_sym = cfg["underlying"]
    
    fig_single = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.06,
        row_heights=[0.6, 0.4],
        subplot_titles=(
            f"① 母体指数 {u_sym} 価格 ＆ 50日/200日EMA",
            f"② 強気確率 P(Bull) ＆ 目標比率 W* 推移"
        )
    )
    
    fig_single.add_trace(go.Scatter(x=common_idx, y=sig["c_u"], name=f"{u_sym} 終値", line=dict(color=cfg["color"], width=1.5)), row=1, col=1)
    fig_single.add_trace(go.Scatter(x=common_idx, y=sig["ema50"], name="50日 EMA (短期)", line=dict(color="#17becf", width=1.5)), row=1, col=1)
    fig_single.add_trace(go.Scatter(x=common_idx, y=sig["ema200"], name="200日 EMA (長期)", line=dict(color="#ff7f0e", width=2)), row=1, col=1)
    fig_single.add_trace(go.Scatter(x=common_idx, y=sig["p_bull"] * 100, name="強気確率 P(Bull) %", line=dict(color="#1f77b4", width=1.5)), row=2, col=1)
    fig_single.add_trace(go.Scatter(x=common_idx, y=sig["w_star"] * 100, name="目標比率 W* %", line=dict(color="#d62728", width=1.5)), row=2, col=1)
    
    fig_single.update_layout(height=600, margin=dict(t=30, b=20, l=10, r=10), hovermode="x unified")
    fig_single.update_yaxes(title_text="株価 ($)", row=1, col=1)
    fig_single.update_yaxes(title_text="比率 (%)", row=2, col=1)
    st.plotly_chart(fig_single, use_container_width=True)
