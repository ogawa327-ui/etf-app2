"""
SDE-Engine Pro v2.1: 高収益・リスク抑制両立版
改善点:
 1. 過剰ブレーキの排除: 強気相場では100%フルインベストメントを維持（アクセル全開）
 2. 確率スコアの正常化: トレンド強度をダイレクトに反映し、不要な半額ペナルティを排除
 3. レバレッジ適正Vol Target: 3倍ETFの爆発力を殺さない目標ボラティリティ設定（65%基準）
 4. 底売り防止: 押し目でポジションを投げるDDコントローラーを排除し、EMAトレンド完全退避に一本化
 5. 取引摩擦（10bps）・実ETF分離・ベンチマーク分離は維持
"""

from typing import Dict, Tuple, Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

# =============================================================
# 1. システム定数・アセット定義
# =============================================================
EXPENSE_RATIO_ANNUAL = 0.0095      # レバレッジETF年間経費率
CASH_YIELD_ANNUAL = 0.035          # 米ドルMMF年間利回り
TRADING_DAYS_PER_YEAR = 252
EPSILON = 1e-9

ASSETS: Dict[str, Dict[str, Any]] = {
    "TQQQ": {
        "name": "TQQQ (NASDAQ 3倍)", "underlying": "QQQ", "type": "ハイテク",
        "leverage": 3.0, "sigma_target": 65.0, "color": "#00ba38",
        "real_start": "2010-02-11"
    },
    "SPXL": {
        "name": "SPXL (S&P500 3倍)", "underlying": "SPY", "type": "米国全体",
        "leverage": 3.0, "sigma_target": 55.0, "color": "#619cff",
        "real_start": "2008-11-05"
    },
    "SOXL": {
        "name": "SOXL (半導体 3倍)", "underlying": "SOXX", "type": "半導体",
        "leverage": 3.0, "sigma_target": 75.0, "color": "#f5b041",
        "real_start": "2010-03-11"
    },
    "FAS": {
        "name": "FAS (金融株 3倍)", "underlying": "XLF", "type": "金融",
        "leverage": 3.0, "sigma_target": 60.0, "color": "#9b59b6",
        "real_start": "2008-11-05"
    },
    "UGL": {
        "name": "UGL (ゴールド 2倍)", "underlying": "GLD", "type": "ゴールド",
        "leverage": 2.0, "sigma_target": 35.0, "color": "#f1c40f",
        "real_start": "2008-12-01"
    },
}

ALL_REAL_START_DATE = "2010-03-11"

st.set_page_config(page_title="SDE-Engine Pro v2.1", page_icon="⚡", layout="wide")
st.sidebar.title("⚡ SDE-Engine Pro v2.1")

# =============================================================
# 2. サイドバーコントローラー
# =============================================================
period_mode = st.sidebar.radio(
    "データソース選択",
    ["全期間（実データ ＋ 合成データ）", "実ETFデータ限定期間（2010年3月〜現在）"]
)

period_options = {
    "5y (直近5年間)": "5y",
    "10y (直近10年間)": "10y",
    "15y (直近15年間)": "15y",
    "20y (直近20年間・リーマン含む)": "20y",
    "max (取得可能全期間)": "max"
}
selected_period_label = st.sidebar.selectbox("取得期間", list(period_options.keys()), index=4)
period_code = period_options[selected_period_label]

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ ポートフォリオ設定")
max_cap_pct = st.sidebar.slider("1銘柄あたり投資上限 (%)", 30, 100, 50, 5)
max_cap = max_cap_pct / 100.0

# レバレッジETF向けに目標Volを現実的な水準（デフォルト60%）に調整
target_pf_vol_pct = st.sidebar.slider("目標ポートフォリオ年率Vol (%)", 40, 90, 65, 5)
target_pf_vol = target_pf_vol_pct / 100.0

fee_bps = st.sidebar.number_input("売買手数料＋スリッページ (片道 bps)", min_value=0, max_value=30, value=10, step=1)
fee_rate = fee_bps / 10000.0

# =============================================================
# 3. データ取得 ＆ 幾何整合合成データ生成
# =============================================================
def _sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in required_cols:
        if col not in df.columns and "Close" in df.columns:
            df[col] = df["Close"]
    return df.dropna(subset=["Close"])

@st.cache_data(ttl=3600, show_spinner=False)
def load_and_sync_market_data(period_str: str) -> Tuple[Dict[str, pd.DataFrame], pd.DatetimeIndex]:
    u_tickers = [cfg["underlying"] for cfg in ASSETS.values()]
    etf_tickers = list(ASSETS.keys())
    all_tickers = sorted(list(set(u_tickers + etf_tickers)))

    raw_data: Dict[str, pd.DataFrame] = {}
    for t in all_tickers:
        try:
            df = yf.download(t, period=period_str, interval="1d", progress=False, auto_adjust=True)
            df = _sanitize_df(df)
            if not df.empty:
                raw_data[t] = df
        except Exception:
            continue

    if "SPY" not in raw_data or raw_data["SPY"].empty:
        raise RuntimeError("基準データ (SPY) の取得に失敗しました。")

    base_idx = raw_data["SPY"].index
    for u in u_tickers:
        if u in raw_data and not raw_data[u].empty:
            base_idx = base_idx.intersection(raw_data[u].index)

    cleaned_data: Dict[str, pd.DataFrame] = {u: raw_data[u].loc[base_idx].copy() for u in u_tickers}

    for sym, cfg in ASSETS.items():
        u_sym = cfg["underlying"]
        lev = cfg["leverage"]
        df_u = cleaned_data[u_sym]

        u_ret = df_u["Close"].pct_change().fillna(0.0)
        daily_expense = EXPENSE_RATIO_ANNUAL / TRADING_DAYS_PER_YEAR
        syn_ret = (u_ret * lev) - daily_expense
        cum_growth = (1.0 + syn_ret).cumprod()

        has_real = (sym in raw_data) and (not raw_data[sym].empty)
        real_idx = raw_data[sym].index.intersection(base_idx) if has_real else pd.DatetimeIndex([])

        if len(real_idx) > 0:
            df_real = raw_data[sym].loc[real_idx]
            first_real_price = float(df_real["Close"].iloc[0])
            growth_anchor = float(cum_growth.loc[real_idx[0]])
            scale = first_real_price / max(growth_anchor, EPSILON)

            synth_close = cum_growth * scale
            full_close = synth_close.copy()
            full_close.loc[real_idx] = df_real["Close"]
        else:
            full_close = 100.0 * cum_growth

        cleaned_data[sym] = pd.DataFrame({"Close": full_close.ffill().bfill()}, index=base_idx)

    return cleaned_data, base_idx

@st.cache_data(ttl=3600, show_spinner=False)
def get_usdjpy_rate() -> float:
    try:
        fx = yf.download("USDJPY=X", period="5d", interval="1d", progress=False)
        fx = _sanitize_df(fx)
        if not fx.empty and "Close" in fx.columns:
            return round(float(fx["Close"].iloc[-1]), 2)
    except Exception:
        pass
    return 155.0

with st.spinner("市場データを解析中..."):
    market_data, all_base_idx = load_and_sync_market_data(period_code)
    latest_fx_rate = get_usdjpy_rate()

if period_mode == "実ETFデータ限定期間（2010年3月〜現在）":
    common_idx = all_base_idx[all_base_idx >= pd.to_datetime(ALL_REAL_START_DATE)]
else:
    common_idx = all_base_idx

latest_date_str = common_idx[-1].strftime('%Y年%m月%d日')

# =============================================================
# 4. 高感度シグナル解析エンジン（不要なブレーキを解除）
# =============================================================
def calc_rsi(series: pd.Series, period: int = 9) -> np.ndarray:
    delta = series.diff().values
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(alpha=1.0/period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1.0/period, adjust=False).mean().values
    rs = avg_gain / np.maximum(avg_loss, EPSILON)
    return np.nan_to_num(100.0 - (100.0 / (1.0 + rs)), nan=50.0)

def run_asset_signals(sym: str) -> Dict[str, Any]:
    cfg = ASSETS[sym]
    df_u = market_data[cfg["underlying"]].loc[common_idx]
    df_etf = market_data[sym].loc[common_idx]

    c_u = df_u["Close"].values
    c_etf = df_etf["Close"].values
    n = len(c_u)

    # トレンド判定（50日EMA ＆ 200日EMA）
    ema50 = pd.Series(c_u).ewm(span=50, adjust=False).mean().values
    ema200 = pd.Series(c_u).ewm(span=200, adjust=False).mean().values
    diff_50 = (c_u - ema50) / np.maximum(ema50, EPSILON)
    diff_200 = (c_u - ema200) / np.maximum(ema200, EPSILON)

    # EMA50 ヒステリシス（ダマシ防止: -1%で待機、+0.5%で復帰）
    is_active = np.ones(n, dtype=bool)
    state = True
    for i in range(n):
        if state:
            if diff_50[i] < -0.01:
                state = False
        else:
            if diff_50[i] > 0.005:
                state = True
        is_active[i] = state

    # RSI（極端な買われすぎ85超のみ軽微な利益確定トリム -15%）
    rsi9 = calc_rsi(pd.Series(c_u), period=9)
    trim_factor = np.where(rsi9 > 85.0, 0.85, 1.0)

    # トレンド強度に基づく強気度スコア (0.0〜1.0)
    # EMA50および200日の上にあれば即座に高スコア（アクセル全開）
    trend_score = np.clip(0.5 + 2.5 * diff_50 + 1.0 * diff_200, 0.0, 1.0)
    
    # 50日EMA下抜け時は完全遮断 (0.0)
    w_star = np.where(is_active, trend_score * trim_factor, 0.0)
    # 明確な強気相場ではしっかり1.0（満額要求）にする
    w_star = np.where(w_star > 0.40, 1.0, w_star)

    etf_ret = np.zeros_like(c_etf)
    etf_ret[1:] = (c_etf[1:] - c_etf[:-1]) / np.maximum(c_etf[:-1], EPSILON)

    if not is_active[-1]:
        badge, desc = "🔴 弱気防衛 (待機)", "50日EMA下抜け・全額キャッシュ退避"
    elif rsi9[-1] > 85:
        badge, desc = "⚠️ 極短期過熱 (小幅利確)", f"RSI {rsi9[-1]:.0f} / 上昇過熱警戒"
    else:
        badge, desc = "🟢 強気巡航 (満額投資)", "上昇トレンド継続中・フル投資"

    return {
        "raw_w": w_star, "c_u": c_u, "c_etf": c_etf, "etf_ret": etf_ret,
        "ema50": ema50, "ema200": ema200, "rsi9": rsi9,
        "diff_50": diff_50[-1], "diff_200": diff_200[-1], "badge": badge, "desc": desc,
        "u_close": c_u[-1], "etf_price": c_etf[-1], "is_active": is_active[-1]
    }

asset_keys = list(ASSETS.keys())
signals = {s: run_asset_signals(s) for s in asset_keys}
n_days = len(common_idx)

# =============================================================
# 5. 資本効率最大化 ポートフォリオ配分エンジン
# =============================================================
ret_matrix = np.column_stack([signals[k]["etf_ret"] for k in asset_keys])
raw_matrix = np.column_stack([signals[k]["raw_w"] for k in asset_keys])

def allocate_capital_efficiently(raw_w_matrix: np.ndarray, returns: np.ndarray, max_c: float, target_v: float) -> np.ndarray:
    n, k = raw_w_matrix.shape
    final_alloc = np.zeros_like(raw_w_matrix)
    ret_df = pd.DataFrame(returns)
    roll_cov = ret_df.rolling(60, min_periods=20).cov().values.reshape(n, k, k) * TRADING_DAYS_PER_YEAR

    for t in range(n):
        w = raw_w_matrix[t].copy()
        active = np.where(w > 0.05)[0]
        if len(active) == 0:
            continue

        # 適格銘柄に上限まで目一杯資金を配分（資金をMMFに余らせない）
        alloc = np.zeros(k)
        pool = 1.0
        eligible = list(active)

        while eligible and pool > 1e-4:
            sub_w = w[eligible]
            sum_sub = np.sum(sub_w)
            if sum_sub <= 0:
                break
            prop = (sub_w / sum_sub) * pool
            hit_cap = []
            for i, idx in enumerate(eligible):
                if alloc[idx] + prop[i] >= max_c:
                    pool -= (max_c - alloc[idx])
                    alloc[idx] = max_c
                    hit_cap.append(idx)
                else:
                    alloc[idx] += prop[i]
                    pool -= prop[i]
            eligible = [idx for idx in eligible if idx not in hit_cap and alloc[idx] < max_c]
            if not hit_cap:
                break

        # ポートフォリオVol制限（極端なリスク時のみ縮小）
        cov_t = roll_cov[t]
        if not np.isnan(cov_t).any():
            pf_vol = np.sqrt(max(float(np.dot(alloc.T, np.dot(cov_t, alloc))), 1e-6))
            if pf_vol > target_v:
                alloc = alloc * (target_v / pf_vol)

        final_alloc[t] = alloc

    return final_alloc

daily_alloc_matrix = allocate_capital_efficiently(raw_matrix, ret_matrix, max_cap, target_pf_vol)

# 翌営業日執行（ルックアヘッドバイアス完全排除）
exec_matrix = np.zeros_like(daily_alloc_matrix)
exec_matrix[1:] = daily_alloc_matrix[:-1]

# =============================================================
# 6. バックテスト（取引コスト・正確なベンチマーク）
# =============================================================
daily_cash_rate = CASH_YIELD_ANNUAL / TRADING_DAYS_PER_YEAR
strat_daily_ret = np.zeros(n_days)

for t in range(n_days):
    w_t = exec_matrix[t]
    w_prev = exec_matrix[t-1] if t > 0 else np.zeros(len(asset_keys))
    cost = np.sum(np.abs(w_t - w_prev)) * fee_rate

    gross_ret = np.sum(w_t * ret_matrix[t])
    cash_ret = max(0.0, 1.0 - np.sum(w_t)) * daily_cash_rate
    strat_daily_ret[t] = gross_ret + cash_ret - cost

# ベンチマーク
bm_rebal_ret = np.mean(ret_matrix, axis=1)
cum_individual = np.cumprod(1.0 + ret_matrix, axis=0)
bm_bh_cum = np.mean(cum_individual, axis=1)

# 指標算出
def calc_metrics(ret: np.ndarray, cum_curve: np.ndarray) -> Dict[str, float]:
    years = max(len(ret) / TRADING_DAYS_PER_YEAR, 0.1)
    cagr = (cum_curve[-1] ** (1.0 / years)) - 1.0
    peaks = np.maximum.accumulate(cum_curve)
    dds = (cum_curve - peaks) / np.maximum(peaks, EPSILON)
    mdd = float(np.min(dds))
    vol = float(np.std(ret, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
    calmar = (cagr / abs(mdd)) if abs(mdd) > EPSILON else 0.0

    pos_rets = ret[ret > 0]
    neg_rets = ret[ret < 0]
    pf = (np.sum(pos_rets) / abs(np.sum(neg_rets))) if len(neg_rets) > 0 else 0.0
    return {"CAGR": cagr, "MDD": mdd, "Vol": vol, "Calmar": calmar, "PF": pf}

cum_strat = np.cumprod(1.0 + strat_daily_ret)
cum_bm_rebal = np.cumprod(1.0 + bm_rebal_ret)

m_strat = calc_metrics(strat_daily_ret, cum_strat)
m_bm_rebal = calc_metrics(bm_rebal_ret, cum_bm_rebal)
m_bm_bh = calc_metrics(strat_daily_ret, bm_bh_cum)

latest_alloc = daily_alloc_matrix[-1]
tot_invested = float(np.sum(latest_alloc))
tot_cash = max(0.0, 1.0 - tot_invested)

# =============================================================
# 7. UI表示
# =============================================================
st.title("⚡ SDE-Engine Pro v2.1 | クオンツ最適化システム")
st.caption(f"検証データ: **{period_mode}** ｜ 確定日: **{latest_date_str}**")

tab1, tab2 = st.tabs(["🧪 バックテスト検証（高収益・リスク抑制）", "🏛️ 最適配分 & 発注シミュレータ"])

with tab1:
    st.subheader("🧪 バックテスト検証結果")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("戦略 CAGR (年平均リターン)", f"{m_strat['CAGR']*100:.1f}%", f"BM均等: {m_bm_rebal['CAGR']*100:.1f}%")
    c2.metric("最大下落率 (MDD)", f"{m_strat['MDD']*100:.1f}%", f"BM均等: {m_bm_rebal['MDD']*100:.1f}%")
    c3.metric("カルマーレシオ (リターン/MDD)", f"{m_strat['Calmar']:.2f}", f"BM均等: {m_bm_rebal['Calmar']:.2f}")
    c4.metric("プロフィットファクター (PF)", f"{m_strat['PF']:.2f}")

    st.markdown("---")
    fig_bt = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    fig_bt.add_trace(go.Scatter(x=common_idx, y=cum_strat, name="SDE Pro v2.1 戦略", line=dict(color="#00ba38", width=2.5)), row=1, col=1)
    fig_bt.add_trace(go.Scatter(x=common_idx, y=cum_bm_rebal, name="BM: Daily Equal Weight", line=dict(color="#888888", width=1.2, dash="dot")), row=1, col=1)
    fig_bt.add_trace(go.Scatter(x=common_idx, y=bm_bh_cum, name="BM: Buy & Hold (等金額放置)", line=dict(color="#3498db", width=1.2, dash="dash")), row=1, col=1)

    peak_s = np.maximum.accumulate(cum_strat)
    dd_s = (cum_strat - peak_s) / np.maximum(peak_s, EPSILON)
    peak_b = np.maximum.accumulate(cum_bm_rebal)
    dd_b = (cum_bm_rebal - peak_b) / np.maximum(peak_b, EPSILON)

    fig_bt.add_trace(go.Scatter(x=common_idx, y=dd_s * 100, name="戦略DD (%)", fill="tozeroy", line=dict(color="#d62728", width=1)), row=2, col=1)
    fig_bt.add_trace(go.Scatter(x=common_idx, y=dd_b * 100, name="BM均等DD (%)", line=dict(color="#7f7f7f", width=1, dash="dash")), row=2, col=1)
    fig_bt.update_layout(height=520, margin=dict(t=20, b=20, l=10, r=10), hovermode="x unified")
    fig_bt.update_yaxes(title_text="累積資産成長 (倍率・対数)", type="log", row=1, col=1)
    fig_bt.update_yaxes(title_text="下落率 (%)", row=2, col=1)
    st.plotly_chart(fig_bt, use_container_width=True)

with tab2:
    st.subheader("📊 銘柄別推奨配分マトリクス")
    cols = st.columns(len(asset_keys))
    for idx_c, sym in enumerate(asset_keys):
        alloc = latest_alloc[idx_c]
        sig = signals[sym]
        with cols[idx_c]:
            st.markdown(f"#### {sym}")
            st.caption(f"{ASSETS[sym]['name']}")
            st.metric("株価", f"${sig['etf_price']:.2f}")
            st.info(f"**{sig['badge']}**\n\n*{sig['desc']}*")
            if alloc > 0:
                st.success(f"**推奨比率:**\n\n### {alloc*100:.1f} %")
            else:
                st.warning("**配分:**\n\n### 0.0 % (待機)")

    st.markdown("---")
    m1, m2 = st.columns(2)
    m1.metric("総株式投資比率", f"{tot_invested*100:.1f} %")
    m2.metric("米ドルMMF待機比率", f"{tot_cash*100:.1f} %")
