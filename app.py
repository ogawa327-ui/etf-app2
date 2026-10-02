"""
SDE-Engine Pro v2.1: クオンツ型動的レバレッジETFポートフォリオ管理システム
【機関投資家・クオンツ水準 厳密検証 ＆ 指標可視化 ＆ Pareto改善 完全版】
 - 10日 / 50日 / 200日 移動平均線乖離率の同時表示
 - 主要指標（P(Bull) / RSI / 確信度上限キャップ）の解説Expander
 - Pareto Frontier: 30秒解説カード、3大おすすめ設定の自動提案、文字重なり解消
 - Look-ahead Biasの完全排除: Purged Walk-Forward OOS シグナル生成
 - 統計的確率校正: CalibratedClassifierCV (Platt Scaling)
 - 翌朝寄り付き(Open)実約定モデル: Open-to-Open
 - 確信度連動型 残余資本再配分 (過大配分防止)
 - トレード単位の真の Trade PF & 勝率トラッキング
 - DDコントローラー連動の実効Exposure・実効資本効率計算
 - 時系列依存性を保持する Circular Block Bootstrap
 - 円建て(JPY) / ドル建て(USD) 切替
"""

from typing import Dict, Tuple, Any, List
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

# 機械学習ライブラリのインポート
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# =============================================================
# 1. 定数・アセット設定
# =============================================================
EXPENSE_RATIO_ANNUAL = 0.0095      # レバレッジETF年間経費率
CASH_YIELD_ANNUAL = 0.035          # 米ドルMMF年間利回り
TRADING_DAYS_PER_YEAR = 252        # 年間営業日数
EPSILON = 1e-9                     # ゼロ除算防止微小値

ASSETS: Dict[str, Dict[str, Any]] = {
    "TQQQ": {
        "name": "TQQQ (NASDAQ 3倍)", "underlying": "QQQ", "type": "ハイテク",
        "leverage": 3.0, "sigma_target": 55.0, "u_vol_norm": 20.0, "color": "#00ba38",
        "real_start": "2010-02-11"
    },
    "SPXL": {
        "name": "SPXL (S&P500 3倍)", "underlying": "SPY", "type": "米国全体",
        "leverage": 3.0, "sigma_target": 45.0, "u_vol_norm": 16.0, "color": "#619cff",
        "real_start": "2008-11-05"
    },
    "SOXL": {
        "name": "SOXL (半導体 3倍)", "underlying": "SOXX", "type": "半導体",
        "leverage": 3.0, "sigma_target": 65.0, "u_vol_norm": 28.0, "color": "#f5b041",
        "real_start": "2010-03-11"
    },
    "FAS": {
        "name": "FAS (金融株 3倍)", "underlying": "XLF", "type": "金融",
        "leverage": 3.0, "sigma_target": 50.0, "u_vol_norm": 18.0, "color": "#9b59b6",
        "real_start": "2008-11-05"
    },
    "UGL": {
        "name": "UGL (ゴールド 2倍)", "underlying": "GLD", "type": "ゴールド",
        "leverage": 2.0, "sigma_target": 35.0, "u_vol_norm": 13.0, "color": "#f1c40f",
        "real_start": "2008-12-01"
    },
}

ALL_REAL_START_DATE = "2010-03-11"

# =============================================================
# 2. UI & サイドバー設定
# =============================================================
st.set_page_config(
    page_title="SDE-Engine Pro v2.1 | 厳密クオンツ検証システム",
    page_icon="⚡",
    layout="wide"
)

st.sidebar.title("⚡ SDE-Engine Pro v2.1")
st.sidebar.caption("機関投資家水準・厳密OOSクオンツモデル")

st.sidebar.markdown("### 📅 データソース ＆ 期間")
period_mode = st.sidebar.radio(
    "データ種別",
    ["全期間（実データ ＋ 合成データ）", "実ETFデータ限定（2010年3月〜現在）"]
)

period_options = {
    "5y (直近5年間)": "5y",
    "10y (直近10年間)": "10y",
    "15y (直近15年間)": "15y",
    "20y (直近20年間・リーマン含む)": "20y",
    "25y (直近25年間・ITバブル含む)": "25y",
    "max (取得可能全期間)": "max"
}
selected_period_label = st.sidebar.selectbox("取得期間", list(period_options.keys()), index=2)
period_code = period_options[selected_period_label]

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ ポートフォリオ ＆ リスク制御")
max_cap_pct = st.sidebar.slider("1銘柄あたり絶対投資上限 (%)", 20, 100, 50, 5)
max_cap = max_cap_pct / 100.0

target_pf_vol_pct = st.sidebar.slider("目標ポートフォリオ年率Vol (%)", 20, 60, 40, 5)
target_pf_vol = target_pf_vol_pct / 100.0

vol_target_mode = st.sidebar.radio("Vol Targeting 方式", ["上限抑制のみ (Cap)", "双方向スケーリング (Targeting)"])

fee_bps = st.sidebar.number_input("売買コスト＋スリッページ (片道 bps)", min_value=0, max_value=50, value=10, step=1)
fee_rate = fee_bps / 10000.0

use_dd_controller = st.sidebar.checkbox("DD連動型リスク抑制 (DD Controller)", value=True)
base_currency = st.sidebar.radio("評価基準通貨", ["米ドル (USD)", "日本円 (JPY)"], horizontal=True)

# =============================================================
# 3. データ取得 & OHLC 幾何整合合成データ生成
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
    return df.dropna(subset=["Close", "Open"])

@st.cache_data(ttl=3600, show_spinner=False)
def load_and_sync_market_data(period_str: str) -> Tuple[Dict[str, pd.DataFrame], pd.Series, pd.DatetimeIndex]:
    u_tickers = [cfg["underlying"] for cfg in ASSETS.values()]
    etf_tickers = list(ASSETS.keys())
    all_tickers = sorted(list(set(u_tickers + etf_tickers + ["SPY", "USDJPY=X"])))

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
        raise RuntimeError("基準データ (SPY) の取得に失敗しました。時間をおいて再試行してください。")

    base_idx = raw_data["SPY"].index
    for u in u_tickers:
        if u in raw_data and not raw_data[u].empty:
            base_idx = base_idx.intersection(raw_data[u].index)

    if len(base_idx) < 60:
        raise ValueError("データ長が不足しています（60営業日未満）。")

    if "USDJPY=X" in raw_data and not raw_data["USDJPY=X"].empty:
        fx_series = raw_data["USDJPY=X"]["Close"].reindex(base_idx).ffill().bfill()
    else:
        fx_series = pd.Series(150.0, index=base_idx)

    cleaned_data: Dict[str, pd.DataFrame] = {}
    for u in u_tickers:
        cleaned_data[u] = raw_data[u].loc[base_idx].copy()

    daily_expense = EXPENSE_RATIO_ANNUAL / TRADING_DAYS_PER_YEAR
    for sym, cfg in ASSETS.items():
        u_sym = cfg["underlying"]
        lev = cfg["leverage"]
        df_u = cleaned_data[u_sym]

        has_real_etf = (sym in raw_data) and (not raw_data[sym].empty)
        real_idx = raw_data[sym].index.intersection(base_idx) if has_real_etf else pd.DatetimeIndex([])

        u_close = df_u["Close"].values
        u_open = df_u["Open"].values
        u_high = df_u["High"].values
        u_low = df_u["Low"].values
        n = len(base_idx)

        syn_close = np.zeros(n)
        syn_open = np.zeros(n)
        syn_high = np.zeros(n)
        syn_low = np.zeros(n)
        syn_close[0] = 100.0
        syn_open[0] = 100.0
        syn_high[0] = 100.0
        syn_low[0] = 100.0

        for t in range(1, n):
            ret_over = (u_open[t] / max(u_close[t-1], EPSILON) - 1.0) * lev
            syn_open[t] = max(syn_close[t-1] * (1.0 + ret_over), EPSILON)
            ret_intra = (u_close[t] / max(u_open[t], EPSILON) - 1.0) * lev - daily_expense
            syn_close[t] = max(syn_open[t] * (1.0 + ret_intra), EPSILON)

            hl_ratio = (u_high[t] - u_low[t]) / max(u_close[t], EPSILON) * lev
            syn_high[t] = max(syn_open[t], syn_close[t]) * (1.0 + hl_ratio * 0.5)
            syn_low[t] = min(syn_open[t], syn_close[t]) * max(1.0 - hl_ratio * 0.5, 0.01)

        syn_df = pd.DataFrame({
            "Open": syn_open, "High": syn_high, "Low": syn_low, "Close": syn_close
        }, index=base_idx)

        if len(real_idx) > 0:
            df_real = raw_data[sym].loc[real_idx]
            first_real_date = real_idx[0]
            first_real_close = float(df_real["Close"].iloc[0])
            scale = first_real_close / max(syn_df.loc[first_real_date, "Close"], EPSILON)

            syn_df.loc[syn_df.index < first_real_date] *= scale
            for col in ["Open", "High", "Low", "Close"]:
                syn_df.loc[real_idx, col] = df_real[col]

        cleaned_data[sym] = syn_df.ffill().bfill()

    return cleaned_data, fx_series, base_idx

with st.spinner("市場データを取得・検証中..."):
    try:
        market_data, fx_rates, all_base_idx = load_and_sync_market_data(period_code)
    except Exception as e:
        st.error(f"データ取得エラー: {str(e)}")
        st.stop()

if period_mode == "実ETFデータ限定（2010年3月〜現在）":
    common_idx = all_base_idx[all_base_idx >= pd.to_datetime(ALL_REAL_START_DATE)]
    if len(common_idx) < 50:
        common_idx = all_base_idx
else:
    common_idx = all_base_idx

latest_date_str = common_idx[-1].strftime('%Y年%m月%d日')
latest_fx = float(fx_rates.loc[common_idx[-1]])

# =============================================================
# 4. 特徴量 ＆ Purged Walk-Forward 確率校正エンジン
# =============================================================
def calc_rsi(series: pd.Series, period: int = 9) -> np.ndarray:
    delta = series.diff().values
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    alpha = 1.0 / period
    avg_gain = pd.Series(gain).ewm(alpha=alpha, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=alpha, adjust=False).mean().values
    rs = avg_gain / np.maximum(avg_loss, EPSILON)
    return np.nan_to_num(100.0 - (100.0 / (1.0 + rs)), nan=50.0)

def calc_parkinson_vol(df: pd.DataFrame, window: int = 10) -> np.ndarray:
    h = np.maximum(df["High"].values, EPSILON)
    l = np.maximum(df["Low"].values, EPSILON)
    factor = 1.0 / (4.0 * np.log(2.0))
    rolling_var = pd.Series((np.log(h / l)) ** 2 * factor).rolling(window, min_periods=1).mean().values
    pv = np.sqrt(np.maximum(rolling_var, 0.0)) * np.sqrt(TRADING_DAYS_PER_YEAR) * 100.0
    return np.where(np.isnan(pv), 25.0, pv)

def compute_asset_signals_purged(sym: str) -> Dict[str, Any]:
    cfg = ASSETS[sym]
    df_u = market_data[cfg["underlying"]].loc[common_idx]
    df_etf = market_data[sym].loc[common_idx]

    c_u = df_u["Close"].values
    o_etf = df_etf["Open"].values
    c_etf = df_etf["Close"].values
    n = len(c_u)

    # 1. 指標計算（10日 / 50日 / 200日 移動平均線と乖離率）
    ema10 = pd.Series(c_u).ewm(span=10, adjust=False).mean().values
    ema50 = pd.Series(c_u).ewm(span=50, adjust=False).mean().values
    ema200 = pd.Series(c_u).ewm(span=200, adjust=False).mean().values

    diff_10 = (c_u - ema10) / np.maximum(ema10, EPSILON)
    diff_50 = (c_u - ema50) / np.maximum(ema50, EPSILON)
    diff_200 = (c_u - ema200) / np.maximum(ema200, EPSILON)
    trend_score = 0.60 * diff_50 + 0.40 * diff_200

    hysteresis = np.ones(n, dtype=bool)
    state = True
    for i in range(n):
        if state and diff_50[i] < -0.01:
            state = False
        elif not state and diff_50[i] > 0.01:
            state = True
        hysteresis[i] = state

    u_ret = pd.Series(c_u).pct_change().fillna(0.0)
    vol_20 = u_ret.rolling(20, min_periods=1).std().values * np.sqrt(TRADING_DAYS_PER_YEAR) * 100.0
    vol_score = -(vol_20 - cfg["u_vol_norm"]) / cfg["u_vol_norm"]

    rsi9 = calc_rsi(pd.Series(c_u), 9)
    rsi_q85 = pd.Series(rsi9).rolling(252, min_periods=30).quantile(0.85).values
    dyn_rsi_th = np.clip(np.nan_to_num(rsi_q85, nan=66.0), 62.0, 75.0)

    p_rsi = np.where(rsi9 > dyn_rsi_th, (rsi9 - dyn_rsi_th) / 20.0, 0.0)
    p_ema = np.where(diff_50 > 0.08, (diff_50 - 0.08) / 0.10, 0.0)
    overheat_penalty = np.clip(p_rsi + p_ema, 0.0, 0.60)

    sigma_local = calc_parkinson_vol(df_etf, 10)

    # 2. 目的変数 (5日先リターン) & 特徴量
    fwd_ret_5d = pd.Series(c_u).pct_change(5).shift(-5).values
    y_target = np.where(fwd_ret_5d > 0, 1, 0)
    X_features = np.column_stack([trend_score, vol_score, overheat_penalty])

    # 3. Purged Walk-Forward 確率予測 & 本物のPlatt Scaling
    p_bull = np.zeros(n)
    train_win = 252 * 3
    embargo = 5
    refit_freq = 42

    current_model = None
    for t in range(n):
        if t < train_win + embargo:
            logit = 4.0 * trend_score[t] + 1.0 * vol_score[t] - 2.5 * overheat_penalty[t]
            p_bull[t] = 1.0 / (1.0 + np.exp(-np.clip(logit, -15.0, 15.0)))
            continue

        if (t % refit_freq == 0) or (current_model is None):
            train_end = t - embargo
            train_start = max(0, train_end - train_win)
            X_tr = X_features[train_start:train_end]
            y_tr = y_target[train_start:train_end]

            if HAS_SKLEARN and len(np.unique(y_tr)) > 1:
                try:
                    base_lr = LogisticRegression(C=1.0, max_iter=200)
                    try:
                        cal_clf = CalibratedClassifierCV(estimator=base_lr, method='sigmoid', cv=3)
                    except TypeError:
                        cal_clf = CalibratedClassifierCV(base_estimator=base_lr, method='sigmoid', cv=3)
                    cal_clf.fit(X_tr, y_tr)
                    current_model = cal_clf
                except Exception:
                    current_model = None
            else:
                current_model = None

        if current_model is not None:
            try:
                p_bull[t] = current_model.predict_proba(X_features[t:t+1])[0, 1]
            except Exception:
                logit = 4.0 * trend_score[t] + 1.0 * vol_score[t] - 2.5 * overheat_penalty[t]
                p_bull[t] = 1.0 / (1.0 + np.exp(-np.clip(logit, -15.0, 15.0)))
        else:
            logit = 4.0 * trend_score[t] + 1.0 * vol_score[t] - 2.5 * overheat_penalty[t]
            p_bull[t] = 1.0 / (1.0 + np.exp(-np.clip(logit, -15.0, 15.0)))

    p_bull = np.where(~hysteresis, np.minimum(p_bull, 0.15), p_bull)

    vol_adj = np.clip(cfg["sigma_target"] / np.maximum(sigma_local, 1e-4), 0.2, 1.2)
    w_star = np.clip(p_bull * vol_adj, 0.0, 1.0) * (1.0 - overheat_penalty)
    w_star = np.where(~hysteresis, 0.0, w_star)

    cond_cap = np.where(
        p_bull < 0.40, 0.10,
        np.where(p_bull < 0.55, 0.25,
        np.where(p_bull < 0.70, 0.40, max_cap))
    )
    cond_cap = np.minimum(cond_cap, max_cap)
    raw_desired_w = np.where((p_bull < 0.35) | (w_star < 0.10), 0.0, w_star)
    raw_desired_w = np.minimum(raw_desired_w, cond_cap)

    if not hysteresis[-1]:
        badge, desc = "🔴 弱気防衛 (待機)", "EMA50割れヒステリシス発動 / 全面防衛待機"
    elif overheat_penalty[-1] > 0.15:
        badge, desc = "⚠️ 過熱警戒 (抑制)", f"RSI {rsi9[-1]:.1f} (動的閾値 {dyn_rsi_th[-1]:.1f}) 過熱ペナルティ作動"
    elif p_bull[-1] < 0.55:
        badge, desc = "🟡 探査打診 (抑制配分)", f"打診フェーズ (P(Bull) {p_bull[-1]*100:.1f}%)"
    else:
        badge, desc = "🟢 本玉巡航 (積極配分)", f"強気トレンド継続 (P(Bull) {p_bull[-1]*100:.1f}%)"

    return {
        "p_bull": p_bull, "w_star": w_star, "raw_w": raw_desired_w, "cond_cap": cond_cap,
        "sigma_local": sigma_local,
        "ema10": ema10, "ema50": ema50, "ema200": ema200,
        "diff_10": diff_10, "diff_50": diff_50, "diff_200": diff_200,
        "rsi9": rsi9, "dyn_rsi_th": dyn_rsi_th, "hysteresis": hysteresis,
        "c_u": c_u, "o_etf": o_etf, "c_etf": c_etf,
        "latest_p": p_bull[-1], "latest_w": w_star[-1], "latest_rsi": rsi9[-1],
        "latest_diff10": diff_10[-1],
        "latest_diff50": diff_50[-1],
        "latest_diff200": diff_200[-1],
        "latest_th": dyn_rsi_th[-1], "badge": badge, "desc": desc,
        "u_close": c_u[-1], "etf_price": c_etf[-1]
    }

asset_keys = list(ASSETS.keys())
signals = {s: compute_asset_signals_purged(s) for s in asset_keys}
n_days = len(common_idx)

# =============================================================
# 5. ポートフォリオ最適化 (確信度連動再配分 ＆ Vol Targeting)
# =============================================================
raw_matrix = np.column_stack([signals[k]["raw_w"] for k in asset_keys])
caps_matrix = np.column_stack([signals[k]["cond_cap"] for k in asset_keys])

o_etf_matrix = np.column_stack([signals[k]["o_etf"] for k in asset_keys])
c_etf_matrix = np.column_stack([signals[k]["c_etf"] for k in asset_keys])

ret_open_to_open = np.zeros((n_days, len(asset_keys)))
ret_open_to_open[1:] = (o_etf_matrix[1:] - o_etf_matrix[:-1]) / np.maximum(o_etf_matrix[:-1], EPSILON)

def optimize_portfolio_flow(raw_w: np.ndarray, caps: np.ndarray, rets: np.ndarray, target_v: float, is_bidirectional: bool) -> np.ndarray:
    n, k = raw_w.shape
    final_alloc = np.zeros_like(raw_w)
    ret_df = pd.DataFrame(rets)
    roll_cov = ret_df.rolling(60, min_periods=20).cov().values.reshape(n, k, k) * TRADING_DAYS_PER_YEAR

    for t in range(n):
        w = raw_w[t].copy()
        c_limits = caps[t].copy()
        active = np.where(w > 0.05)[0]
        if len(active) == 0:
            continue

        alloc = np.minimum(w, c_limits)
        pool = 1.0 - np.sum(alloc)
        eligible = [i for i in active if alloc[i] < c_limits[i]]

        while eligible and pool > 1e-4:
            sub_w = w[eligible]
            s_sum = np.sum(sub_w)
            if s_sum <= 0:
                break
            prop = (sub_w / s_sum) * pool
            hit = []
            for idx_e, a_idx in enumerate(eligible):
                addable = c_limits[a_idx] - alloc[a_idx]
                if prop[idx_e] >= addable:
                    alloc[a_idx] = c_limits[a_idx]
                    pool -= addable
                    hit.append(a_idx)
                else:
                    alloc[a_idx] += prop[idx_e]
                    pool -= prop[idx_e]
            eligible = [i for i in eligible if i not in hit and alloc[i] < c_limits[i]]
            if not hit:
                break

        cov_t = roll_cov[t]
        if not np.isnan(cov_t).any():
            pf_var = float(np.dot(alloc.T, np.dot(cov_t, alloc)))
            pf_vol = np.sqrt(max(pf_var, 1e-6))
            if pf_vol > 0.01:
                scale = target_v / pf_vol
                if is_bidirectional:
                    scale = np.clip(scale, 0.3, 1.3)
                    alloc = np.minimum(alloc * scale, c_limits)
                else:
                    if pf_vol > target_v:
                        alloc = alloc * (target_v / pf_vol)

        final_alloc[t] = alloc

    return final_alloc

is_bidir = (vol_target_mode == "双方向スケーリング (Targeting)")
daily_alloc_matrix = optimize_portfolio_flow(raw_matrix, caps_matrix, ret_open_to_open, target_pf_vol, is_bidir)

# =============================================================
# 6. 実約定バックテスト ＆ トレード単位PF集計エンジン
# =============================================================
exec_alloc = np.zeros_like(daily_alloc_matrix)
exec_alloc[1:] = daily_alloc_matrix[:-1]

def run_rigorous_backtest(exec_w: np.ndarray, o_matrix: np.ndarray, c_matrix: np.ndarray, fx: np.ndarray, fee: float, use_dd: bool, is_jpy: bool):
    n, k = exec_w.shape
    strat_net_ret = np.zeros(n)
    effective_w = np.zeros_like(exec_w)
    effective_turnover = np.zeros(n)
    daily_cash_rate = CASH_YIELD_ANNUAL / TRADING_DAYS_PER_YEAR
    peak = 1.0
    cum = 1.0

    open_trades: Dict[int, Dict[str, float]] = {}
    completed_trades: List[Dict[str, Any]] = []

    for t in range(1, n):
        w_t = exec_w[t].copy()

        if use_dd:
            dd = (cum - peak) / peak
            if dd < -0.20:
                w_t *= 0.40
            elif dd < -0.10:
                w_t *= 0.70

        effective_w[t] = w_t
        turnover = float(np.sum(np.abs(w_t - effective_w[t-1])))
        effective_turnover[t] = turnover
        cost = turnover * fee

        r_asset_daily = (o_matrix[t] - o_matrix[t-1]) / np.maximum(o_matrix[t-1], EPSILON)
        gross_ret = float(np.sum(w_t * r_asset_daily))
        cash_ret = float(max(0.0, 1.0 - np.sum(w_t))) * daily_cash_rate

        net_ret = gross_ret + cash_ret - cost

        if is_jpy and t > 0:
            fx_ret = (fx[t] - fx[t-1]) / max(fx[t-1], EPSILON)
            net_ret = (1.0 + net_ret) * (1.0 + fx_ret) - 1.0

        strat_net_ret[t] = net_ret
        cum *= (1.0 + net_ret)
        if cum > peak:
            peak = cum

        for a_idx in range(k):
            prev_pos = effective_w[t-1, a_idx]
            curr_pos = w_t[a_idx]

            if prev_pos <= 0.01 and curr_pos > 0.01:
                open_trades[a_idx] = {
                    "entry_t": t, "entry_p": o_matrix[t, a_idx], "weight": curr_pos, "cum_ret": 1.0
                }
            elif curr_pos > 0.01 and a_idx in open_trades:
                day_r = (o_matrix[t, a_idx] - o_matrix[t-1, a_idx]) / max(o_matrix[t-1, a_idx], EPSILON)
                open_trades[a_idx]["cum_ret"] *= (1.0 + day_r)
            elif prev_pos > 0.01 and curr_pos <= 0.01 and a_idx in open_trades:
                tr = open_trades.pop(a_idx)
                exit_p = o_matrix[t, a_idx]
                pnl = (exit_p / max(tr["entry_p"], EPSILON) - 1.0) - (fee * 2)
                completed_trades.append({
                    "asset": asset_keys[a_idx], "entry_date": common_idx[int(tr["entry_t"])],
                    "exit_date": common_idx[t], "holding_days": t - int(tr["entry_t"]),
                    "pnl_pct": pnl * 100.0, "net_pnl": pnl
                })

    return strat_net_ret, effective_w, effective_turnover, completed_trades

use_jpy = (base_currency == "日本円 (JPY)")
fx_array = fx_rates.loc[common_idx].values

strat_ret, eff_w, eff_turnover, all_trades = run_rigorous_backtest(
    exec_alloc, o_etf_matrix, c_etf_matrix, fx_array, fee_rate, use_dd_controller, use_jpy
)

bm_eq_ret = np.zeros(n_days)
for t in range(1, n_days):
    r_bm = np.mean((o_etf_matrix[t] - o_etf_matrix[t-1]) / np.maximum(o_etf_matrix[t-1], EPSILON))
    if use_jpy:
        fx_r = (fx_array[t] - fx_array[t-1]) / max(fx_array[t-1], EPSILON)
        bm_eq_ret[t] = (1.0 + r_bm) * (1.0 + fx_r) - 1.0
    else:
        bm_eq_ret[t] = r_bm

def evaluate_performance(rets: np.ndarray, eff_weights: np.ndarray, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    cum = np.cumprod(1.0 + rets)
    years = max(len(rets) / TRADING_DAYS_PER_YEAR, 0.1)
    cagr = (cum[-1] ** (1.0 / years)) - 1.0

    peaks = np.maximum.accumulate(cum)
    dds = (cum - peaks) / np.maximum(peaks, EPSILON)
    mdd = float(np.min(dds))
    vol = float(np.std(rets, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
    calmar = (cagr / abs(mdd)) if abs(mdd) > EPSILON else 0.0

    var_95 = float(np.percentile(rets, 5))
    cvar_95 = float(np.mean(rets[rets <= var_95]))

    mean_exposure = float(np.mean(np.sum(eff_weights, axis=1)))
    cap_efficiency = cagr / max(mean_exposure, 0.05)

    pos_r = rets[rets > 0]
    neg_r = rets[rets < 0]
    daily_pf = (np.sum(pos_r) / abs(np.sum(neg_r))) if len(neg_r) > 0 and abs(np.sum(neg_r)) > EPSILON else 0.0

    if len(trades) > 0:
        win_trades = [t["net_pnl"] for t in trades if t["net_pnl"] > 0]
        loss_trades = [t["net_pnl"] for t in trades if t["net_pnl"] < 0]
        trade_win_rate = len(win_trades) / len(trades)
        sum_win = np.sum(win_trades) if len(win_trades) > 0 else 0.0
        sum_loss = abs(np.sum(loss_trades)) if len(loss_trades) > 0 else 0.0
        trade_pf = (sum_win / sum_loss) if sum_loss > EPSILON else (99.0 if sum_win > 0 else 0.0)
        avg_hold_days = float(np.mean([t["holding_days"] for t in trades]))
    else:
        trade_win_rate, trade_pf, avg_hold_days = 0.0, 0.0, 0.0

    return {
        "CAGR": cagr, "MDD": mdd, "Vol": vol, "Calmar": calmar,
        "Daily_PF": daily_pf, "Trade_PF": trade_pf, "Trade_WinRate": trade_win_rate,
        "CVaR_95": cvar_95, "Mean_Exposure": mean_exposure, "Cap_Efficiency": cap_efficiency,
        "Avg_Hold_Days": avg_hold_days, "Total_Trades": len(trades)
    }

metrics_strat = evaluate_performance(strat_ret, eff_w, all_trades)
metrics_bm = evaluate_performance(bm_eq_ret, np.ones((n_days, len(asset_keys))) * 0.2, [])

cum_strat = np.cumprod(1.0 + strat_ret)
cum_bm = np.cumprod(1.0 + bm_eq_ret)

latest_alloc = daily_alloc_matrix[-1]
tot_inv = float(np.sum(latest_alloc))
tot_cash = max(0.0, 1.0 - tot_inv)

# =============================================================
# 7. ダッシュボード・プレゼンテーション層
# =============================================================
st.title("⚡ SDE-Engine Pro v2.1 | クオンツ最適化システム")
st.caption(f"検証モード: **{period_mode}** ｜ 通貨: **{base_currency}** ｜ データ境界確定日: **{latest_date_str}** ({n_days}営業日)")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏛️ 最適配分 ＆ 発注シミュレータ",
    "🧪 バックテスト検証 ＆ 厳密統計評価",
    "📜 トレード詳細履歴 (真のTrade PF)",
    "🔄 Purged Walk-Forward (OOS) 検証",
    "🎲 Block Bootstrap モンテカルロ",
    "💎 3目的 Pareto Frontier 探索"
])

# -------------------------------------------------------------
# TAB 1: 最適配分 ＆ 発注シミュレータ
# -------------------------------------------------------------
with tab1:
    st.info(f"📌 **判定日: {latest_date_str}（終値確定シグナル → 翌朝NY寄り付き執行）**")

    with st.expander("ℹ️ 主要クオンツ指標の解説・見方（クリックで開閉）"):
        exp_col1, exp_col2, exp_col3 = st.columns(3)
        with exp_col1:
            st.markdown(
                "**📈 校正後 P(Bull)**\n\n"
                "過去データのみで学習・確率校正（Platt Scaling）された**「今後5日間に上昇する確率」**です。"
                "未来データの混入を完全に排除し、実運用で偏りが出ないよう統計的に整合させています。"
            )
        with exp_col2:
            st.markdown(
                "**⚡ 9日動的RSI**\n\n"
                "直近の短期買われすぎ水準を測定します。"
                "固定値（70等）ではなく、過去1年の85%タイルを動的閾値とし、**閾値超過時に買付を自動抑制（過熱ブレーキ）**します。"
            )
        with exp_col3:
            st.markdown(
                "**🛡️ 確信度上限キャップ**\n\n"
                "P(Bull)の確信度に応じて設定される**個別銘柄の最大配分枠（10%〜50%）**です。"
                "余剰資金の再配分時、自信度が低い「打診銘柄」に過大な資金が流れ込むのを防ぎます。"
            )

    st.markdown("---")

    cols = st.columns(len(asset_keys))
    for idx_c, sym in enumerate(asset_keys):
        sig = signals[sym]
        alloc_ratio = latest_alloc[idx_c]
        with cols[idx_c]:
            st.markdown(f"#### {sym}")
            st.caption(f"{ASSETS[sym]['name']}")
            st.metric(f"原資産 {ASSETS[sym]['underlying']}", f"${sig['u_close']:.2f}")

            st.write(f"校正後 P(Bull): **{sig['latest_p']*100:.1f}%**")
            st.progress(float(np.clip(sig["latest_p"], 0.0, 1.0)))

            st.caption(f"**9日RSI:** {sig['latest_rsi']:.1f} (動的閾値: {sig['latest_th']:.1f})")

            # 10日・50日・200日 移動平均線乖離率
            st.caption(
                f"**乖離率:** 10日: `{sig['latest_diff10']*100:+.1f}%` ｜ "
                f"50日: `{sig['latest_diff50']*100:+.1f}%` ｜ "
                f"200日: `{sig['latest_diff200']*100:+.1f}%`"
            )

            st.caption(f"**確信度上限キャップ:** {sig['cond_cap'][-1]*100:.1f}%")

            st.info(f"**{sig['badge']}**\n\n*{sig['desc']}*")
            if alloc_ratio > 0:
                st.success(f"**推奨配分:**\n\n### {alloc_ratio*100:.1f} %")
            else:
                st.warning("**配分:**\n\n### 0.0 % (待機)")

    st.markdown("---")
    c_s1, c_s2 = st.columns([1.5, 1])
    with c_s1:
        st.markdown("### 💼 ポートフォリオ全体の実効アロケーション")
        m_c1, m_c2, m_c3 = st.columns(3)
        m_c1.metric("総株式エクスポージャー", f"{tot_inv*100:.1f} %")
        m_c2.metric("米ドルMMF待機比率", f"{tot_cash*100:.1f} %")
        m_c3.metric("目標PFボラティリティ", f"{target_pf_vol_pct} % ({vol_target_mode})")
        st.caption("※シグナル強度連動キャップにより、低確信度銘柄への過大配分を抑制済み。")

    with c_s2:
        active_labels = [asset_keys[i] for i in range(len(asset_keys)) if latest_alloc[i] > 0] + ["米ドルMMF"]
        active_vals = [latest_alloc[i] * 100 for i in range(len(asset_keys)) if latest_alloc[i] > 0] + [tot_cash * 100]
        fig_pie = go.Figure(data=[go.Pie(labels=active_labels, values=active_vals, hole=.45)])
        fig_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=200)
        st.plotly_chart(fig_pie, use_container_width=True)

    # 発注シミュレータ
    st.markdown("---")
    st.subheader("💡 証券会社 翌朝寄り付き発注シミュレーター")
    curr_c1, curr_c2, curr_c3 = st.columns([1.2, 1.5, 1.3])
    with curr_c1:
        in_curr = st.radio("入力通貨", ["日本円 (万円)", "米ドル (USD)"], horizontal=True)
    with curr_c2:
        if in_curr == "日本円 (万円)":
            f_jpy_man = st.number_input("運用資産総額（万円）", min_value=10, value=500, step=10)
            total_usd = (f_jpy_man * 10000.0) / latest_fx
            total_jpy = f_jpy_man * 10000.0
        else:
            total_usd = float(st.number_input("運用資産総額 (USD)", min_value=1000, value=30000, step=1000))
            total_jpy = total_usd * latest_fx
    with curr_c3:
        st.metric("最新為替レート (USD/JPY)", f"¥{latest_fx:.2f}")

    sim_rows = []
    for idx_c, sym in enumerate(asset_keys):
        alloc_ratio = latest_alloc[idx_c]
        p = signals[sym]["etf_price"]
        t_usd = total_usd * alloc_ratio
        t_jpy = t_usd * latest_fx
        shares = int(t_usd // p) if p > 0 else 0
        action = f"目標 {shares} 株 買付" if alloc_ratio > 0 else "保有なし / 全売却待機"
        sim_rows.append({
            "銘柄": ASSETS[sym]["name"],
            "シグナル判定": signals[sym]["badge"],
            "最適配分": f"{alloc_ratio*100:.1f} %",
            "投資目標額 (USD)": f"${t_usd:,.2f}",
            "概算金額 (JPY)": f"約 {t_jpy:,.0f} 円" if alloc_ratio > 0 else "0 円",
            "参考株価": f"${p:.2f}",
            "執行株数": f"{shares} 株",
            "アクション": action
        })
    c_usd = total_usd * tot_cash
    sim_rows.append({
        "銘柄": "米ドルMMF / 現金待機", "シグナル判定": "🛡️️ 安全待機", "最適配分": f"{tot_cash*100:.1f} %",
        "投資目標額 (USD)": f"${c_usd:,.2f}", "概算金額 (JPY)": f"約 {c_usd*latest_fx:,.0f} 円",
        "参考株価": "-", "執行株数": "-", "アクション": "MMF待機 (年利約3.5%)"
    })
    st.dataframe(pd.DataFrame(sim_rows), use_container_width=True, hide_index=True)

# -------------------------------------------------------------
# TAB 2: バックテスト検証 ＆ 厳密統計評価
# -------------------------------------------------------------
with tab2:
    st.subheader("🧪 厳密クオンツ・バックテスト検証 (ネット費用後)")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("CAGR", f"{metrics_strat['CAGR']*100:.1f}%", f"BM均等: {metrics_bm['CAGR']*100:.1f}%")
    c2.metric("最大下落率 (MDD)", f"{metrics_strat['MDD']*100:.1f}%", f"BM均等: {metrics_bm['MDD']*100:.1f}%")
    c3.metric("Trade PF (真値)", f"{metrics_strat['Trade_PF']:.2f}", f"Daily PF: {metrics_strat['Daily_PF']:.2f}")
    c4.metric("トレード勝率", f"{metrics_strat['Trade_WinRate']*100:.1f}%", f"全 {metrics_strat['Total_Trades']} 件")
    c5.metric("実効資本効率 (CAGR/Exp)", f"{metrics_strat['Cap_Efficiency']:.2f}", f"平均Exp: {metrics_strat['Mean_Exposure']*100:.1f}%")
    c6.metric("95% CVaR (日次)", f"{metrics_strat['CVaR_95']*100:.2f}%")

    st.markdown("---")
    fig_bt = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    fig_bt.add_trace(go.Scatter(x=common_idx, y=cum_strat, name="SDE Pro v2.1 (厳密Open約定・ネット費用後)", line=dict(color="#00ba38", width=2.5)), row=1, col=1)
    fig_bt.add_trace(go.Scatter(x=common_idx, y=cum_bm, name="BM: Daily Equal Weight (毎日均等)", line=dict(color="#888888", width=1.5, dash="dot")), row=1, col=1)

    peak_s = np.maximum.accumulate(cum_strat)
    dd_s = (cum_strat - peak_s) / np.maximum(peak_s, EPSILON)
    peak_b = np.maximum.accumulate(cum_bm)
    dd_b = (cum_bm - peak_b) / np.maximum(peak_b, EPSILON)

    fig_bt.add_trace(go.Scatter(x=common_idx, y=dd_s * 100, name="戦略DD (%)", fill="tozeroy", line=dict(color="#d62728", width=1)), row=2, col=1)
    fig_bt.add_trace(go.Scatter(x=common_idx, y=dd_b * 100, name="BM均等DD (%)", line=dict(color="#7f7f7f", width=1, dash="dash")), row=2, col=1)
    fig_bt.update_layout(height=520, margin=dict(t=20, b=20, l=10, r=10), hovermode="x unified")
    fig_bt.update_yaxes(title_text="累積資産成長 (対数軸)", type="log", row=1, col=1)
    fig_bt.update_yaxes(title_text="ドローダウン (%)", row=2, col=1)
    st.plotly_chart(fig_bt, use_container_width=True)

    st.markdown("### 📋 リスク・リターン ＆ 資本効率 詳細対比")
    perf_data = [
        {"戦略": "SDE-Engine Pro v2.1", "CAGR": f"{metrics_strat['CAGR']*100:.1f}%", "MDD": f"{metrics_strat['MDD']*100:.1f}%", "Calmar": f"{metrics_strat['Calmar']:.2f}", "真のTrade PF": f"{metrics_strat['Trade_PF']:.2f}", "Daily PF": f"{metrics_strat['Daily_PF']:.2f}", "トレード勝率": f"{metrics_strat['Trade_WinRate']*100:.1f}%", "実効平均Exp": f"{metrics_strat['Mean_Exposure']*100:.1f}%", "資本効率": f"{metrics_strat['Cap_Efficiency']:.2f}", "日次95% CVaR": f"{metrics_strat['CVaR_95']*100:.2f}%"},
        {"戦略": "BM: Equal Weight", "CAGR": f"{metrics_bm['CAGR']*100:.1f}%", "MDD": f"{metrics_bm['MDD']*100:.1f}%", "Calmar": f"{metrics_bm['Calmar']:.2f}", "真のTrade PF": "-", "Daily PF": f"{metrics_bm['Daily_PF']:.2f}", "トレード勝率": "-", "実効平均Exp": "100.0%", "資本効率": f"{metrics_bm['Cap_Efficiency']:.2f}", "日次95% CVaR": f"{metrics_bm['CVaR_95']*100:.2f}%"}
    ]
    st.dataframe(pd.DataFrame(perf_data), use_container_width=True, hide_index=True)

# -------------------------------------------------------------
# TAB 3: トレード詳細履歴 (真のTrade PF)
# -------------------------------------------------------------
with tab3:
    st.subheader("📜 1トレード単位 (Entry〜Exit) の純損益集計")
    st.caption("日次リターン合計ではなく、ポジション保有開始から全売却までの実トレード損益（往復コスト控除後）を集計しています。")
    t_c1, t_c2, t_c3 = st.columns(3)
    t_c1.metric("総完結トレード数", f"{metrics_strat['Total_Trades']} 件")
    t_c2.metric("真の Trade PF", f"{metrics_strat['Trade_PF']:.2f}")
    t_c3.metric("平均保有日数", f"{metrics_strat['Avg_Hold_Days']:.1f} 営業日")

    if len(all_trades) > 0:
        df_trades = pd.DataFrame(all_trades)
        df_trades["entry_date"] = pd.to_datetime(df_trades["entry_date"]).dt.strftime('%Y-%m-%d')
        df_trades["exit_date"] = pd.to_datetime(df_trades["exit_date"]).dt.strftime('%Y-%m-%d')
        df_trades["pnl_pct_str"] = df_trades["pnl_pct"].apply(lambda x: f"{x:+.2f} %")
        st.dataframe(
            df_trades[["asset", "entry_date", "exit_date", "holding_days", "pnl_pct_str"]].rename(columns={
                "asset": "銘柄", "entry_date": "買付日", "exit_date": "売却日",
                "holding_days": "保有日数", "pnl_pct_str": "純実現損益 (%)"
            }),
            use_container_width=True, height=400, hide_index=True
        )
    else:
        st.info("集計対象の完結トレードがありません。")

# -------------------------------------------------------------
# TAB 4: Purged Walk-Forward (OOS) 検証
# -------------------------------------------------------------
with tab4:
    st.subheader("🔄 Purged Walk-Forward (完全Out-of-Sample) 検証")
    st.markdown("本システムでは、シグナル生成時点で**学習データ（3年）と予測対象の間に5営業日のEmbargo**を設け、未学習データのみで逐次OOSリターンを生成しています。")

    wf_train_days = 252 * 3
    wf_test_days = 252 * 1
    total_len = len(strat_ret)

    if total_len > wf_train_days + wf_test_days:
        oos_rets, oos_dates, oos_splits = [], [], []
        step = wf_test_days

        for start in range(0, total_len - wf_train_days - wf_test_days + 1, step):
            test_start = start + wf_train_days
            test_end = test_start + wf_test_days
            seg = strat_ret[test_start:test_end]
            oos_rets.extend(seg)
            oos_dates.extend(common_idx[test_start:test_end])

            seg_cum = np.cumprod(1.0 + seg)
            seg_ret = (seg_cum[-1] - 1.0) * 100.0
            seg_dd = np.min((seg_cum - np.maximum.accumulate(seg_cum)) / np.maximum.accumulate(seg_cum)) * 100.0
            oos_splits.append({
                "検証期間": f"{common_idx[test_start].strftime('%Y/%m')} - {common_idx[test_end-1].strftime('%Y/%m')}",
                "OOS 累積リターン": f"{seg_ret:+.1f} %",
                "OOS MDD": f"{seg_dd:.1f} %"
            })

        cum_oos = np.cumprod(1.0 + np.array(oos_rets))
        years_oos = max(len(oos_rets) / TRADING_DAYS_PER_YEAR, 0.1)
        oos_cagr = (cum_oos[-1] ** (1.0 / years_oos)) - 1.0

        w_col1, w_col2 = st.columns([2, 1])
        with w_col1:
            fig_wf = go.Figure()
            fig_wf.add_trace(go.Scatter(x=oos_dates, y=cum_oos, name="厳密OOS 連結資産成長", line=dict(color="#e67e22", width=2)))
            fig_wf.update_layout(title="完全未学習期間 (OOS) 累積パフォーマンス", height=380, margin=dict(t=40, b=20, l=10, r=10))
            fig_wf.update_yaxes(type="log")
            st.plotly_chart(fig_wf, use_container_width=True)
        with w_col2:
            st.markdown("#### 🏆 厳密OOS 通算実績")
            st.metric("OOS 通算 CAGR", f"{oos_cagr*100:.1f}%")
            st.dataframe(pd.DataFrame(oos_splits), use_container_width=True, hide_index=True)

# -------------------------------------------------------------
# TAB 5: Block Bootstrap モンテカルロ
# -------------------------------------------------------------
with tab5:
    st.subheader("🎲 Circular Block Bootstrap (ボラティリティ連鎖保持)")
    st.caption("1日ごとの独立復元抽出ではなく、10営業日のブロック単位でリサンプリングし、ボラティリティ・クラスタリングを再現したファンチャートです。")

    block_size = 10
    n_sims = 500
    n_steps = min(252 * 5, len(strat_ret))
    sim_paths = np.zeros((n_sims, n_steps))
    base_r = strat_ret[-n_steps:]
    n_base = len(base_r)

    np.random.seed(42)
    for i in range(n_sims):
        sampled = []
        while len(sampled) < n_steps:
            rand_idx = np.random.randint(0, n_base)
            block = [base_r[(rand_idx + b) % n_base] for b in range(block_size)]
            sampled.extend(block)
        sim_paths[i] = np.cumprod(1.0 + np.array(sampled[:n_steps]))

    p5 = np.percentile(sim_paths, 5, axis=0)
    p25 = np.percentile(sim_paths, 25, axis=0)
    p50 = np.percentile(sim_paths, 50, axis=0)
    p75 = np.percentile(sim_paths, 75, axis=0)
    p95 = np.percentile(sim_paths, 95, axis=0)
    step_axis = np.arange(1, n_steps + 1)

    mc_c1, mc_c2 = st.columns([2, 1])
    with mc_c1:
        fig_mc = go.Figure()
        fig_mc.add_trace(go.Scatter(x=step_axis, y=p95, name="95% 楽観ケース", line=dict(color="rgba(46, 204, 113, 0.4)")))
        fig_mc.add_trace(go.Scatter(x=step_axis, y=p75, name="75% タイル", line=dict(color="rgba(52, 152, 219, 0.5)")))
        fig_mc.add_trace(go.Scatter(x=step_axis, y=p50, name="中央値 (Median)", line=dict(color="#f39c12", width=2.5)))
        fig_mc.add_trace(go.Scatter(x=step_axis, y=p25, name="25% タイル", line=dict(color="rgba(231, 76, 60, 0.5)")))
        fig_mc.add_trace(go.Scatter(x=step_axis, y=p5, name="5% 悲観ケース", line=dict(color="rgba(192, 57, 43, 0.4)")))
        fig_mc.update_layout(title="今後5年間の期待資産分布 (Block Bootstrap 500試行)", xaxis_title="営業日数", yaxis_title="資産倍率", height=400, margin=dict(t=40, b=20, l=10, r=10))
        st.plotly_chart(fig_mc, use_container_width=True)

    with mc_c2:
        st.markdown("#### ⚡ 取引摩擦ストレステスト")
        stress_list = []
        for test_bps in [0, 10, 20, 30, 50]:
            r_st, _, _, _ = run_rigorous_backtest(exec_alloc, o_etf_matrix, c_etf_matrix, fx_array, test_bps/10000.0, use_dd_controller, use_jpy)
            c_st = (np.cumprod(1.0 + r_st)[-1] ** (1.0 / max(len(r_st)/252, 0.1))) - 1.0
            stress_list.append({
                "片道コスト": f"{test_bps} bps", "CAGR": f"{c_st*100:.1f}%", "劣化幅": f"{(c_st - metrics_strat['CAGR'])*100:+.1f}%"
            })
        st.dataframe(pd.DataFrame(stress_list), use_container_width=True, hide_index=True)

# -------------------------------------------------------------
# TAB 6: 3目的 Pareto Frontier 探索 (全面改良版)
# -------------------------------------------------------------
with tab6:
    st.subheader("💎 3目的 Pareto Frontier (収益 vs MDD vs CVaR) 探索")

    # 1. 直感的な図の見方解説カード
    with st.expander("💡 30秒でわかる！この図の見方・選び方（クリックで開閉）", expanded=True):
        st.markdown("""
        **【グラフの重要ルール：一番優秀なのは『左上』にある点です】**
        * **縦軸（上に行くほど良い）**: 年平均リターン（CAGR）が高く、資産が増えるスピードが速い。
        * **横軸（左に行くほど良い）**: 最大下落率（MDD）が小さく、暴落時の痛手が浅い（資産が守られる）。
        * **丸の大きさ（大きいほど良い）**: 下落リスクに対する収益効率（Calmar比率）が高い。
        * **丸の色（濃い青・紫ほど安全）**: 1日の最大想定損失（日次95% CVaR）が小さく堅牢。

        > **迷ったときの設定選びの目安:**
        > * **「下落が怖い（最大損失20%以下）」**: 横軸20%より左側の中で、最も上にある点を選択。
        > * **「下落25%程度まで許容して収益を伸ばしたい」**: 横軸25%付近で最も上に位置する点を選択。
        """)

    if st.button("🚀 パレート探索を実行 (主要パラメータグリッドスキャン)"):
        with st.spinner("パラメータ空間を走査中..."):
            grid_results = []
            for scan_vol in [0.30, 0.40, 0.50]:
                for scan_cap in [0.35, 0.50, 0.70]:
                    for scan_dd in [True, False]:
                        scan_caps = np.minimum(caps_matrix, scan_cap)
                        scan_alloc = optimize_portfolio_flow(raw_matrix, scan_caps, ret_open_to_open, scan_vol, False)
                        s_exec = np.zeros_like(scan_alloc)
                        s_exec[1:] = scan_alloc[:-1]
                        r_sc, eff_w_sc, _, tr_sc = run_rigorous_backtest(s_exec, o_etf_matrix, c_etf_matrix, fx_array, fee_rate, scan_dd, use_jpy)
                        m_sc = evaluate_performance(r_sc, eff_w_sc, tr_sc)
                        grid_results.append({
                            "TargetVol": f"{int(scan_vol*100)}%",
                            "MaxCap": f"{int(scan_cap*100)}%",
                            "DD_Ctrl": "ON" if scan_dd else "OFF",
                            "設定ラベル": f"Vol {int(scan_vol*100)}% / 上限 {int(scan_cap*100)}% / DD:{'ON' if scan_dd else 'OFF'}",
                            "CAGR": m_sc["CAGR"] * 100.0,
                            "MDD": abs(m_sc["MDD"] * 100.0),
                            "CVaR_95": abs(m_sc["CVaR_95"] * 100.0),
                            "Calmar": m_sc["Calmar"]
                        })

            df_grid = pd.DataFrame(grid_results)

            # 3大おすすめ設定を自動特定
            best_balanced = df_grid.loc[df_grid["Calmar"].idxmax()]
            best_safe = df_grid.loc[df_grid["MDD"].idxmin()]
            best_growth = df_grid.loc[df_grid["CAGR"].idxmax()]

            st.markdown("### 🏆 目的別・推奨おすすめ3大設定")
            col_b1, col_b2, col_b3 = st.columns(3)
            with col_b1:
                st.success(
                    f"**① 総合バランス最優秀 (Calmar最大)**\n\n"
                    f"**【{best_balanced['設定ラベル']}】**\n\n"
                    f"* CAGR: **{best_balanced['CAGR']:.1f}%**\n"
                    f"* MDD: **-{best_balanced['MDD']:.1f}%**\n"
                    f"* Calmar比率: **{best_balanced['Calmar']:.2f}**"
                )
            with col_b2:
                st.info(
                    f"**② 最も安全重視 (下落最小)**\n\n"
                    f"**【{best_safe['設定ラベル']}】**\n\n"
                    f"* CAGR: **{best_safe['CAGR']:.1f}%**\n"
                    f"* MDD: **-{best_safe['MDD']:.1f}%** (最小リスク)\n"
                    f"* Calmar比率: **{best_safe['Calmar']:.2f}**"
                )
            with col_b3:
                st.warning(
                    f"**③ 最も収益重視 (リターン最大)**\n\n"
                    f"**【{best_growth['設定ラベル']}】**\n\n"
                    f"* CAGR: **{best_growth['CAGR']:.1f}%** (最高益)\n"
                    f"* MDD: **-{best_growth['MDD']:.1f}%**\n"
                    f"* Calmar比率: **{best_growth['Calmar']:.2f}**"
                )

            # 文字重なりを解消し、ホバーで詳細が浮かび上がる散布図
            fig_pareto = go.Figure()
            fig_pareto.add_trace(go.Scatter(
                x=df_grid["MDD"],
                y=df_grid["CAGR"],
                mode="markers",
                marker=dict(
                    size=np.clip(df_grid["Calmar"] * 9, 12, 36),
                    color=df_grid["CVaR_95"],
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="日次CVaR (%)"),
                    line=dict(width=1, color="black")
                ),
                text=df_grid["設定ラベル"],
                customdata=np.stack((df_grid["Calmar"], df_grid["CVaR_95"]), axis=-1),
                hovertemplate=(
                    "<b>%{text}</b><br><br>"
                    "年平均リターン (CAGR): %{y:.1f}%<br>"
                    "最大下落率 (MDD): -%{x:.1f}%<br>"
                    "Calmar比率: %{customdata[0]:.2f}<br>"
                    "日次95% CVaR: %{customdata[1]:.2f}%<extra></extra>"
                )
            ))

            # おすすめ3点をグラフ上に矢印ハイライト
            fig_pareto.add_annotation(
                x=best_balanced["MDD"], y=best_balanced["CAGR"],
                text="🏆 総合最優秀", showarrow=True, arrowhead=2,
                arrowsize=1, arrowwidth=2, arrowcolor="#2ecc71", ax=35, ay=-35
            )
            fig_pareto.add_annotation(
                x=best_safe["MDD"], y=best_safe["CAGR"],
                text="🛡️ 最安全", showarrow=True, arrowhead=2,
                arrowsize=1, arrowwidth=2, arrowcolor="#3498db", ax=-35, ay=-35
            )
            fig_pareto.add_annotation(
                x=best_growth["MDD"], y=best_growth["CAGR"],
                text="🚀 最高益", showarrow=True, arrowhead=2,
                arrowsize=1, arrowwidth=2, arrowcolor="#e67e22", ax=35, ay=35
            )

            fig_pareto.update_layout(
                title="Pareto 最適空間 (左上ほど優秀 ｜ 丸サイズ: Calmar比率)",
                xaxis_title="最大ドローダウン MDD (%) [← 左ほど下落が小さく安全]",
                yaxis_title="通算 CAGR (%) [↑ 上ほど収益が高い]",
                height=520,
                hovermode="closest"
            )
            st.plotly_chart(fig_pareto, use_container_width=True)

            st.markdown("### 📋 全パラメータ走査結果（Calmar比率 順）")
            st.dataframe(df_grid.sort_values(by="Calmar", ascending=False), use_container_width=True, hide_index=True)
    else:
        st.info("上のボタンを押すと、全18通りのパラメータ走査とPareto最適解の散布図が生成されます。")
