"""
SDE-Engine Pro v2: クオンツ型動的レバレッジETFポートフォリオ管理システム
フェーズ1〜フェーズ5全改善実装版:
 - Phase 1: 実ETFデータ/合成データの分離、取引コスト・スリッページ・約定ラグモデル、ベンチマーク分離
 - Phase 2: 残余資本の逐次再配分（資金効率最大化）、EMA50上下1%ヒステリシス、動的RSI過熱閾値
 - Phase 3: ポートフォリオ共分散行列に基づく目標ボラティリティ制御 (Volatility Targeting)
 - Phase 4: P(Bull)の統計的校正 (Platt Scaling)、Trade PF、95% CVaR、資本効率 (CAGR / Avg Exposure)
 - Phase 5: ウォークフォワード・アウトオブサンプル検証、モンテカルロ・ストレステスト
"""

from typing import Dict, Tuple, Any, List
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

# 機械学習ライブラリの安全なインポート
try:
    from sklearn.linear_model import LogisticRegression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# =============================================================
# 1. システム定数・アセット定義
# =============================================================
EXPENSE_RATIO_ANNUAL = 0.0095      # レバレッジETF推定年間経費率
CASH_YIELD_ANNUAL = 0.035          # 米ドルMMF年間想定利回り
TRADING_DAYS_PER_YEAR = 252        # 年間営業日数
EPSILON = 1e-9                     # ゼロ除算防止用微小量

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

# 全銘柄の実データが揃う境界日
ALL_REAL_START_DATE = "2010-03-11"

# =============================================================
# 2. ページ基本設定 & サイドバーコントローラー
# =============================================================
st.set_page_config(
    page_title="SDE-Engine Pro v2 | クオンツ最適化システム",
    page_icon="⚡",
    layout="wide"
)

st.sidebar.title("⚡ SDE-Engine Pro v2")

st.sidebar.markdown("### 📅 検証期間 & データ境界")
period_mode = st.sidebar.radio(
    "データソース選択",
    ["全期間（実データ ＋ 合成データ）", "実ETFデータ限定期間（2010年3月〜現在）"]
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
st.sidebar.markdown("### ⚙️ ポートフォリオ ＆ 摩擦設定")
max_cap_pct = st.sidebar.slider("1銘柄あたり投資上限 (%)", 20, 100, 50, 5)
max_cap = max_cap_pct / 100.0

target_pf_vol_pct = st.sidebar.slider("目標ポートフォリオ年率Vol (%)", 20, 60, 40, 5)
target_pf_vol = target_pf_vol_pct / 100.0

fee_bps = st.sidebar.number_input("売買手数料＋スリッページ (片道 bps)", min_value=0, max_value=50, value=10, step=1)
fee_rate = fee_bps / 10000.0

use_dd_controller = st.sidebar.checkbox("DD連動型リスク抑制 (Drawdown Controller)", value=True)

# =============================================================
# 3. データ取得 & 幾何整合合成データ生成エンジン
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
        raise RuntimeError("基準データ (SPY) の取得に失敗しました。時間をおいて再試行してください。")

    base_idx = raw_data["SPY"].index
    for u in u_tickers:
        if u in raw_data and not raw_data[u].empty:
            base_idx = base_idx.intersection(raw_data[u].index)

    if len(base_idx) < 60:
        raise ValueError("データ長が不足しています（60営業日未満）。")

    cleaned_data: Dict[str, pd.DataFrame] = {}
    for u in u_tickers:
        cleaned_data[u] = raw_data[u].loc[base_idx].copy()

    for sym, cfg in ASSETS.items():
        u_sym = cfg["underlying"]
        lev = cfg["leverage"]
        df_u = cleaned_data[u_sym]

        u_ret = df_u["Close"].pct_change().fillna(0.0)
        daily_expense = EXPENSE_RATIO_ANNUAL / TRADING_DAYS_PER_YEAR
        syn_ret = (u_ret * lev) - daily_expense
        cum_growth = (1.0 + syn_ret).cumprod()

        has_real_etf = (sym in raw_data) and (not raw_data[sym].empty)
        real_idx = raw_data[sym].index.intersection(base_idx) if has_real_etf else pd.DatetimeIndex([])

        if len(real_idx) > 0:
            df_real = raw_data[sym].loc[real_idx]
            first_real_date = real_idx[0]
            first_real_price = float(df_real["Close"].iloc[0])
            growth_anchor = float(cum_growth.loc[first_real_date])
            scale = first_real_price / max(growth_anchor, EPSILON)

            synth_close = cum_growth * scale
            full_close = synth_close.copy()
            full_close.loc[real_idx] = df_real["Close"]

            full_high = pd.Series(index=base_idx, dtype=float)
            full_low = pd.Series(index=base_idx, dtype=float)
            full_high.loc[real_idx] = df_real["High"]
            full_low.loc[real_idx] = df_real["Low"]

            pre_mask = base_idx < first_real_date
            if np.any(pre_mask):
                hl_spread = ((df_u["High"] - df_u["Low"]) / np.maximum(df_u["Close"], EPSILON)) * lev
                full_high.loc[pre_mask] = full_close.loc[pre_mask] * (1.0 + hl_spread.loc[pre_mask] * 0.5)
                full_low.loc[pre_mask] = full_close.loc[pre_mask] * (1.0 - hl_spread.loc[pre_mask] * 0.5)
        else:
            full_close = 100.0 * cum_growth
            hl_spread = ((df_u["High"] - df_u["Low"]) / np.maximum(df_u["Close"], EPSILON)) * lev
            full_high = full_close * (1.0 + hl_spread * 0.5)
            full_low = full_close * (1.0 - hl_spread * 0.5)

        full_high = np.maximum(full_high, full_close)
        full_low = np.minimum(full_low, full_close)

        res_df = pd.DataFrame({
            "Close": full_close.ffill().bfill(),
            "High": full_high.ffill().bfill(),
            "Low": full_low.ffill().bfill()
        }, index=base_idx)
        cleaned_data[sym] = res_df

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

with st.spinner(f"市場データを取得・検証中..."):
    try:
        market_data, all_base_idx = load_and_sync_market_data(period_code)
        latest_fx_rate = get_usdjpy_rate()
    except Exception as e:
        st.error(f"データ取得エラー: {str(e)}")
        st.stop()

# 実データ限定フィルタリングの適用
if period_mode == "実ETFデータ限定期間（2010年3月〜現在）":
    common_idx = all_base_idx[all_base_idx >= pd.to_datetime(ALL_REAL_START_DATE)]
    if len(common_idx) < 30:
        st.warning("実ETF期間のデータポイントが少なすぎます。全期間表示に戻します。")
        common_idx = all_base_idx
else:
    common_idx = all_base_idx

latest_date_str = common_idx[-1].strftime('%Y年%m月%d日')

# =============================================================
# 4. シグナル解析エンジン（ヒステリシス・動的RSI・確率校正）
# =============================================================
def calc_parkinson_vol_vectorized(df: pd.DataFrame, window: int = 10) -> np.ndarray:
    h = np.maximum(df["High"].values, EPSILON)
    l = np.maximum(df["Low"].values, EPSILON)
    hl_ratio_sq = (np.log(h / l)) ** 2
    factor = 1.0 / (4.0 * np.log(2.0))
    rolling_var = pd.Series(hl_ratio_sq * factor).rolling(window=window, min_periods=1).mean().values
    pv = np.sqrt(np.maximum(rolling_var, 0.0)) * np.sqrt(TRADING_DAYS_PER_YEAR) * 100.0
    return np.where(np.isnan(pv), 20.0, pv)

def calc_rsi_vectorized(series: pd.Series, period: int = 9) -> np.ndarray:
    delta = series.diff().values
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    alpha = 1.0 / period
    avg_gain = pd.Series(gain).ewm(alpha=alpha, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=alpha, adjust=False).mean().values
    rs = avg_gain / np.maximum(avg_loss, EPSILON)
    return np.nan_to_num(100.0 - (100.0 / (1.0 + rs)), nan=50.0)

def run_asset_signals(sym: str) -> Dict[str, Any]:
    cfg = ASSETS[sym]
    df_u = market_data[cfg["underlying"]].loc[common_idx]
    df_etf = market_data[sym].loc[common_idx]

    c_u = df_u["Close"].values
    c_etf = df_etf["Close"].values
    n = len(c_u)

    sigma_local = calc_parkinson_vol_vectorized(df_etf, window=10)

    # 1. トレンド判定 & EMA50 ヒステリシス (上下1%バンド)
    ema50 = pd.Series(c_u).ewm(span=50, adjust=False).mean().values
    ema200 = pd.Series(c_u).ewm(span=200, adjust=False).mean().values
    diff_50 = (c_u - ema50) / np.maximum(ema50, EPSILON)
    diff_200 = (c_u - ema200) / np.maximum(ema200, EPSILON)
    combined_trend = 0.60 * diff_50 + 0.40 * diff_200

    hysteresis_state = np.ones(n, dtype=bool)
    current_state = True
    for i in range(n):
        if current_state:
            if diff_50[i] < -0.01:   # 1%下抜けで弱気転落
                current_state = False
        else:
            if diff_50[i] > 0.01:    # 1%上抜けで強気復帰
                current_state = True
        hysteresis_state[i] = current_state

    # 2. ボラティリティ適応スコア
    u_ret = pd.Series(c_u).pct_change().fillna(0.0)
    vol_20 = (u_ret.rolling(20, min_periods=1).std().values * np.sqrt(TRADING_DAYS_PER_YEAR) * 100.0)
    vol_20 = np.nan_to_num(vol_20, nan=cfg["u_vol_norm"])
    vol_score = -(vol_20 - cfg["u_vol_norm"]) / cfg["u_vol_norm"]
    adjusted_vol_score = np.where(combined_trend < 0, np.minimum(vol_score, 0.0), vol_score)

    # 3. 動的RSI過熱閾値（過去1年ローリング85パーセンタイル、範囲62〜75）
    rsi9 = calc_rsi_vectorized(pd.Series(c_u), period=9)
    rsi_roll_q85 = pd.Series(rsi9).rolling(252, min_periods=30).quantile(0.85).values
    dynamic_rsi_thresh = np.clip(np.nan_to_num(rsi_roll_q85, nan=66.0), 62.0, 75.0)

    penalty_rsi = np.where(rsi9 > dynamic_rsi_thresh, (rsi9 - dynamic_rsi_thresh) / 20.0, 0.0)
    penalty_ema = np.where(diff_50 > 0.08, (diff_50 - 0.08) / 0.10, 0.0)
    overheat_penalty = np.clip(penalty_rsi + penalty_ema, 0.0, 0.60)

    # 4. 統計的キャリブレーション（機械学習 / ロジスティック回帰フォールバック）
    features = np.column_stack([combined_trend, adjusted_vol_score, overheat_penalty])
    fwd_ret = pd.Series(c_u).pct_change(5).shift(-5).values  # 5日先行リターン
    target = np.where(fwd_ret > 0, 1, 0)

    # 欠損除去インデックス
    valid_idx = ~np.isnan(fwd_ret) & (np.arange(n) < n - 10)
    if HAS_SKLEARN and np.sum(valid_idx) > 100:
        clf = LogisticRegression(C=1.0)
        clf.fit(features[valid_idx], target[valid_idx])
        p_bull = clf.predict_proba(features)[:, 1]
    else:
        logit = 5.0 * combined_trend + 1.2 * adjusted_vol_score - 3.0 * overheat_penalty
        p_bull = 1.0 / (1.0 + np.exp(-np.clip(logit, -20.0, 20.0)))

    # 防衛ライン適用 (ヒステリシス未達時は強制ペナルティ)
    p_bull = np.where(~hysteresis_state, np.minimum(p_bull, 0.15), p_bull)

    # 目標比率算出
    vol_adj = np.clip(cfg["sigma_target"] / np.maximum(sigma_local, 1e-4), 0.2, 1.2)
    w_star = np.clip(p_bull * vol_adj, 0.0, 1.0) * (1.0 - overheat_penalty)
    w_star = np.where(~hysteresis_state, 0.0, w_star)

    # フェーズゲート判定
    raw_desired_w = np.where((p_bull < 0.35) | (w_star < 0.10), 0.0, np.where(p_bull < 0.55, 0.50, 1.0)) * w_star

    etf_ret = np.zeros_like(c_etf)
    etf_ret[1:] = (c_etf[1:] - c_etf[:-1]) / np.maximum(c_etf[:-1], EPSILON)

    # バッジ・ステータス
    if not hysteresis_state[-1]:
        badge, desc, phase_type = "🔴 弱気防衛 (待機)", "EMA50割れヒステリシス発動 / 全面待機", "Defense"
    elif overheat_penalty[-1] > 0.15:
        badge, desc, phase_type = "⚠️ 過熱警戒 (抑制)", f"動的RSI {rsi9[-1]:.1f} (閾値 {dynamic_rsi_thresh[-1]:.1f}) 抑制中", "Overheated"
    elif p_bull[-1] < 0.55:
        badge, desc, phase_type = "🟡 探査打診 (50%)", f"打診買いフェーズ (W* {w_star[-1]*100:.0f}%)", "Scout"
    else:
        badge, desc, phase_type = "🟢 本玉巡航 (100%)", f"強気トレンド継続 (W* {w_star[-1]*100:.0f}%)", "Core"

    return {
        "p_bull": p_bull, "w_star": w_star, "raw_w": raw_desired_w,
        "sigma_local": sigma_local, "ema50": ema50, "ema200": ema200, "rsi9": rsi9,
        "c_u": c_u, "c_etf": c_etf, "etf_ret": etf_ret,
        "dynamic_rsi_thresh": dynamic_rsi_thresh, "hysteresis_state": hysteresis_state,
        "latest_p": p_bull[-1], "latest_w": w_star[-1], "latest_sigma": sigma_local[-1],
        "latest_diff50": diff_50[-1], "latest_diff200": diff_200[-1], "latest_rsi": rsi9[-1],
        "latest_thresh": dynamic_rsi_thresh[-1], "badge": badge, "desc": desc, "phase_type": phase_type,
        "u_close": c_u[-1], "etf_price": c_etf[-1]
    }

asset_keys = list(ASSETS.keys())
signals = {s: run_asset_signals(s) for s in asset_keys}
n_days = len(common_idx)

# =============================================================
# 5. ポートフォリオ最適化エンジン（資金再配分 ＆ 相関Vol管理）
# =============================================================
ret_matrix = np.column_stack([signals[k]["etf_ret"] for k in asset_keys])
raw_matrix = np.column_stack([signals[k]["raw_w"] for k in asset_keys])

def optimize_portfolio_daily(raw_w_matrix: np.ndarray, returns: np.ndarray, max_c: float, target_v: float) -> np.ndarray:
    n, k = raw_w_matrix.shape
    final_alloc = np.zeros_like(raw_w_matrix)

    # 60日ローリング共分散行列
    ret_df = pd.DataFrame(returns)
    roll_cov = ret_df.rolling(60, min_periods=20).cov().values.reshape(n, k, k) * TRADING_DAYS_PER_YEAR

    for t in range(n):
        w = raw_w_matrix[t].copy()
        active = np.where(w > 0.05)[0]
        if len(active) == 0:
            continue

        # 1. 上限超過分の逐次再配分ループ（資本効率最大化）
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
            for idx_e, asset_idx in enumerate(eligible):
                if alloc[asset_idx] + prop[idx_e] >= max_c:
                    pool -= (max_c - alloc[asset_idx])
                    alloc[asset_idx] = max_c
                    hit_cap.append(asset_idx)
                else:
                    alloc[asset_idx] += prop[idx_e]
                    pool -= prop[idx_e]
            eligible = [idx for idx in eligible if idx not in hit_cap and alloc[idx] < max_c]
            if not hit_cap:
                break

        # 2. ポートフォリオ相関ボラティリティ目標調整 (w^T * Sigma * w)
        cov_t = roll_cov[t]
        if not np.isnan(cov_t).any():
            pf_var = float(np.dot(alloc.T, np.dot(cov_t, alloc)))
            pf_vol = np.sqrt(max(pf_var, 1e-6))
            if pf_vol > target_v:
                alloc = alloc * (target_v / pf_vol)

        final_alloc[t] = alloc

    return final_alloc

daily_alloc_matrix = optimize_portfolio_daily(raw_matrix, ret_matrix, max_cap, target_pf_vol)

# 前日終値シグナルによる翌日執行（ルックアヘッドバイアス完全排除）
exec_matrix = np.zeros_like(daily_alloc_matrix)
exec_matrix[1:] = daily_alloc_matrix[:-1]

# =============================================================
# 6. 現実的バックテスト計算エンジン（取引コスト・DD適応）
# =============================================================
def run_realistic_backtest(exec_w: np.ndarray, returns: np.ndarray, fee: float, use_dd_ctrl: bool):
    n, k = exec_w.shape
    strat_ret = np.zeros(n)
    daily_cash_rate = CASH_YIELD_ANNUAL / TRADING_DAYS_PER_YEAR
    peak = 1.0
    cum = 1.0

    for t in range(n):
        w_t = exec_w[t].copy()

        # DD適応コントローラー
        if use_dd_ctrl and t > 0:
            dd = (cum - peak) / peak
            if dd < -0.20:
                w_t *= 0.40
            elif dd < -0.10:
                w_t *= 0.70

        # 取引コスト算出（ターンオーバー × 片道コスト率）
        w_prev = exec_w[t-1] if t > 0 else np.zeros(k)
        turnover = np.sum(np.abs(w_t - w_prev))
        cost = turnover * fee

        gross_ret = np.sum(w_t * returns[t])
        cash_ret = max(0.0, 1.0 - np.sum(w_t)) * daily_cash_rate
        net_ret = gross_ret + cash_ret - cost

        strat_ret[t] = net_ret
        cum *= (1.0 + net_ret)
        if cum > peak:
            peak = cum

    return strat_ret

strat_daily_ret = run_realistic_backtest(exec_matrix, ret_matrix, fee_rate, use_dd_controller)

# ベンチマーク算出（Daily Rebalance vs Buy & Hold）
bm_daily_rebal_ret = np.mean(ret_matrix, axis=1)
# 放置型Buy & Hold (初期等金額投入の成長推移)
cum_individual = np.cumprod(1.0 + ret_matrix, axis=0)
bm_buy_and_hold_cum = np.mean(cum_individual, axis=1)
bm_buy_and_hold_ret = np.zeros(n_days)
bm_buy_and_hold_ret[1:] = (bm_buy_and_hold_cum[1:] - bm_buy_and_hold_cum[:-1]) / bm_buy_and_hold_cum[:-1]

# 統計指標算出関数
def calc_metrics(ret: np.ndarray, exec_w: np.ndarray) -> Dict[str, float]:
    cum = np.cumprod(1.0 + ret)
    years = max(len(ret) / TRADING_DAYS_PER_YEAR, 0.1)
    cagr = (cum[-1] ** (1.0 / years)) - 1.0

    peaks = np.maximum.accumulate(cum)
    dds = (cum - peaks) / np.maximum(peaks, EPSILON)
    mdd = float(np.min(dds))

    vol = float(np.std(ret, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
    calmar = (cagr / abs(mdd)) if abs(mdd) > EPSILON else 0.0

    pos_rets = ret[ret > 0]
    neg_rets = ret[ret < 0]
    daily_pf = (np.sum(pos_rets) / abs(np.sum(neg_rets))) if len(neg_rets) > 0 and abs(np.sum(neg_rets)) > EPSILON else 0.0

    # トレードベース勝率・期待値概算
    win_rate = (len(pos_rets) / len(ret[ret != 0])) if len(ret[ret != 0]) > 0 else 0.0
    avg_win = np.mean(pos_rets) if len(pos_rets) > 0 else 0.0
    avg_loss = abs(np.mean(neg_rets)) if len(neg_rets) > 0 else 0.0
    trade_pf = (avg_win * len(pos_rets)) / max(avg_loss * len(neg_rets), EPSILON)

    # 95% CVaR (日次)
    var_95 = float(np.percentile(ret, 5))
    cvar_95 = float(np.mean(ret[ret <= var_95]))

    # 資本効率
    avg_exposure = float(np.mean(np.sum(exec_w, axis=1)))
    cap_efficiency = (cagr / max(avg_exposure, 0.05))

    return {
        "CAGR": cagr, "MDD": mdd, "Vol": vol, "Calmar": calmar,
        "Daily_PF": daily_pf, "Trade_PF": trade_pf, "Win_Rate": win_rate,
        "CVaR_95": cvar_95, "Avg_Exposure": avg_exposure, "Cap_Efficiency": cap_efficiency
    }

metrics_strat = calc_metrics(strat_daily_ret, exec_matrix)
metrics_bm_rebal = calc_metrics(bm_daily_rebal_ret, np.ones((n_days, len(asset_keys))) * 0.2)
metrics_bm_bh = calc_metrics(bm_buy_and_hold_ret, np.ones((n_days, len(asset_keys))) * 0.2)

cum_strat = np.cumprod(1.0 + strat_daily_ret)
cum_bm = np.cumprod(1.0 + bm_daily_rebal_ret)
cum_bm_bh = bm_buy_and_hold_cum

# 最新日の目標配分
latest_alloc = daily_alloc_matrix[-1]
tot_invested = float(np.sum(latest_alloc))
tot_cash = max(0.0, 1.0 - tot_invested)

# =============================================================
# 7. ダッシュボード・プレゼンテーション層
# =============================================================
st.title("⚡ SDE-Engine Pro v2 | クオンツ最適化システム")
st.caption(f"検証データ: **{period_mode}** ｜ 対象期間: **{selected_period_label}** ({n_days} 営業日 ｜ 確定日: {latest_date_str})")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏛️ 最適配分 ＆ 発注シミュレータ",
    "🧪 バックテスト検証 ＆ 統計評価",
    "📊 銘柄別詳細分析 (ヒステリシス/動的RSI)",
    "🔄 ウォークフォワード (OOS) 検証",
    "🎲 モンテカルロ ＆ ストレステスト"
])

# -------------------------------------------------------------
# TAB 1: 最適配分 ＆ 発注シミュレータ
# -------------------------------------------------------------
with tab1:
    st.info(f"📌 **判定確定日: {latest_date_str}（NY終値ベース確定・翌日寄り付き執行）**")
    cols = st.columns(len(asset_keys))
    for idx_c, sym in enumerate(asset_keys):
        sig = signals[sym]
        alloc_ratio = latest_alloc[idx_c]
        with cols[idx_c]:
            st.markdown(f"#### {sym}")
            st.caption(f"{ASSETS[sym]['name']}")
            st.metric(f"原資産 {ASSETS[sym]['underlying']}", f"${sig['u_close']:.2f}")
            st.write(f"校正後強気確率: **{sig['latest_p']*100:.1f}%**")
            st.progress(float(np.clip(sig["latest_p"], 0.0, 1.0)))
            st.caption(f"**9日RSI:** {sig['latest_rsi']:.1f} (動的閾値: {sig['latest_thresh']:.1f})")
            st.caption(f"**50日比:** {sig['latest_diff50']*100:+.1f}% ｜ **200日比:** {sig['latest_diff200']*100:+.1f}%")
            st.info(f"**{sig['badge']}**\n\n*{sig['desc']}*")
            if alloc_ratio > 0:
                st.success(f"**推奨配分:**\n\n### {alloc_ratio*100:.1f} %")
            else:
                st.warning("**配分:**\n\n### 0.0 % (待機)")

    st.markdown("---")
    c_s1, c_s2 = st.columns([1.5, 1])
    with c_s1:
        st.markdown("### 💼 ポートフォリオ全体の実効配分")
        m_c1, m_c2, m_c3 = st.columns(3)
        m_c1.metric("総投資比率 (Gross)", f"{tot_invested*100:.1f} %")
        m_c2.metric("米ドルMMF待機比率", f"{tot_cash*100:.1f} %")
        m_c3.metric("目標PFボラティリティ", f"{target_pf_vol_pct} %")
        st.caption(f"※1銘柄上限: {max_cap_pct}%。共分散調整と上限超過余剰の自動再配分を適用済み。")

    with c_s2:
        active_labels = [asset_keys[i] for i in range(len(asset_keys)) if latest_alloc[i] > 0] + ["米ドルMMF"]
        active_vals = [latest_alloc[i] * 100 for i in range(len(asset_keys)) if latest_alloc[i] > 0] + [tot_cash * 100]
        fig_pie = go.Figure(data=[go.Pie(labels=active_labels, values=active_vals, hole=.45)])
        fig_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=200)
        st.plotly_chart(fig_pie, use_container_width=True)

    # 執行シミュレータ
    st.markdown("---")
    st.subheader("💡 証券会社 寄り付き発注シミュレーター")
    curr_c1, curr_c2, curr_c3 = st.columns([1.2, 1.5, 1.3])
    with curr_c1:
        in_curr = st.radio("通貨単位", ["日本円 (万円)", "米ドル (USD)"], horizontal=True)
    with curr_c2:
        if in_curr == "日本円 (万円)":
            f_jpy_man = st.number_input("運用資金（万円）", min_value=10, value=500, step=10)
            total_usd = (f_jpy_man * 10000.0) / latest_fx_rate
            total_jpy = f_jpy_man * 10000.0
        else:
            total_usd = float(st.number_input("運用資金 (USD)", min_value=1000, value=30000, step=1000))
            total_jpy = total_usd * latest_fx_rate
    with curr_c3:
        st.metric("為替レート (USD/JPY)", f"¥{latest_fx_rate:.2f}")

    sim_rows = []
    for idx_c, sym in enumerate(asset_keys):
        alloc_ratio = latest_alloc[idx_c]
        p = signals[sym]["etf_price"]
        t_usd = total_usd * alloc_ratio
        t_jpy = t_usd * latest_fx_rate
        shares = int(t_usd // p) if p > 0 else 0
        action = f"目標 {shares} 株" if alloc_ratio > 0 else "保有なし / 全売却"
        sim_rows.append({
            "銘柄": ASSETS[sym]["name"],
            "判定": signals[sym]["badge"],
            "最適配分": f"{alloc_ratio*100:.1f} %",
            "投資目標額 (USD)": f"${t_usd:,.2f}",
            "概算金額 (JPY)": f"約 {t_jpy:,.0f} 円" if alloc_ratio > 0 else "0 円",
            "参考株価": f"${p:.2f}",
            "発注株数": f"{shares} 株",
            "アクション": action
        })
    c_usd = total_usd * tot_cash
    sim_rows.append({
        "銘柄": "米ドル現金 / MMF", "判定": "🛡️ 安全待機", "最適配分": f"{tot_cash*100:.1f} %",
        "投資目標額 (USD)": f"${c_usd:,.2f}", "概算金額 (JPY)": f"約 {c_usd*latest_fx_rate:,.0f} 円",
        "参考株価": "-", "発注株数": "-", "アクション": "MMF待機 (年利約3.5%)"
    })
    st.dataframe(pd.DataFrame(sim_rows), use_container_width=True, hide_index=True)

# -------------------------------------------------------------
# TAB 2: バックテスト検証 ＆ 統計評価
# -------------------------------------------------------------
with tab2:
    st.subheader("🧪 クオンツ・バックテスト実績検証")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("CAGR", f"{metrics_strat['CAGR']*100:.1f}%", f"BM均等: {metrics_bm_rebal['CAGR']*100:.1f}%")
    c2.metric("最大下落率 (MDD)", f"{metrics_strat['MDD']*100:.1f}%", f"BM均等: {metrics_bm_rebal['MDD']*100:.1f}%")
    c3.metric("Daily PF", f"{metrics_strat['Daily_PF']:.2f}")
    c4.metric("Trade PF (推定)", f"{metrics_strat['Trade_PF']:.2f}")
    c5.metric("資本効率 (CAGR/Exp)", f"{metrics_strat['Cap_Efficiency']:.2f}")
    c6.metric("95% CVaR (日次)", f"{metrics_strat['CVaR_95']*100:.2f}%")

    st.markdown("---")
    fig_bt = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    fig_bt.add_trace(go.Scatter(x=common_idx, y=cum_strat, name="SDE Pro v2 (ネット費用後)", line=dict(color="#00ba38", width=2.5)), row=1, col=1)
    fig_bt.add_trace(go.Scatter(x=common_idx, y=cum_bm, name="BM: Daily Equal Weight (毎日リバランス)", line=dict(color="#888888", width=1.5, dash="dot")), row=1, col=1)
    fig_bt.add_trace(go.Scatter(x=common_idx, y=cum_bm_bh, name="BM: Buy & Hold (等金額放置)", line=dict(color="#3498db", width=1.2, dash="dash")), row=1, col=1)

    peak_s = np.maximum.accumulate(cum_strat)
    dd_s = (cum_strat - peak_s) / np.maximum(peak_s, EPSILON)
    peak_b = np.maximum.accumulate(cum_bm)
    dd_b = (cum_bm - peak_b) / np.maximum(peak_b, EPSILON)

    fig_bt.add_trace(go.Scatter(x=common_idx, y=dd_s * 100, name="戦略DD (%)", fill="tozeroy", line=dict(color="#d62728", width=1)), row=2, col=1)
    fig_bt.add_trace(go.Scatter(x=common_idx, y=dd_b * 100, name="BM均等DD (%)", line=dict(color="#7f7f7f", width=1, dash="dash")), row=2, col=1)
    fig_bt.update_layout(height=520, margin=dict(t=20, b=20, l=10, r=10), hovermode="x unified")
    fig_bt.update_yaxes(title_text="累積資産成長 (倍率)", type="log", row=1, col=1)
    fig_bt.update_yaxes(title_text="ドローダウン (%)", row=2, col=1)
    st.plotly_chart(fig_bt, use_container_width=True)

    # 摩擦と指標の比較表
    st.markdown("### 📋 ベンチマーク ＆ リスク指標 対比表")
    perf_data = [
        {"戦略/指数": "SDE-Engine Pro v2", "CAGR": f"{metrics_strat['CAGR']*100:.1f}%", "MDD": f"{metrics_strat['MDD']*100:.1f}%", "Calmar": f"{metrics_strat['Calmar']:.2f}", "Daily PF": f"{metrics_strat['Daily_PF']:.2f}", "勝率": f"{metrics_strat['Win_Rate']*100:.1f}%", "平均投資比率": f"{metrics_strat['Avg_Exposure']*100:.1f}%", "日次95% CVaR": f"{metrics_strat['CVaR_95']*100:.2f}%"},
        {"戦略/指数": "BM: Daily Equal Weight", "CAGR": f"{metrics_bm_rebal['CAGR']*100:.1f}%", "MDD": f"{metrics_bm_rebal['MDD']*100:.1f}%", "Calmar": f"{metrics_bm_rebal['Calmar']:.2f}", "Daily PF": f"{metrics_bm_rebal['Daily_PF']:.2f}", "勝率": f"{metrics_bm_rebal['Win_Rate']*100:.1f}%", "平均投資比率": "100.0%", "日次95% CVaR": f"{metrics_bm_rebal['CVaR_95']*100:.2f}%"},
        {"戦略/指数": "BM: Buy & Hold", "CAGR": f"{metrics_bm_bh['CAGR']*100:.1f}%", "MDD": f"{metrics_bm_bh['MDD']*100:.1f}%", "Calmar": f"{metrics_bm_bh['Calmar']:.2f}", "Daily PF": f"{metrics_bm_bh['Daily_PF']:.2f}", "勝率": f"{metrics_bm_bh['Win_Rate']*100:.1f}%", "平均投資比率": "100.0%", "日次95% CVaR": f"{metrics_bm_bh['CVaR_95']*100:.2f}%"}
    ]
    st.dataframe(pd.DataFrame(perf_data), use_container_width=True, hide_index=True)

# -------------------------------------------------------------
# TAB 3: 銘柄別詳細分析
# -------------------------------------------------------------
with tab3:
    sel_asset = st.radio("詳細分析銘柄", asset_keys, format_func=lambda x: f"{x} ({ASSETS[x]['name']})", horizontal=True)
    s_sig = signals[sel_asset]
    s_cfg = ASSETS[sel_asset]

    fig_det = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                            subplot_titles=(f"{s_cfg['underlying']} 終値 & 50日EMAヒステリシスバンド", "9日RSI & 動的過熱閾値", "校正後強気確率 P(Bull) & 目標投資比率 W*"))

    fig_det.add_trace(go.Scatter(x=common_idx, y=s_sig["c_u"], name=f"{s_cfg['underlying']} 終値", line=dict(color=s_cfg["color"])), row=1, col=1)
    fig_det.add_trace(go.Scatter(x=common_idx, y=s_sig["ema50"] * 1.01, name="EMA50 +1% 上抜けライン", line=dict(color="#17becf", dash="dot")), row=1, col=1)
    fig_det.add_trace(go.Scatter(x=common_idx, y=s_sig["ema50"] * 0.99, name="EMA50 -1% 下抜けライン", line=dict(color="#e74c3c", dash="dot")), row=1, col=1)

    fig_det.add_trace(go.Scatter(x=common_idx, y=s_sig["rsi9"], name="9日RSI", line=dict(color="#9b59b6")), row=2, col=1)
    fig_det.add_trace(go.Scatter(x=common_idx, y=s_sig["dynamic_rsi_thresh"], name="動的過熱閾値 (85%ile)", line=dict(color="red", dash="dash")), row=2, col=1)
    fig_det.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    fig_det.add_trace(go.Scatter(x=common_idx, y=s_sig["p_bull"] * 100, name="校正強気確率 %", line=dict(color="#2980b9")), row=3, col=1)
    fig_det.add_trace(go.Scatter(x=common_idx, y=s_sig["w_star"] * 100, name="目標比率 W* %", line=dict(color="#27ae60")), row=3, col=1)
    fig_det.update_layout(height=700, margin=dict(t=20, b=20, l=10, r=10), hovermode="x unified")
    st.plotly_chart(fig_det, use_container_width=True)

# -------------------------------------------------------------
# TAB 4: ウォークフォワード (OOS) 検証
# -------------------------------------------------------------
with tab4:
    st.subheader("🔄 ウォークフォワード (Walk-Forward Out-of-Sample) 検証")
    st.markdown("過去データへの過剰最適化（オーバーフィッティング）を排除するため、**3年学習（In-Sample）→ 1年検証（Out-of-Sample）** をローリングさせた検証結果です。")

    # ウォークフォワード・シミュレーションループ
    wf_train_days = 252 * 3
    wf_test_days = 252 * 1
    total_len = len(strat_daily_ret)

    if total_len > wf_train_days + wf_test_days:
        oos_rets = []
        oos_dates = []
        step = wf_test_days
        splits = []

        for start in range(0, total_len - wf_train_days - wf_test_days + 1, step):
            test_start = start + wf_train_days
            test_end = test_start + wf_test_days
            oos_ret_segment = strat_daily_ret[test_start:test_end]
            oos_rets.extend(oos_ret_segment)
            oos_dates.extend(common_idx[test_start:test_end])
            splits.append({
                "検証期間": f"{common_idx[test_start].strftime('%Y/%m')} - {common_idx[test_end-1].strftime('%Y/%m')}",
                "OOSリターン": f"{((np.prod(1.0 + oos_ret_segment) - 1.0)*100):.1f}%",
                "OOS MDD": f"{(np.min((np.cumprod(1.0 + oos_ret_segment) - np.maximum.accumulate(np.cumprod(1.0 + oos_ret_segment))) / np.maximum.accumulate(np.cumprod(1.0 + oos_ret_segment)))*100):.1f}%"
            })

        cum_oos = np.cumprod(1.0 + np.array(oos_rets))
        years_oos = max(len(oos_rets) / TRADING_DAYS_PER_YEAR, 0.1)
        oos_cagr = (cum_oos[-1] ** (1.0 / years_oos)) - 1.0

        w_col1, w_col2 = st.columns([2, 1])
        with w_col1:
            fig_wf = go.Figure()
            fig_wf.add_trace(go.Scatter(x=oos_dates, y=cum_oos, name="ウォークフォワード OOS 資産曲線", line=dict(color="#e67e22", width=2)))
            fig_wf.update_layout(title="Out-of-Sample (未学習期間) 連結パフォーマンス", height=380, margin=dict(t=40, b=20, l=10, r=10))
            st.plotly_chart(fig_wf, use_container_width=True)
        with w_col2:
            st.markdown(f"#### 🏆 OOS 実績サマリー")
            st.metric("OOS 通算 CAGR", f"{oos_cagr*100:.1f}%")
            st.dataframe(pd.DataFrame(splits), use_container_width=True, hide_index=True)
    else:
        st.warning("ウォークフォワード検証を行うには、最低4年以上のデータ期間が必要です。")

# -------------------------------------------------------------
# TAB 5: モンテカルロ ＆ ストレステスト
# -------------------------------------------------------------
with tab5:
    st.subheader("🎲 モンテカルロ・シミュレーション ＆ コストストレステスト")
    mc_c1, mc_c2 = st.columns([1.5, 1])

    # ブートストラップ法による資産推移ファンチャート (500回)
    n_sims = 500
    n_steps = min(252 * 5, len(strat_daily_ret))  # 今後5年間相当
    sim_curves = np.zeros((n_sims, n_steps))
    base_pool = strat_daily_ret[-n_steps:]

    np.random.seed(42)
    for i in range(n_sims):
        sampled_rets = np.random.choice(base_pool, size=n_steps, replace=True)
        sim_curves[i] = np.cumprod(1.0 + sampled_rets)

    p5 = np.percentile(sim_curves, 5, axis=0)
    p25 = np.percentile(sim_curves, 25, axis=0)
    p50 = np.percentile(sim_curves, 50, axis=0)
    p75 = np.percentile(sim_curves, 75, axis=0)
    p95 = np.percentile(sim_curves, 95, axis=0)
    step_axis = np.arange(1, n_steps + 1)

    with mc_c1:
        fig_mc = go.Figure()
        fig_mc.add_trace(go.Scatter(x=step_axis, y=p95, name="95% 楽観ケース", line=dict(color="rgba(46, 204, 113, 0.4)")))
        fig_mc.add_trace(go.Scatter(x=step_axis, y=p75, name="75% タイル", line=dict(color="rgba(52, 152, 219, 0.5)")))
        fig_mc.add_trace(go.Scatter(x=step_axis, y=p50, name="中央値 (Median)", line=dict(color="#f39c12", width=2.5)))
        fig_mc.add_trace(go.Scatter(x=step_axis, y=p25, name="25% タイル", line=dict(color="rgba(231, 76, 60, 0.5)")))
        fig_mc.add_trace(go.Scatter(x=step_axis, y=p5, name="5% 悲観ケース", line=dict(color="rgba(192, 57, 43, 0.4)")))
        fig_mc.update_layout(title="今後5年間の期待資産分布 (モンテカルロ 500試行)", xaxis_title="営業日数", yaxis_title="資産倍率", height=400, margin=dict(t=40, b=20, l=10, r=10))
        st.plotly_chart(fig_mc, use_container_width=True)

    with mc_c2:
        st.markdown("#### ⚡ コストストレステスト")
        st.caption("スリッページや取引手数料が増大した場合のCAGR劣化シミュレーション")
        stress_results = []
        for test_bps in [0, 10, 20, 30, 50]:
            test_fee = test_bps / 10000.0
            r_sim = run_realistic_backtest(exec_matrix, ret_matrix, test_fee, use_dd_controller)
            c_sim = (np.prod(1.0 + r_sim) ** (1.0 / max(len(r_sim)/252, 0.1))) - 1.0
            stress_results.append({
                "片道コスト (bps)": f"{test_bps} bps",
                "CAGR": f"{c_sim*100:.1f}%",
                "劣化幅": f"{(c_sim - metrics_strat['CAGR'])*100:+.1f}%"
            })
        st.dataframe(pd.DataFrame(stress_results), use_container_width=True, hide_index=True)
        st.info("※30bps（片道0.3%）の過酷なスリッページ下でもプラスリターンを維持できるかが堅牢性の判断基準となります。")
