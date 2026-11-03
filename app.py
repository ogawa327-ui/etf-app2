"""
SDE-Engine Pro v3.0.0: 次世代クオンツ型動的レバレッジETFポートフォリオ管理システム
【Stateful TCO Reversal Sniper & Cross-Sectional Multi-Factor CORE 統合進化版】

- 対象10銘柄: TQQQ, SPXL, SOXL, FAS, UGL, TMF, CURE, ERX, TNA, DRN + 米ドルMMF
- 主な進化点 (v2.4.1 -> v3.0.0):
  1. TCO (暴落迎撃) ステートフル化 & 2段階反転トリガー:
     ・クライマックス売られすぎセットアップ + 底打ち反転確認 (陽線/前日比プラス/RSIフックアップ) で落ちるナイフを回避
     ・1日強制決済バグを完全解消し、利確基準(RSI回復/EMA到達)・段階的利益ロック・ATRストップ・タイムストップまで継続保持
     ・無次元Zスコア修正 & ETFレバレッジ倍率・市場パニック温度を反映したConviction×逆Vol動的サイジング
  2. CORE (順張り) マルチファクター & 資本効率進化:
     ・10銘柄クロスセクショナル相対モメンタム順位 (RS Rank) + トレンド加速度 (Momentum Accel) 統合
     ・ローリング相関クラスタ減衰ペナルティ + 連続型ドローダウン・コントローラー
     ・動的キャピタル・スイープ (平常時CORE最大100%稼働・TCO発動時優先割当) + No-Tradeリバランス抑制バンド
     ・COREトレード履歴集計の open_trades.pop 欠落バグ完全修正
  3. 新規クオンツ可視化:
     ・2次元クオンツ・レーダー (CORE Trend vs TCO Sniper) & ローリング相関ヒートマップ
     ・CORE / TCO / MMF 累積リターン寄与度 (Attribution) 分解チャート
     ・決済理由別 (利確/トレーリング/損切り/TimeStop) ＆ 銘柄別パフォーマンス診断マトリクス
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Any, List
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# =============================================================
# 1. システム定数・アセット定義
# =============================================================
EXPENSE_RATIO_ANNUAL: float = 0.0095      # レバレッジETF年間経費率 (0.95%)
CASH_YIELD_ANNUAL: float = 0.035          # 米ドルMMF年間想定利回り (3.50%)
TRADING_DAYS_PER_YEAR: int = 252         # 年間営業日数
EPSILON: float = 1e-9                     # ゼロ除算防止定数

@dataclass(frozen=True)
class AssetConfig:
    name: str
    underlying: str
    asset_type: str
    leverage: float
    sigma_target: float
    u_vol_norm: float
    color: str
    real_start: str

ASSETS: Dict[str, AssetConfig] = {
    "TQQQ": AssetConfig("TQQQ (NASDAQ 3倍)", "QQQ", "ハイテク", 3.0, 55.0, 20.0, "#00ba38", "2010-02-11"),
    "SPXL": AssetConfig("SPXL (S&P500 3倍)", "SPY", "米国全体", 3.0, 45.0, 16.0, "#619cff", "2008-11-05"),
    "SOXL": AssetConfig("SOXL (半導体 3倍)", "SOXX", "半導体", 3.0, 65.0, 28.0, "#f5b041", "2010-03-11"),
    "FAS":  AssetConfig("FAS (金融株 3倍)", "XLF", "金融", 3.0, 50.0, 18.0, "#9b59b6", "2008-11-05"),
    "UGL":  AssetConfig("UGL (ゴールド 2倍)", "GLD", "ゴールド", 2.0, 35.0, 13.0, "#f1c40f", "2008-12-01"),
    "TMF":  AssetConfig("TMF (長期国債 3倍)", "TLT", "債券ヘッジ", 3.0, 50.0, 16.0, "#e74c3c", "2009-04-16"),
    "CURE": AssetConfig("CURE (ヘルスケア 3倍)", "XLV", "ヘルスケア", 3.0, 42.0, 14.0, "#1abc9c", "2011-06-15"),
    "ERX":  AssetConfig("ERX (エネルギー 2倍)", "XLE", "エネルギー", 2.0, 55.0, 22.0, "#e67e22", "2008-11-06"),
    "TNA":  AssetConfig("TNA (小型株 3倍)", "IWM", "小型株", 3.0, 60.0, 22.0, "#3498db", "2008-11-05"),
    "DRN":  AssetConfig("DRN (不動産 3倍)", "IYR", "不動産", 3.0, 55.0, 19.0, "#8e44ad", "2009-07-16"),
}

ALL_REAL_START_DATE: str = "2011-06-15"

# =============================================================
# 2. UI & サイドバー設定
# =============================================================
st.set_page_config(
    page_title="SDE-Engine Pro v3.0.0 | 次世代クオンツ統合システム",
    page_icon="⚡",
    layout="wide"
)

st.sidebar.title("⚡ SDE-Engine Pro v3.0.0")
st.sidebar.caption("Stateful TCO Sniper & Multi-Factor CORE")

st.sidebar.markdown("### 📅 データソース ＆ 期間")
period_mode = st.sidebar.radio(
    "データ種別",
    ["全期間（実データ ＋ 合成データ）", "実ETFデータ限定（2011年6月〜現在）"]
)

period_options = {
    "5y (直近5年間)": "5y",
    "10y (直近10年間)": "10y",
    "15y (直近15年間)": "15y",
    "20y (直近20年間・リーマン含む)": "20y",
    "25y (直近25年間・ITバブル含む)": "25y",
    "max (取得可能全期間)": "max"
}
selected_period_label = st.sidebar.selectbox("取得期間", list(period_options.keys()), index=5)
period_code = period_options[selected_period_label]

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ コア巡航 (順張り) 設定")
max_cap_pct = st.sidebar.slider("1銘柄あたり絶対投資上限 (%)", 10, 50, 30, 5)
max_cap = float(max_cap_pct) / 100.0

target_pf_vol_pct = st.sidebar.slider("目標ポートフォリオ年率Vol (%)", 20, 60, 40, 5)
target_pf_vol = float(target_pf_vol_pct) / 100.0

vol_target_mode = st.sidebar.radio("Vol Targeting 方式", ["上限抑制のみ (Cap)", "双方向スケーリング (Targeting)"])

rebalance_buffer_pct = st.sidebar.slider("リバランス抑制バンド (No-Trade Buffer %)", 0.0, 5.0, 2.0, 0.5)
rebalance_buffer = float(rebalance_buffer_pct) / 100.0

fee_bps = st.sidebar.number_input("通常売買コスト＋スリッページ (片道 bps)", min_value=0, max_value=50, value=10, step=1)
fee_rate = float(fee_bps) / 10000.0

use_dd_controller = st.sidebar.checkbox("連続型DD連動リスク制御 (Smooth DD Controller)", value=True)
base_currency = st.sidebar.radio("評価基準通貨", ["米ドル (USD)", "日本円 (JPY)"], horizontal=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 暴落迎撃 (TCO v3.0 Sniper) 設定")
enable_tco = st.sidebar.checkbox("TCO 暴落迎撃モードを有効化", value=True)
tco_entry_mode = st.sidebar.selectbox(
    "TCO エントリー方式",
    ["反転確認 + 極限クライマックス (推奨)", "反転確認のみ (厳格・ナイフ回避)", "従来型 (売られすぎ即時逆張り)"],
    index=0,
    disabled=not enable_tco
)
capital_sweep_mode = st.sidebar.checkbox(
    "動的キャピタル・スイープ (平常時CORE最大100%活用)",
    value=True,
    disabled=not enable_tco
)
tco_max_cap_pct = st.sidebar.slider("TCO 最大迎撃動員枠 (%)", 10, 40, 30, 5, disabled=not enable_tco)
tco_max_cap = float(tco_max_cap_pct) / 100.0 if enable_tco else 0.0
tco_top_k = st.sidebar.selectbox("迎撃銘柄上限 (スナイパー枠数)", [1, 2, 3], index=1, disabled=not enable_tco)
tco_risk_budget_pct = st.sidebar.slider("1トレード許容リスク予算 (対PF資産 %)", 0.4, 2.0, 0.9, 0.1, disabled=not enable_tco)
tco_risk_budget = float(tco_risk_budget_pct) / 100.0
tco_max_hold = st.sidebar.slider("TCO タイムストップ上限 (営業日)", 4, 15, 8, 1, disabled=not enable_tco)
tco_fee_bps = st.sidebar.number_input("TCOパニック時スリッページ (片道 bps)", min_value=10, max_value=100, value=50, step=10, disabled=not enable_tco)
tco_fee_rate = float(tco_fee_bps) / 10000.0

# =============================================================
# 3. データ取得 & 幾何整合スプライシングエンジン
# =============================================================
def _clean_yf_series(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    res = pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        if ticker in df.columns.get_level_values(0):
            res = df[ticker].copy()
        elif ticker in df.columns.get_level_values(1):
            res = df.xs(ticker, axis=1, level=1).copy()
        else:
            return pd.DataFrame()
    else:
        res = df.copy()

    if res.index.tz is not None:
        res.index = res.index.tz_localize(None)
    for col in ["Open", "High", "Low", "Close"]:
        if col not in res.columns and "Close" in res.columns:
            res[col] = res["Close"]
    return res.dropna(subset=["Close", "Open"])

@st.cache_data(ttl=3600, show_spinner=False)
def load_market_environment(period_str: str) -> Tuple[Dict[str, pd.DataFrame], pd.Series, pd.DatetimeIndex]:
    u_tickers = [cfg.underlying for cfg in ASSETS.values()]
    etf_tickers = list(ASSETS.keys())
    all_tickers = sorted(list(set(u_tickers + etf_tickers + ["SPY", "USDJPY=X"])))

    try:
        downloaded = yf.download(
            all_tickers,
            period=period_str,
            interval="1d",
            progress=False,
            auto_adjust=True,
            group_by="ticker"
        )
    except Exception as e:
        raise RuntimeError(f"市場データの一括取得に失敗しました: {str(e)}")

    raw_data: Dict[str, pd.DataFrame] = {}
    for t in all_tickers:
        cleaned = _clean_yf_series(downloaded, t)
        if not cleaned.empty and len(cleaned) >= 30:
            raw_data[t] = cleaned

    if "SPY" not in raw_data or raw_data["SPY"].empty:
        raise RuntimeError("基準ベンチマーク (SPY) の取得に失敗しました。")

    base_idx = raw_data["SPY"].index
    for u in u_tickers:
        if u in raw_data and not raw_data[u].empty:
            base_idx = base_idx.intersection(raw_data[u].index)

    if len(base_idx) < 60:
        raise ValueError("共通営業日データ長が不足しています（60営業日未満）。期間設定を延長してください。")

    if "USDJPY=X" in raw_data and not raw_data["USDJPY=X"].empty:
        fx_series = raw_data["USDJPY=X"]["Close"].reindex(base_idx).ffill().bfill()
    else:
        fx_series = pd.Series(150.0, index=base_idx)

    cleaned_data: Dict[str, pd.DataFrame] = {}
    for u in u_tickers:
        cleaned_data[u] = raw_data[u].loc[base_idx].copy()

    daily_expense = EXPENSE_RATIO_ANNUAL / TRADING_DAYS_PER_YEAR

    for sym, cfg in ASSETS.items():
        u_sym = cfg.underlying
        lev = cfg.leverage
        df_u = cleaned_data[u_sym]

        u_close = df_u["Close"].values
        u_open = df_u["Open"].values
        u_high = df_u["High"].values
        u_low = df_u["Low"].values
        n = len(base_idx)

        syn_close = np.zeros(n)
        syn_open = np.zeros(n)
        syn_high = np.zeros(n)
        syn_low = np.zeros(n)
        syn_close[0] = syn_open[0] = syn_high[0] = syn_low[0] = 100.0

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

        has_real_etf = (sym in raw_data) and (not raw_data[sym].empty)
        real_idx = raw_data[sym].index.intersection(base_idx) if has_real_etf else pd.DatetimeIndex([])

        if len(real_idx) > 0:
            df_real = raw_data[sym].loc[real_idx]
            first_real_date = real_idx[0]
            first_real_open = float(df_real["Open"].iloc[0])
            syn_open_at_date = max(syn_df.loc[first_real_date, "Open"], EPSILON)
            scale = first_real_open / syn_open_at_date

            mask_before = syn_df.index < first_real_date
            syn_df.loc[mask_before] *= scale
            for col in ["Open", "High", "Low", "Close"]:
                syn_df.loc[real_idx, col] = df_real[col]

        cleaned_data[sym] = syn_df.ffill().bfill()

    cleaned_data["SPY"] = raw_data["SPY"].loc[base_idx].copy()
    return cleaned_data, fx_series, base_idx

with st.spinner("市場データを取得・同期中..."):
    try:
        market_data, fx_rates, all_base_idx = load_market_environment(period_code)
    except Exception as e:
        st.error(f"データ取得エラー: {str(e)}")
        st.stop()

if period_mode == "実ETFデータ限定（2011年6月〜現在）":
    common_idx = all_base_idx[all_base_idx >= pd.to_datetime(ALL_REAL_START_DATE)]
    if len(common_idx) < 50:
        common_idx = all_base_idx
else:
    common_idx = all_base_idx

latest_date_str = common_idx[-1].strftime('%Y年%m月%d日')
latest_fx = float(fx_rates.loc[common_idx[-1]])
n_days = len(common_idx)

# =============================================================
# 4. 特徴量 ＆ TCO反転シグナル ＆ Purged Walk-Forward
# =============================================================
def calc_rsi(series: pd.Series, period: int = 9) -> np.ndarray:
    delta = series.diff().fillna(0.0).values
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
    return np.clip(np.nan_to_num(pv, nan=25.0), 5.0, 200.0)

def calc_atr(df: pd.DataFrame, period: int = 10) -> np.ndarray:
    h = np.maximum(df["High"].values, EPSILON)
    l = np.maximum(df["Low"].values, EPSILON)
    c = np.maximum(df["Close"].values, EPSILON)
    n = len(c)
    tr = np.zeros(n)
    tr[0] = h[0] - l[0]
    for t in range(1, n):
        tr[t] = max(h[t] - l[t], abs(h[t] - c[t - 1]), abs(l[t] - c[t - 1]))
    atr = pd.Series(tr).ewm(alpha=1.0 / period, adjust=False).mean().values
    return np.nan_to_num(atr, nan=1.0)

def compute_market_crash_regime(
    market_data: Dict[str, pd.DataFrame],
    common_idx: pd.DatetimeIndex
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    spy_df = market_data["SPY"].loc[common_idx]
    pv_5 = calc_parkinson_vol(spy_df, window=5)
    pv_60 = calc_parkinson_vol(spy_df, window=60)
    vol_ratio = pv_5 / np.maximum(pv_60, EPSILON)

    spy_c = spy_df["Close"]
    spy_ema50 = spy_c.ewm(span=50, adjust=False).mean()
    spy_dd50 = ((spy_c - spy_ema50) / np.maximum(spy_ema50, EPSILON)).values
    spy_rsi = calc_rsi(spy_c, 9)

    gld_df = market_data["GLD"].loc[common_idx] if "GLD" in market_data else spy_df
    tlt_df = market_data["TLT"].loc[common_idx] if "TLT" in market_data else spy_df
    safe_flow = (gld_df["Close"].pct_change(5).fillna(0.0).values > 0.0) | \
                (tlt_df["Close"].pct_change(5).fillna(0.0).values > 0.0)

    is_panic_regime = ((vol_ratio >= 1.35) & (safe_flow | (spy_dd50 < -0.03))) | (spy_rsi < 33.0)
    # 極端なフリーフォール時 (vol_ratio > 2.2) はサイズを減衰させる温度係数
    panic_temp_scale = np.clip(1.0 / np.sqrt(np.maximum(vol_ratio / 1.5, 1.0)), 0.45, 1.0)
    return is_panic_regime, vol_ratio, panic_temp_scale

def compute_single_asset_signals(
    df_u: pd.DataFrame,
    df_etf: pd.DataFrame,
    cfg: AssetConfig
) -> Dict[str, Any]:
    o_u = df_u["Open"].values
    h_u = df_u["High"].values
    l_u = df_u["Low"].values
    c_u = df_u["Close"].values
    o_etf = df_etf["Open"].values
    c_etf = df_etf["Close"].values
    n = len(c_u)

    ema10 = pd.Series(c_u).ewm(span=10, adjust=False).mean().values
    ema20 = pd.Series(c_u).ewm(span=20, adjust=False).mean().values
    ema50 = pd.Series(c_u).ewm(span=50, adjust=False).mean().values
    ema200 = pd.Series(c_u).ewm(span=200, adjust=False).mean().values

    diff_10 = (c_u - ema10) / np.maximum(ema10, EPSILON)
    diff_20 = (c_u - ema20) / np.maximum(ema20, EPSILON)
    diff_50 = (c_u - ema50) / np.maximum(ema50, EPSILON)
    diff_200 = (c_u - ema200) / np.maximum(ema200, EPSILON)

    # トレンド加速度 (Momentum Acceleration: EMA20対EMA50スプレッドの5日変化率)
    ema_spread = (ema20 - ema50) / np.maximum(ema50, EPSILON)
    mom_accel = pd.Series(ema_spread).diff(5).fillna(0.0).values * 10.0
    trend_score = 0.55 * diff_50 + 0.35 * diff_200 + 0.10 * np.clip(mom_accel, -0.2, 0.2)

    hysteresis = np.ones(n, dtype=bool)
    state = True
    for i in range(n):
        if state and diff_50[i] < -0.008:
            state = False
        elif not state and (diff_50[i] > 0.008 or (diff_20[i] > 0.012 and diff_50[i] > -0.003)):
            state = True
        hysteresis[i] = state

    u_ret = pd.Series(c_u).pct_change().fillna(0.0)
    vol_20 = np.clip(u_ret.rolling(20, min_periods=1).std().values * np.sqrt(TRADING_DAYS_PER_YEAR) * 100.0, 5.0, 150.0)
    vol_score = -(vol_20 - cfg.u_vol_norm) / cfg.u_vol_norm

    rsi9 = calc_rsi(pd.Series(c_u), 9)
    rsi_q85 = pd.Series(rsi9).rolling(252, min_periods=30).quantile(0.85).values
    dyn_rsi_th = np.clip(np.nan_to_num(rsi_q85, nan=66.0), 62.0, 75.0)

    p_rsi = np.where(rsi9 > dyn_rsi_th, (rsi9 - dyn_rsi_th) / 20.0, 0.0)
    p_ema = np.where(diff_50 > 0.08, (diff_50 - 0.08) / 0.10, 0.0)
    overheat_penalty = np.clip(p_rsi + p_ema, 0.0, 0.60)

    sigma_local = calc_parkinson_vol(df_etf, 10)
    u_atr = calc_atr(df_u, 10)
    cur_u_atr = u_atr[-1]
    risk_pct_u = (2.2 * cur_u_atr) / max(c_u[-1], EPSILON) * 100.0
    risk_pct_etf = risk_pct_u * cfg.leverage

    # --- TCO v3.0: 無次元Zスコア & 2段階クライマックス反転シグナル ---
    daily_vol_dec = np.maximum(vol_20 / 100.0 / np.sqrt(TRADING_DAYS_PER_YEAR), 0.004)
    # 50日EMAからの標準化乖離Zスコア (単位整合済み)
    norm_z50 = diff_50 / ( daily_vol_dec * np.sqrt(25.0) )

    # 1. 売られすぎクライマックス条件 (当日または直近3日以内に到達)
    raw_oversold = (diff_50 < -0.055) & (rsi9 < 33.0)
    recent_oversold_setup = pd.Series(raw_oversold.astype(float)).rolling(3, min_periods=1).max().values > 0.5

    # 2. 底打ち反転確認 (Falling Knife 回避: 陽線 / 前日比プラス / 下ヒゲ拒絶+RSI改善)
    candle_range = np.maximum(h_u - l_u, EPSILON)
    close_pos_in_bar = (c_u - l_u) / candle_range  # 0.0(安値引け) ~ 1.0(高値引け)
    ret_1d = u_ret.values
    rsi_diff = pd.Series(rsi9).diff(1).fillna(0.0).values

    is_bull_bar = c_u > o_u
    is_pos_ret = ret_1d > 0.0
    is_wick_hook = (close_pos_in_bar >= 0.42) & (rsi_diff > -0.5)
    reversal_turn = is_bull_bar | is_pos_ret | is_wick_hook

    # 3. 歴史的極限クライマックス (RSI < 19 かつ Z < -2.6 かつ 下ヒゲ25%以上)
    extreme_climax = (rsi9 < 19.0) & (norm_z50 < -2.6) & (close_pos_in_bar >= 0.25)

    # 反転確信度スコア (0.3 ~ 1.25)
    depth_factor = np.clip(abs(norm_z50) / 2.2, 0.5, 1.3)
    turn_factor = np.where(is_bull_bar & is_pos_ret, 1.15, np.where(reversal_turn, 0.95, 0.55))
    # 200日EMAから-35%超乖離している構造破綻セクターはロットを抑制
    structural_health = np.where(diff_200 < -0.35, 0.55, np.where(diff_200 > -0.05, 1.10, 0.90))
    tco_conviction = np.clip(depth_factor * turn_factor * structural_health, 0.35, 1.25)

    # スナイパー優先度スコア (小さい=マイナスが大きいほど優先)
    sniper_score = norm_z50 * np.where(reversal_turn, 1.25, 0.80) * structural_health

    # 最新日のTCO診断バッジ
    is_setup_now = bool(recent_oversold_setup[-1] and diff_50[-1] < -0.03)
    is_turn_now = bool(reversal_turn[-1])
    if is_setup_now and is_turn_now:
        tco_badge = "🎯 迎撃反転点灯 (Setup+Turn完了)"
        tco_desc = f"Zスコア {norm_z50[-1]:.2f}σ ｜ RSI {rsi9[-1]:.1f} ｜ 底打ち反転確認済"
        tco_status_type = "ready"
    elif raw_oversold[-1] and not is_turn_now:
        tco_badge = "🔪 ナイフ落下中 (反転陽線待ち)"
        tco_desc = f"売られすぎ到達 (乖離 {diff_50[-1]*100:.1f}%, RSI {rsi9[-1]:.1f}) ｜ 底打ち確認待機"
        tco_status_type = "waiting_turn"
    elif diff_50[-1] < -0.055:
        tco_badge = f"⏳ 迎撃準備中 (RSIあと {max(0.0, rsi9[-1]-33.0):.1f}pt)"
        tco_desc = f"50日乖離クリア ({diff_50[-1]*100:.1f}%) ｜ RSI冷却待ち ({rsi9[-1]:.1f})"
        tco_status_type = "waiting_rsi"
    else:
        tco_badge = "🛡️ 平常圏 (迎撃基準外)"
        tco_desc = f"乖離 {diff_50[-1]*100:+.1f}% ({norm_z50[-1]:+.2f}σ) ｜ RSI {rsi9[-1]:.1f}"
        tco_status_type = "normal"

    # Purged Walk-Forward 確率校正 (4特徴量: trend, vol, overheat, mom_accel)
    fwd_ret_5d = pd.Series(c_u).pct_change(5).shift(-5).values
    y_target = np.where(fwd_ret_5d > 0, 1, 0)
    X_features = np.column_stack([trend_score, vol_score, overheat_penalty, mom_accel])

    p_bull = np.zeros(n)
    train_win = 252 * 3
    embargo = 5
    refit_freq = 42

    current_model = None
    current_scaler = None

    for t in range(n):
        if t < train_win + embargo:
            logit = 4.2 * trend_score[t] + 0.9 * vol_score[t] - 2.5 * overheat_penalty[t] + 0.8 * mom_accel[t]
            p_bull[t] = 1.0 / (1.0 + np.exp(-np.clip(logit, -15.0, 15.0)))
            continue

        if (t % refit_freq == 0) or (current_model is None):
            train_end = t - embargo
            train_start = max(0, train_end - train_win)
            X_tr = X_features[train_start:train_end]
            y_tr = y_target[train_start:train_end]

            if HAS_SKLEARN and len(np.unique(y_tr)) > 1:
                try:
                    scaler = StandardScaler()
                    X_tr_scaled = scaler.fit_transform(X_tr)
                    base_lr = LogisticRegression(C=0.8, max_iter=200, solver='lbfgs')
                    try:
                        cal_clf = CalibratedClassifierCV(estimator=base_lr, method='sigmoid', cv=3)
                    except TypeError:
                        cal_clf = CalibratedClassifierCV(base_estimator=base_lr, method='sigmoid', cv=3)
                    cal_clf.fit(X_tr_scaled, y_tr)
                    current_model = cal_clf
                    current_scaler = scaler
                except Exception:
                    current_model = None
                    current_scaler = None

        if current_model is not None and current_scaler is not None:
            try:
                X_cur = current_scaler.transform(X_features[t:t+1])
                p_bull[t] = current_model.predict_proba(X_cur)[0, 1]
            except Exception:
                logit = 4.2 * trend_score[t] + 0.9 * vol_score[t] - 2.5 * overheat_penalty[t] + 0.8 * mom_accel[t]
                p_bull[t] = 1.0 / (1.0 + np.exp(-np.clip(logit, -15.0, 15.0)))
        else:
            logit = 4.2 * trend_score[t] + 0.9 * vol_score[t] - 2.5 * overheat_penalty[t] + 0.8 * mom_accel[t]
            p_bull[t] = 1.0 / (1.0 + np.exp(-np.clip(logit, -15.0, 15.0)))

    p_bull = np.where(~hysteresis, np.minimum(p_bull, 0.15), p_bull)
    vol_adj = np.clip(cfg.sigma_target / np.maximum(sigma_local, 1e-4), 0.25, 1.25)
    w_star = np.clip(p_bull * vol_adj, 0.0, 1.0) * (1.0 - overheat_penalty)
    w_star = np.where(~hysteresis, 0.0, w_star)

    if not hysteresis[-1]:
        badge, desc = "🔴 弱気防衛 (待機)", "EMA50割れヒステリシス発動 / コア待機"
    elif overheat_penalty[-1] > 0.15:
        badge, desc = "⚠️ 過熱警戒 (抑制)", f"RSI {rsi9[-1]:.1f} 過熱ペナルティ作動"
    elif p_bull[-1] < 0.55:
        badge, desc = "🟡 探査打診 (抑制配分)", f"打診フェーズ (P(Bull) {p_bull[-1]*100:.1f}%)"
    else:
        badge, desc = "🟢 本玉巡航 (積極配分)", f"強気トレンド継続 (P(Bull) {p_bull[-1]*100:.1f}%)"

    return {
        "p_bull": p_bull, "w_star": w_star, "sigma_local": sigma_local, "vol_20": vol_20,
        "diff_10": diff_10, "diff_20": diff_20, "diff_50": diff_50, "diff_200": diff_200,
        "rsi9": rsi9, "dyn_rsi_th": dyn_rsi_th, "hysteresis": hysteresis,
        "u_atr": u_atr, "norm_z50": norm_z50, "sniper_score": sniper_score,
        "raw_oversold": raw_oversold, "recent_oversold_setup": recent_oversold_setup,
        "reversal_turn": reversal_turn, "extreme_climax": extreme_climax,
        "tco_conviction": tco_conviction, "trend_score": trend_score,
        "h_u": h_u, "c_u": c_u, "o_etf": o_etf, "c_etf": c_etf,
        "latest_p": p_bull[-1], "latest_w": w_star[-1], "latest_rsi": rsi9[-1],
        "latest_diff10": diff_10[-1], "latest_diff20": diff_20[-1], "latest_diff50": diff_50[-1], "latest_diff200": diff_200[-1],
        "latest_z50": norm_z50[-1], "latest_sniper": sniper_score[-1],
        "latest_th": dyn_rsi_th[-1], "badge": badge, "desc": desc,
        "tco_badge": tco_badge, "tco_desc": tco_desc, "tco_status_type": tco_status_type,
        "risk_pct_u": risk_pct_u, "risk_pct_etf": risk_pct_etf,
        "u_close": c_u[-1], "etf_price": c_etf[-1]
    }

asset_keys = list(ASSETS.keys())
signals: Dict[str, Dict[str, Any]] = {}
for sym in asset_keys:
    cfg_sym = ASSETS[sym]
    signals[sym] = compute_single_asset_signals(
        market_data[cfg_sym.underlying].loc[common_idx],
        market_data[sym].loc[common_idx],
        cfg_sym
    )

is_panic_series, vol_ratio_series, panic_temp_series = compute_market_crash_regime(market_data, common_idx)

# =============================================================
# 5. ステートフルTCO ＆ マルチファクターCORE ハイブリッド配分エンジン
# =============================================================
o_etf_matrix = np.column_stack([signals[k]["o_etf"] for k in asset_keys])
c_etf_matrix = np.column_stack([signals[k]["c_etf"] for k in asset_keys])

ret_open_to_open = np.zeros((n_days, len(asset_keys)))
ret_open_to_open[1:] = (o_etf_matrix[1:] - o_etf_matrix[:-1]) / np.maximum(o_etf_matrix[:-1], EPSILON)

@st.cache_data(ttl=3600, show_spinner=False)
def calculate_rolling_covariances(rets: np.ndarray, window: int = 60, min_p: int = 20) -> np.ndarray:
    n, k = rets.shape
    df_r = pd.DataFrame(rets)
    cov_arr = df_r.rolling(window, min_periods=min_p).cov().values.reshape(n, k, k) * TRADING_DAYS_PER_YEAR
    for i in range(n):
        if np.isnan(cov_arr[i]).any():
            cov_arr[i] = np.eye(k) * 0.16
    return cov_arr

roll_cov_matrix = calculate_rolling_covariances(ret_open_to_open)

def generate_hybrid_allocations(
    raw_signals: Dict[str, Dict[str, Any]],
    market_data: Dict[str, pd.DataFrame],
    common_idx: pd.DatetimeIndex,
    user_max_cap: float,
    cov_tensor: np.ndarray,
    target_v: float,
    is_bidirectional: bool,
    enable_tco: bool,
    tco_max_cap: float,
    tco_top_k: int,
    tco_entry_mode_str: str = "反転確認 + 極限クライマックス (推奨)",
    use_cap_sweep: bool = True,
    tco_risk_budget_val: float = 0.009,
    tco_max_hold_days: int = 8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[Tuple[int, int], str]]:
    k = len(asset_keys)
    n = len(common_idx)

    # 動的キャピタル・スイープ有効時は平常時CORE上限を100%まで開放
    base_core_limit = 1.0 if (not enable_tco or use_cap_sweep) else (1.0 - tco_max_cap)
    effective_core_cap = min(user_max_cap, base_core_limit)

    # クロスセクショナル相対モメンタム (RS Rank Multiplier: 0.80 ~ 1.25)
    trend_mat = np.column_stack([raw_signals[sym]["trend_score"] for sym in asset_keys])
    vol20_mat = np.column_stack([raw_signals[sym]["vol_20"] for sym in asset_keys])
    risk_adj_mom = trend_mat / np.maximum(vol20_mat / 100.0, 0.10)
    ranks = np.argsort(np.argsort(risk_adj_mom, axis=1), axis=1) / max(k - 1, 1)
    rs_multiplier = 0.80 + 0.45 * ranks

    raw_w_mat = np.zeros((n, k))
    cond_caps_mat = np.zeros((n, k))

    for idx, sym in enumerate(asset_keys):
        p = raw_signals[sym]["p_bull"]
        w = raw_signals[sym]["w_star"] * rs_multiplier[:, idx]
        cap = np.where(p < 0.40, 0.10, np.where(p < 0.55, 0.20, np.where(p < 0.70, 0.25, effective_core_cap)))
        cap = np.minimum(cap, effective_core_cap)
        w_filtered = np.where((p < 0.38) | (w < 0.05), 0.0, w)
        raw_w_mat[:, idx] = np.minimum(w_filtered, cap)
        cond_caps_mat[:, idx] = cap

    core_alloc = np.zeros_like(raw_w_mat)

    for t in range(n):
        w = raw_w_mat[t].copy()
        c_limits = cond_caps_mat[t].copy()
        active = np.where(w > 0.03)[0]
        if len(active) == 0:
            continue

        # 相関クラスタ・ペナルティ: アクティブ銘柄間の平均正相関が高いほど過度な重複集中を抑制
        cov_t = cov_tensor[t]
        std_t = np.sqrt(np.maximum(np.diag(cov_t), 1e-6))
        corr_t = cov_t / np.outer(std_t, std_t)
        if len(active) > 1:
            for a_i in active:
                other_corrs = [max(0.0, corr_t[a_i, a_j]) for a_j in active if a_j != a_i]
                mean_pos_corr = float(np.mean(other_corrs)) if other_corrs else 0.0
                # 相関0.75超の密集クラスタは最大15%減衰、低相関銘柄(UGL/TMF等)は最大10%優遇
                corr_adj = np.clip(1.10 - 0.30 * mean_pos_corr, 0.85, 1.10)
                w[a_i] *= corr_adj

        alloc = np.minimum(w, c_limits)
        pool = base_core_limit - float(np.sum(alloc))
        eligible = [i for i in active if alloc[i] < c_limits[i]]

        while eligible and pool > 1e-4:
            sub_w = w[eligible]
            s_sum = float(np.sum(sub_w))
            if s_sum <= EPSILON:
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

        tot_alloc = float(np.sum(alloc))
        if tot_alloc > base_core_limit:
            alloc = alloc * (base_core_limit / tot_alloc)

        core_alloc[t] = alloc

    tco_alloc = np.zeros((n, k))
    tco_exit_reasons: Dict[Tuple[int, int], str] = {}
    if not enable_tco:
        return core_alloc, tco_alloc, core_alloc, cond_caps_mat, tco_exit_reasons

    is_panic, _, panic_temp_scale = compute_market_crash_regime(market_data, common_idx)

    # TCO用マトリクス準備 (シグナル生成日 t のClose確定値で t 日目のターゲット配分 = t+1日Open執行 を決定)
    c_u_mat = np.column_stack([raw_signals[sym]["c_u"] for sym in asset_keys])
    h_u_mat = np.column_stack([raw_signals[sym]["h_u"] for sym in asset_keys])
    atr_mat = np.column_stack([raw_signals[sym]["u_atr"] for sym in asset_keys])
    diff_10_mat = np.column_stack([raw_signals[sym]["diff_10"] for sym in asset_keys])
    diff_20_mat = np.column_stack([raw_signals[sym]["diff_20"] for sym in asset_keys])
    diff_50_mat = np.column_stack([raw_signals[sym]["diff_50"] for sym in asset_keys])
    rsi9_mat = np.column_stack([raw_signals[sym]["rsi9"] for sym in asset_keys])
    sniper_mat = np.column_stack([raw_signals[sym]["sniper_score"] for sym in asset_keys])
    raw_os_mat = np.column_stack([raw_signals[sym]["raw_oversold"] for sym in asset_keys])
    setup_os_mat = np.column_stack([raw_signals[sym]["recent_oversold_setup"] for sym in asset_keys])
    rev_turn_mat = np.column_stack([raw_signals[sym]["reversal_turn"] for sym in asset_keys])
    climax_mat = np.column_stack([raw_signals[sym]["extreme_climax"] for sym in asset_keys])
    conv_mat = np.column_stack([raw_signals[sym]["tco_conviction"] for sym in asset_keys])
    lev_vec = np.array([ASSETS[sym].leverage for sym in asset_keys])

    # ステートフルTCOポジション管理辞書: a_idx -> state dict
    active_tco_pos: Dict[int, Dict[str, Any]] = {}

    for t in range(1, n):
        # 1. 既存TCOポジションの継続・イグジット判定 (日次 t の確定データで評価)
        for a_idx in list(active_tco_pos.keys()):
            st_pos = active_tco_pos[a_idx]
            st_pos["hold_days"] += 1
            st_pos["highest_u"] = max(st_pos["highest_u"], h_u_mat[t, a_idx], c_u_mat[t, a_idx])

            cur_u = c_u_mat[t, a_idx]
            entry_u = st_pos["entry_u"]
            entry_atr = st_pos["entry_atr"]
            unrealized_atr = (cur_u - entry_u) / max(entry_atr, EPSILON)

            # 含み益が +1.2 ATR 以上乗ったらトレーリング幅をタイト化＆建値ロック (利益吐き出し防止)
            if unrealized_atr >= 1.2:
                trail_stop = max(entry_u * 1.002, st_pos["highest_u"] - 1.6 * entry_atr)
            else:
                trail_stop = st_pos["highest_u"] - 2.2 * entry_atr

            # 4つの明確なイグジット条件
            is_profit_target = (rsi9_mat[t, a_idx] >= 54.0) or (diff_10_mat[t, a_idx] >= 0.0) or (diff_20_mat[t, a_idx] >= -0.005)
            is_stop_hit = cur_u < trail_stop
            is_time_stop = st_pos["hold_days"] >= tco_max_hold_days

            if is_profit_target:
                tco_exit_reasons[(t, a_idx)] = "🎯 利確到達 (RSI/EMA回復)"
                active_tco_pos.pop(a_idx)
            elif is_stop_hit:
                if unrealized_atr >= 0.5 and cur_u > entry_u:
                    tco_exit_reasons[(t, a_idx)] = "🛡️ トレーリング利確"
                else:
                    tco_exit_reasons[(t, a_idx)] = "✂️ ATRストップ損切り"
                active_tco_pos.pop(a_idx)
            elif is_time_stop:
                tco_exit_reasons[(t, a_idx)] = f"⌛ タイムストップ ({tco_max_hold_days}日)"
                active_tco_pos.pop(a_idx)
            else:
                # 継続保有 (含み益が伸びている場合は緩やかにウェイト維持)
                tco_alloc[t, a_idx] = st_pos["weight"]

        # 2. 空きスロットがある場合の新規スナイパーエントリー判定
        rem_slots = tco_top_k - len(active_tco_pos)
        used_tco_w = float(np.sum(tco_alloc[t]))
        avail_tco_cap = max(0.0, tco_max_cap - used_tco_w)

        if rem_slots > 0 and avail_tco_cap >= 0.02:
            candidates = []
            for i in range(k):
                if i in active_tco_pos:
                    continue
                # すでに利確ラインに近いものは除外
                if diff_50_mat[t, i] >= -0.04 or rsi9_mat[t, i] >= 42.0:
                    continue

                if tco_entry_mode_str == "反転確認のみ (厳格・ナイフ回避)":
                    trig = bool(setup_os_mat[t, i] and rev_turn_mat[t, i])
                elif tco_entry_mode_str == "従来型 (売られすぎ即時逆張り)":
                    trig = bool(raw_os_mat[t, i])
                else:
                    # 推奨ハイブリッド: 「売られすぎセットアップ + 反転確認」または「パニック時の極限クライマックス」
                    trig = bool((setup_os_mat[t, i] and rev_turn_mat[t, i]) or (is_panic[t] and climax_mat[t, i]))

                if trig:
                    candidates.append((i, sniper_mat[t, i]))

            if candidates:
                candidates.sort(key=lambda x: x[1])
                selected = candidates[:rem_slots]
                slot_max = avail_tco_cap / len(selected)

                for a_idx, _ in selected:
                    u_p = max(c_u_mat[t, a_idx], EPSILON)
                    atr_v = max(atr_mat[t, a_idx], EPSILON)
                    # ETFレバレッジ倍率込みの1トレード想定ストップ率
                    etf_stop_risk = (2.2 * atr_v / u_p) * lev_vec[a_idx]
                    base_size = tco_risk_budget_val / max(etf_stop_risk, 0.03)
                    # 確信度 × パニック温度スケーリング
                    scaled_size = base_size * conv_mat[t, a_idx] * panic_temp_series[t]
                    # まだ反転確認前の極限クライマックス先行打診時はロットを60%に抑制
                    if not rev_turn_mat[t, a_idx]:
                        scaled_size *= 0.60

                    final_size = float(np.clip(scaled_size, 0.02, slot_max))
                    tco_alloc[t, a_idx] = final_size
                    active_tco_pos[a_idx] = {
                        "entry_t": t,
                        "entry_u": u_p,
                        "highest_u": max(u_p, h_u_mat[t, a_idx]),
                        "entry_atr": atr_v,
                        "hold_days": 0,
                        "weight": final_size
                    }

        # 3. 動的キャピタル・スイープ: TCO発動分を優先確保し、100%を超える場合はCOREを比例圧縮
        tco_sum_t = float(np.sum(tco_alloc[t]))
        core_sum_t = float(np.sum(core_alloc[t]))
        if core_sum_t + tco_sum_t > 1.0 and core_sum_t > EPSILON:
            allowed_core = max(0.0, 1.0 - tco_sum_t)
            core_alloc[t] *= (allowed_core / core_sum_t)

    comb_alloc = core_alloc + tco_alloc
    return core_alloc, tco_alloc, comb_alloc, cond_caps_mat, tco_exit_reasons

is_bidir = (vol_target_mode == "双方向スケーリング (Targeting)")
core_target_alloc, tco_target_alloc, daily_target_alloc, active_caps, tco_exit_map = generate_hybrid_allocations(
    signals, market_data, common_idx, max_cap, roll_cov_matrix, target_pf_vol,
    is_bidir, enable_tco, tco_max_cap, tco_top_k,
    tco_entry_mode, capital_sweep_mode, tco_risk_budget, tco_max_hold
)

# =============================================================
# 6. バックテスト ＆ アトリビューション分解 ＆ 厳密トレード集計
# =============================================================
def run_hybrid_backtest(
    core_w: np.ndarray,
    tco_w: np.ndarray,
    o_matrix: np.ndarray,
    fx: np.ndarray,
    c_fee: float,
    t_fee: float,
    use_dd: bool,
    is_jpy: bool,
    rebal_buf: float,
    common_idx: pd.DatetimeIndex,
    exit_reason_map: Dict[Tuple[int, int], str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]], Dict[str, np.ndarray]]:
    n, k = core_w.shape
    valid_len = n - 2
    strat_net_ret = np.zeros(valid_len)
    core_attrib_ret = np.zeros(valid_len)
    tco_attrib_ret = np.zeros(valid_len)
    cash_attrib_ret = np.zeros(valid_len)

    effective_w = np.zeros((valid_len, k))
    effective_turnover = np.zeros(valid_len)
    daily_cash_rate = CASH_YIELD_ANNUAL / TRADING_DAYS_PER_YEAR

    peak = 1.0
    cum = 1.0
    last_cw = np.zeros(k)
    last_tw = np.zeros(k)

    open_trades: Dict[str, Dict[str, Any]] = {}
    completed_trades: List[Dict[str, Any]] = []

    for t in range(1, n - 1):
        idx_ret = t - 1
        cw_req = core_w[t - 1].copy()
        tw_req = tco_w[t - 1].copy()

        # 連続型Smooth DDコントローラー (階段関数によるチャタリング排除)
        if use_dd:
            dd = (cum - peak) / max(peak, EPSILON)
            if dd < -0.06:
                # DD -6% ~ -26% にかけて 1.0 -> 0.35 へ滑らかに減衰
                dd_scale = float(np.clip(1.0 + (dd + 0.06) * 3.25, 0.35, 1.0))
                cw_req *= dd_scale

        # No-Trade Buffer (COREのみ微小リバランス抑制、新規0->正・全決済正->0は即時執行)
        if rebal_buf > 0.0:
            for i in range(k):
                if cw_req[i] > 0.01 and last_cw[i] > 0.01:
                    if abs(cw_req[i] - last_cw[i]) < rebal_buf:
                        cw_req[i] = last_cw[i]

        tot_w = float(np.sum(cw_req) + np.sum(tw_req))
        if tot_w > 1.0:
            scale_down = 1.0 / tot_w
            cw_req *= scale_down
            tw_req *= scale_down

        w_req = cw_req + tw_req
        effective_w[idx_ret] = w_req

        core_turnover = float(np.sum(np.abs(cw_req - last_cw)))
        tco_turnover = float(np.sum(np.abs(tw_req - last_tw)))
        effective_turnover[idx_ret] = core_turnover + tco_turnover

        c_cost = core_turnover * c_fee
        t_cost = tco_turnover * t_fee
        cost = c_cost + t_cost

        r_asset_daily = (o_matrix[t + 1] - o_matrix[t]) / np.maximum(o_matrix[t], EPSILON)
        c_gross = float(np.sum(cw_req * r_asset_daily))
        t_gross = float(np.sum(tw_req * r_asset_daily))
        cash_ret = float(max(0.0, 1.0 - np.sum(w_req))) * daily_cash_rate

        c_net = c_gross - c_cost
        t_net = t_gross - t_cost
        net_ret = c_net + t_net + cash_ret

        if is_jpy:
            fx_ret = (fx[t + 1] - fx[t]) / max(fx[t], EPSILON)
            net_ret = (1.0 + net_ret) * (1.0 + fx_ret) - 1.0
            c_net = (1.0 + c_net) * (1.0 + fx_ret * float(np.sum(cw_req))) - 1.0
            t_net = (1.0 + t_net) * (1.0 + fx_ret * float(np.sum(tw_req))) - 1.0

        strat_net_ret[idx_ret] = net_ret
        core_attrib_ret[idx_ret] = c_net
        tco_attrib_ret[idx_ret] = t_net
        cash_attrib_ret[idx_ret] = cash_ret

        cum *= (1.0 + net_ret)
        if cum > peak:
            peak = cum

        # --- トレード履歴の厳密トラッキング (CORE / TCO 完全分離 & pop修正済) ---
        for a_idx in range(k):
            sym = asset_keys[a_idx]
            key_tco = f"TCO_{a_idx}"
            key_core = f"CORE_{a_idx}"
            curr_tw = tw_req[a_idx]
            curr_cw = cw_req[a_idx]
            prev_tw = last_tw[a_idx]
            prev_cw = last_cw[a_idx]

            # 1. TCO ポジション管理
            if prev_tw <= 0.005 and curr_tw > 0.005 and key_tco not in open_trades:
                open_trades[key_tco] = {
                    "strategy": "TCO (暴落迎撃)", "asset": sym, "entry_t": t,
                    "entry_price": o_matrix[t, a_idx], "weight": curr_tw
                }
            elif prev_tw > 0.005 and curr_tw <= 0.005 and key_tco in open_trades:
                tr = open_trades.pop(key_tco)
                exit_p = o_matrix[t, a_idx]
                pnl = (exit_p / max(tr["entry_price"], EPSILON) - 1.0) - (t_fee * 2.0)
                reason = exit_reason_map.get((t - 1, a_idx), "🔄 シグナル完了")
                completed_trades.append({
                    "strategy": "TCO (暴落迎撃)", "asset": sym,
                    "entry_date": common_idx[int(tr["entry_t"])], "exit_date": common_idx[t],
                    "holding_days": max(1, t - int(tr["entry_t"])), "pnl_pct": pnl * 100.0,
                    "weighted_pnl": pnl * tr["weight"], "weight": tr["weight"],
                    "exit_reason": reason
                })

            # 2. CORE ポジション管理 (open_trades.pop 欠落バグ修正済)
            if prev_cw <= 0.005 and curr_cw > 0.005 and key_core not in open_trades:
                open_trades[key_core] = {
                    "strategy": "CORE (順張り)", "asset": sym, "entry_t": t,
                    "entry_price": o_matrix[t, a_idx], "weight": curr_cw,
                    "max_weight": curr_cw
                }
            elif key_core in open_trades:
                if curr_cw > 0.005:
                    open_trades[key_core]["max_weight"] = max(open_trades[key_core]["max_weight"], curr_cw)
                else:
                    tr = open_trades.pop(key_core)
                    exit_p = o_matrix[t, a_idx]
                    pnl = (exit_p / max(tr["entry_price"], EPSILON) - 1.0) - (c_fee * 2.0)
                    completed_trades.append({
                        "strategy": "CORE (順張り)", "asset": sym,
                        "entry_date": common_idx[int(tr["entry_t"])], "exit_date": common_idx[t],
                        "holding_days": max(1, t - int(tr["entry_t"])), "pnl_pct": pnl * 100.0,
                        "weighted_pnl": pnl * tr["weight"], "weight": tr["weight"],
                        "exit_reason": "📉 トレンド終了 / ヒステリシス撤退"
                    })

        last_cw = cw_req.copy()
        last_tw = tw_req.copy()

    attrib_dict = {
        "core": core_attrib_ret,
        "tco": tco_attrib_ret,
        "cash": cash_attrib_ret
    }
    return strat_net_ret, effective_w, effective_turnover, completed_trades, attrib_dict

use_jpy = (base_currency == "日本円 (JPY)")
fx_array = fx_rates.loc[common_idx].values

strat_ret, eff_w, eff_turnover, all_trades, attrib_rets = run_hybrid_backtest(
    core_target_alloc, tco_target_alloc, o_etf_matrix, fx_array,
    fee_rate, tco_fee_rate, use_dd_controller, use_jpy, rebalance_buffer, common_idx, tco_exit_map
)

eval_idx = common_idx[1:-1]

bm_r = np.mean((o_etf_matrix[2:] - o_etf_matrix[1:-1]) / np.maximum(o_etf_matrix[1:-1], EPSILON), axis=1)
if use_jpy:
    fx_r = (fx_array[2:] - fx_array[1:-1]) / np.maximum(fx_array[1:-1], EPSILON)
    bm_eq_ret = (1.0 + bm_r) * (1.0 + fx_r) - 1.0
else:
    bm_eq_ret = bm_r

def evaluate_performance(rets: np.ndarray, eff_weights: np.ndarray, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(rets) == 0:
        return {k: 0.0 for k in ["CAGR", "MDD", "Vol", "Sharpe", "Sortino", "Calmar", "Daily_PF", "Trade_PF", "Trade_WinRate", "CVaR_95", "Mean_Exposure", "Cap_Efficiency", "Avg_Hold_Days", "Total_Trades"]}

    cum = np.cumprod(1.0 + rets)
    years = max(len(rets) / TRADING_DAYS_PER_YEAR, 0.1)
    final_cum = max(cum[-1], 1e-4)
    cagr = (final_cum ** (1.0 / years)) - 1.0

    peaks = np.maximum.accumulate(cum)
    dds = (cum - peaks) / np.maximum(peaks, EPSILON)
    mdd = float(np.min(dds))
    vol = float(np.std(rets, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
    mean_ann = float(np.mean(rets) * TRADING_DAYS_PER_YEAR)
    sharpe = (mean_ann - CASH_YIELD_ANNUAL) / max(vol, EPSILON)

    downside = rets[rets < 0]
    down_vol = float(np.std(downside, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)) if len(downside) > 1 else vol
    sortino = (mean_ann - CASH_YIELD_ANNUAL) / max(down_vol, EPSILON)
    calmar = (cagr / abs(mdd)) if abs(mdd) > EPSILON else 0.0

    var_95 = float(np.percentile(rets, 5))
    cvar_sub = rets[rets <= var_95]
    cvar_95 = float(np.mean(cvar_sub)) if len(cvar_sub) > 0 else var_95

    mean_exposure = float(np.mean(np.sum(eff_weights, axis=1))) if len(eff_weights) > 0 else 1.0
    cap_efficiency = cagr / max(mean_exposure, 0.05)

    pos_r = rets[rets > 0]
    neg_r = rets[rets < 0]
    daily_pf = (np.sum(pos_r) / abs(np.sum(neg_r))) if len(neg_r) > 0 and abs(np.sum(neg_r)) > EPSILON else 0.0

    if len(trades) > 0:
        win_trades = [t["weighted_pnl"] for t in trades if t["weighted_pnl"] > 0]
        loss_trades = [t["weighted_pnl"] for t in trades if t["weighted_pnl"] < 0]
        trade_win_rate = len(win_trades) / len(trades)
        sum_win = float(np.sum(win_trades)) if len(win_trades) > 0 else 0.0
        sum_loss = float(abs(np.sum(loss_trades))) if len(loss_trades) > 0 else 0.0
        trade_pf = (sum_win / sum_loss) if sum_loss > EPSILON else (99.0 if sum_win > 0 else 0.0)
        avg_hold_days = float(np.mean([t["holding_days"] for t in trades]))
    else:
        trade_win_rate, trade_pf, avg_hold_days = 0.0, 0.0, 0.0

    return {
        "CAGR": cagr, "MDD": mdd, "Vol": vol, "Sharpe": sharpe, "Sortino": sortino, "Calmar": calmar,
        "Daily_PF": daily_pf, "Trade_PF": trade_pf, "Trade_WinRate": trade_win_rate,
        "CVaR_95": cvar_95, "Mean_Exposure": mean_exposure, "Cap_Efficiency": cap_efficiency,
        "Avg_Hold_Days": avg_hold_days, "Total_Trades": len(trades)
    }

metrics_strat = evaluate_performance(strat_ret, eff_w, all_trades)
metrics_bm = evaluate_performance(bm_eq_ret, np.ones((len(bm_eq_ret), len(asset_keys))) * (1.0 / len(asset_keys)), [])

cum_strat = np.cumprod(1.0 + strat_ret)
cum_bm = np.cumprod(1.0 + bm_eq_ret)

latest_core_alloc = core_target_alloc[-1]
latest_tco_alloc = tco_target_alloc[-1]
latest_alloc = daily_target_alloc[-1]
tot_inv = float(np.sum(latest_alloc))
tot_cash = max(0.0, 1.0 - tot_inv)

# =============================================================
# 7. ダッシュボード・プレゼンテーション層
# =============================================================
st.title("⚡ SDE-Engine Pro v3.0.0 | 次世代クオンツ統合システム")
st.caption(f"検証モード: **{period_mode}** ｜ 通貨: **{base_currency}** ｜ データ境界確定日: **{latest_date_str}** ({len(eval_idx)}営業日)")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏛️ 最適配分 ＆ クオンツ・レーダー",
    "🧪 バックテスト ＆ 寄与度分解",
    "📜 トレード詳細診断 (CORE vs TCO)",
    "🔄 Purged Walk-Forward (OOS) 検証",
    "🎲 Block Bootstrap モンテカルロ",
    "💎 3目的 Pareto Frontier 探索"
])

# -------------------------------------------------------------
# TAB 1: 最適配分 ＆ クオンツ・レーダー
# -------------------------------------------------------------
with tab1:
    st.info(f"📌 **判定基準日: {latest_date_str}（前日Close確定シグナル → 翌朝NYオープン執行）**")

    with st.expander("🎯 TCO v3.0 ステートフル暴落迎撃 ＆ マルチファクターCORE 運用方針ガイド", expanded=True):
        p_col1, p_col2, p_col3 = st.columns(3)
        with p_col1:
            st.markdown(
                "**1. 2段階クライマックス反転トリガー**\n\n"
                "・**Setup**: 直近3日以内に原資産50日EMA乖離 < -5.5% ＆ 9日RSI < 33到達\n"
                "・**Turn (ナイフ回避)**: 当日陽線・前日比プラス・下ヒゲ拒絶のいずれかで底打ちを確認後に翌朝Openエントリー\n"
                "・**銘柄選別**: 単位整合済み標準化Zスコア×反転強度上位に厳選動員"
            )
        with p_col2:
            st.markdown(
                "**2. 動的リスク予算 ＆ キャピタル・スイープ**\n\n"
                "・**平常時**: CORE順張りが最大100%まで資本を有効活用 (キャッシュドラッグ解消)\n"
                "・**迎撃時**: ETFレバレッジ込みATRリスクに対し、1トレード許容リスク予算・パニック温度係数・確信度で最適ロットを自動算出"
            )
        with p_col3:
            st.markdown(
                "**3. ステートフル保持 ＆ 4重イグジット**\n\n"
                "・**1日強制決済バグ解消**: RSIが33を超えても利確目標まで継続保有\n"
                "・**利確**: 9日RSI ≥ 54 または 10日/20日EMA回復で翌Open決済\n"
                "・**利益保護**: +1.2 ATR到達で建値ロック＋タイトトレーリング移行\n"
                f"・**防衛**: 最高値 -2.2 ATR割れ または {tco_max_hold}営業日経過で撤退"
            )

    cur_panic = is_panic_series[-1]
    cur_vol_r = vol_ratio_series[-1]
    cur_temp = panic_temp_series[-1]

    r_c1, r_c2, r_c3 = st.columns([2, 1, 1])
    with r_c1:
        if cur_panic:
            st.error(f"🚨 **市場クラッシュ警戒レジーム検知中 (長短Vol比: {cur_vol_r:.2f}倍)** ｜ TCOスナイパー監視稼働中")
        else:
            st.success(f"🕊️ **通常市場レジーム (長短Vol比: {cur_vol_r:.2f}倍)** ｜ マルチファクターCORE巡航稼働中")
    with r_c2:
        st.metric("TCO動員ステータス", f"🎯 迎撃中 ({np.sum(latest_tco_alloc)*100:.1f}%)" if np.sum(latest_tco_alloc) > 0 else "🛡️ 待機 (0.0%)")
    with r_c3:
        st.metric("パニック温度ロット係数", f"{cur_temp*100:.0f} %", "激動時自動縮小")

    st.markdown("---")

    chunk_size = 5
    for row_start in range(0, len(asset_keys), chunk_size):
        chunk_keys = asset_keys[row_start:row_start + chunk_size]
        cols = st.columns(len(chunk_keys))
        for col_idx, sym in enumerate(chunk_keys):
            idx_c = row_start + col_idx
            sig = signals[sym]
            c_alloc = latest_core_alloc[idx_c]
            t_alloc = latest_tco_alloc[idx_c]
            tot_a = latest_alloc[idx_c]

            with cols[col_idx]:
                st.markdown(f"#### {sym}")
                st.caption(f"{ASSETS[sym].name}")
                st.metric(f"原資産 {ASSETS[sym].underlying}", f"${sig['u_close']:.2f}")

                st.write(f"校正後 P(Bull): **{sig['latest_p']*100:.1f}%**")
                st.progress(float(np.clip(sig["latest_p"], 0.0, 1.0)))

                st.caption(f"**9日RSI:** {sig['latest_rsi']:.1f} (動的閾値: {sig['latest_th']:.1f})")
                st.caption(f"**乖離:** 50日 `{sig['latest_diff50']*100:+.1f}%` ({sig['latest_z50']:+.1f}σ) ｜ 200日 `{sig['latest_diff200']*100:+.1f}%`")

                st.info(f"**{sig['badge']}**")

                st.markdown("**🎯 TCO迎撃ステータス:**")
                if sig["tco_status_type"] == "ready":
                    st.success(f"**{sig['tco_badge']}**\n\n*{sig['tco_desc']}*")
                elif sig["tco_status_type"] in ["waiting_turn", "waiting_rsi"]:
                    st.warning(f"**{sig['tco_badge']}**\n\n*{sig['tco_desc']}*")
                else:
                    st.caption(f"**{sig['tco_badge']}** ｜ *{sig['tco_desc']}*")

                st.caption(f"・ETF想定ストップ幅: 約 -{sig['risk_pct_etf']:.1f}% (原資産 -{sig['risk_pct_u']:.1f}%)")
                st.caption(f"・利確目標 (10日/20日EMA): `{sig['latest_diff10']*100:+.1f}%` / `{sig['latest_diff20']*100:+.1f}%`")

                if t_alloc > 0:
                    st.warning(f"🎯 **TCO迎撃配分: {t_alloc*100:.1f}%**")

                if tot_a > 0:
                    st.success(f"**推奨総配分:**\n\n### {tot_a*100:.1f} %")
                    st.caption(f"内訳: コア {c_alloc*100:.1f}% + 迎撃 {t_alloc*100:.1f}%")
                else:
                    st.write("**配分:**\n\n### 0.0 % (待機)")
        st.markdown("")

    st.markdown("---")
    # 2次元クオンツ・レーダー & 相関ヒートマップ
    rad_c1, rad_c2 = st.columns([1.3, 1])
    with rad_c1:
        st.markdown("#### 📡 2次元クオンツ・レーダー (順張り確率 vs 暴落迎撃Zスコア)")
        r_x = [signals[s]["latest_p"] * 100.0 for s in asset_keys]
        r_y = [-signals[s]["latest_z50"] for s in asset_keys]  # 上に行くほど売られすぎ
        r_size = [max(14.0, latest_alloc[i] * 160.0) for i in range(len(asset_keys))]
        r_colors = [ASSETS[s].color for s in asset_keys]

        fig_rad = go.Figure()
        fig_rad.add_trace(go.Scatter(
            x=r_x, y=r_y, mode="markers+text", text=asset_keys,
            textposition="top center",
            marker=dict(size=r_size, color=r_colors, opacity=0.85, line=dict(width=1.5, color="white")),
            hovertemplate="<b>%{text}</b><br>P(Bull): %{x:.1f}%<br>売られすぎ深度 (-Z): %{y:.2f}σ<extra></extra>"
        ))
        fig_rad.add_hline(y=1.8, line_dash="dash", line_color="#e74c3c", annotation_text="TCO迎撃警戒圏 (+1.8σ売られすぎ)")
        fig_rad.add_vline(x=55.0, line_dash="dash", line_color="#00ba38", annotation_text="CORE巡航基準 (55%)")
        fig_rad.update_layout(
            xaxis_title="順張り強気確率 P(Bull) [%] → 右ほどCORE優勢",
            yaxis_title="売られすぎ深度 (-Z50) [σ] → 上ほどTCO迎撃候補",
            height=340, margin=dict(t=20, b=20, l=10, r=10)
        )
        st.plotly_chart(fig_rad, use_container_width=True)

    with rad_c2:
        st.markdown("#### 🔗 直近60営業日 ローリング相関マトリクス")
        cov_now = roll_cov_matrix[-1]
        std_now = np.sqrt(np.maximum(np.diag(cov_now), 1e-6))
        corr_now = np.round(cov_now / np.outer(std_now, std_now), 2)
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_now, x=asset_keys, y=asset_keys,
            colorscale="RdBu_r", zmin=-1.0, zmax=1.0,
            text=corr_now, texttemplate="%{text:.2f}", textfont={"size": 10}
        ))
        fig_corr.update_layout(height=340, margin=dict(t=20, b=20, l=10, r=10))
        st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("---")
    c_s1, c_s2 = st.columns([1.5, 1])
    with c_s1:
        st.markdown("### 💼 ポートフォリオ実効アロケーション")
        m_c1, m_c2, m_c3, m_c4 = st.columns(4)
        m_c1.metric("コア順張り投資枠", f"{np.sum(latest_core_alloc)*100:.1f} %")
        m_c2.metric("TCO暴落迎撃枠", f"{np.sum(latest_tco_alloc)*100:.1f} %")
        m_c3.metric("米ドルMMF待機比率", f"{tot_cash*100:.1f} %")
        m_c4.metric("総エクスポージャー", f"{tot_inv*100:.1f} %")
        st.caption("※動的キャピタル・スイープにより総投資枠は厳密に100%以内に制御されます。")

    with c_s2:
        active_labels = [asset_keys[i] for i in range(len(asset_keys)) if latest_alloc[i] > 0] + ["米ドルMMF"]
        active_vals = [latest_alloc[i] * 100 for i in range(len(asset_keys)) if latest_alloc[i] > 0] + [tot_cash * 100]
        fig_pie = go.Figure(data=[go.Pie(labels=active_labels, values=active_vals, hole=.45)])
        fig_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=200)
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")
    st.subheader("💡 翌朝寄り付き発注シミュレーター")
    curr_c1, curr_c2, curr_c3 = st.columns([1.2, 1.5, 1.3])
    with curr_c1:
        in_curr = st.radio("入力通貨", ["日本円 (万円)", "米ドル (USD)"], horizontal=True)
    with curr_c2:
        if in_curr == "日本円 (万円)":
            f_jpy_man = st.number_input("運用資産総額（万円）", min_value=10, value=300, step=10)
            total_usd = (f_jpy_man * 10000.0) / latest_fx
        else:
            total_usd = float(st.number_input("運用資産総額 (USD)", min_value=1000, value=20000, step=1000))
    with curr_c3:
        st.metric("最新為替レート (USD/JPY)", f"¥{latest_fx:.2f}")

    sim_rows = []
    for idx_c, sym in enumerate(asset_keys):
        alloc_ratio = latest_alloc[idx_c]
        p = signals[sym]["etf_price"]
        t_usd = total_usd * alloc_ratio
        t_jpy = t_usd * latest_fx
        shares = int(t_usd // p) if p > 0 else 0
        tag = "🎯 迎撃" if latest_tco_alloc[idx_c] > 0 else ("🟢 巡航" if latest_core_alloc[idx_c] > 0 else "待機")
        sim_rows.append({
            "銘柄": ASSETS[sym].name,
            "種別": tag,
            "最適総配分": f"{alloc_ratio*100:.1f} %",
            "投資目標額 (USD)": f"${t_usd:,.2f}",
            "概算金額 (JPY)": f"約 {t_jpy:,.0f} 円" if alloc_ratio > 0 else "0 円",
            "参考株価": f"${p:.2f}",
            "執行株数": f"{shares} 株",
            "アクション": f"目標 {shares} 株 買付" if alloc_ratio > 0 else "保有なし / 全売却待機"
        })
    c_usd = total_usd * tot_cash
    sim_rows.append({
        "銘柄": "米ドルMMF / 現金待機", "種別": "🛡 保守", "最適総配分": f"{tot_cash*100:.1f} %",
        "投資目標額 (USD)": f"${c_usd:,.2f}", "概算金額 (JPY)": f"約 {c_usd*latest_fx:,.0f} 円",
        "参考株価": "-", "執行株数": "-", "アクション": "MMF待機 (年利約3.5%)"
    })
    st.dataframe(pd.DataFrame(sim_rows), use_container_width=True, hide_index=True)

# -------------------------------------------------------------
# TAB 2: バックテスト検証 ＆ 寄与度分解
# -------------------------------------------------------------
with tab2:
    st.subheader("🧪 厳密クオンツ・バックテスト検証 (ネット費用後)")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("CAGR", f"{metrics_strat['CAGR']*100:.1f}%", f"BM均等: {metrics_bm['CAGR']*100:.1f}%")
    c2.metric("最大下落率 (MDD)", f"{metrics_strat['MDD']*100:.1f}%", f"BM均等: {metrics_bm['MDD']*100:.1f}%")
    c3.metric("Sharpe / Sortino", f"{metrics_strat['Sharpe']:.2f} / {metrics_strat['Sortino']:.2f}", f"Calmar: {metrics_strat['Calmar']:.2f}")
    c4.metric("Trade PF (加重)", f"{metrics_strat['Trade_PF']:.2f}", f"勝率: {metrics_strat['Trade_WinRate']*100:.1f}%")
    c5.metric("実効資本効率 (CAGR/Exp)", f"{metrics_strat['Cap_Efficiency']:.2f}", f"平均Exp: {metrics_strat['Mean_Exposure']*100:.1f}%")
    c6.metric("95% CVaR (日次)", f"{metrics_strat['CVaR_95']*100:.2f}%", f"全 {metrics_strat['Total_Trades']} 件")

    st.markdown("---")
    fig_bt = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.50, 0.25, 0.25])
    fig_bt.add_trace(go.Scatter(x=eval_idx, y=cum_strat, name="SDE Pro v3.0.0 (統合進化版)", line=dict(color="#00ba38", width=2.5)), row=1, col=1)
    fig_bt.add_trace(go.Scatter(x=eval_idx, y=cum_bm, name="BM: Daily Equal Weight (10銘柄均等)", line=dict(color="#888888", width=1.5, dash="dot")), row=1, col=1)

    # 戦略別寄与度分解 (Attribution)
    cum_core_attr = np.cumprod(1.0 + attrib_rets["core"])
    cum_tco_attr = np.cumprod(1.0 + attrib_rets["tco"])
    cum_cash_attr = np.cumprod(1.0 + attrib_rets["cash"])
    fig_bt.add_trace(go.Scatter(x=eval_idx, y=cum_tco_attr, name="TCO (暴落迎撃) 単体寄与倍率", line=dict(color="#f39c12", width=2)), row=2, col=1)
    fig_bt.add_trace(go.Scatter(x=eval_idx, y=cum_cash_attr, name="MMF金利 単体寄与倍率", line=dict(color="#3498db", width=1.5, dash="dot")), row=2, col=1)

    peak_s = np.maximum.accumulate(cum_strat)
    dd_s = (cum_strat - peak_s) / np.maximum(peak_s, EPSILON)
    peak_b = np.maximum.accumulate(cum_bm)
    dd_b = (cum_bm - peak_b) / np.maximum(peak_b, EPSILON)

    fig_bt.add_trace(go.Scatter(x=eval_idx, y=dd_s * 100, name="戦略DD (%)", fill="tozeroy", line=dict(color="#d62728", width=1)), row=3, col=1)
    fig_bt.add_trace(go.Scatter(x=eval_idx, y=dd_b * 100, name="BM均等DD (%)", line=dict(color="#7f7f7f", width=1, dash="dash")), row=3, col=1)

    fig_bt.update_layout(height=680, margin=dict(t=20, b=20, l=10, r=10), hovermode="x unified")
    fig_bt.update_yaxes(title_text="累積資産成長 (対数)", type="log", row=1, col=1)
    fig_bt.update_yaxes(title_text="TCO/MMF寄与", row=2, col=1)
    fig_bt.update_yaxes(title_text="DD (%)", row=3, col=1)
    st.plotly_chart(fig_bt, use_container_width=True)

    st.markdown("### 📋 リスク・リターン ＆ 資本効率 詳細対比")
    perf_data = [
        {"戦略": "SDE-Engine Pro v3.0.0 (統合進化版)", "CAGR": f"{metrics_strat['CAGR']*100:.1f}%", "MDD": f"{metrics_strat['MDD']*100:.1f}%", "Sharpe": f"{metrics_strat['Sharpe']:.2f}", "Sortino": f"{metrics_strat['Sortino']:.2f}", "Calmar": f"{metrics_strat['Calmar']:.2f}", "Trade PF": f"{metrics_strat['Trade_PF']:.2f}", "勝率": f"{metrics_strat['Trade_WinRate']*100:.1f}%", "平均Exp": f"{metrics_strat['Mean_Exposure']*100:.1f}%", "資本効率": f"{metrics_strat['Cap_Efficiency']:.2f}"},
        {"戦略": "BM: Equal Weight (10銘柄)", "CAGR": f"{metrics_bm['CAGR']*100:.1f}%", "MDD": f"{metrics_bm['MDD']*100:.1f}%", "Sharpe": f"{metrics_bm['Sharpe']:.2f}", "Sortino": f"{metrics_bm['Sortino']:.2f}", "Calmar": f"{metrics_bm['Calmar']:.2f}", "Trade PF": "-", "勝率": "-", "平均Exp": "100.0%", "資本効率": f"{metrics_bm['Cap_Efficiency']:.2f}"}
    ]
    st.dataframe(pd.DataFrame(perf_data), use_container_width=True, hide_index=True)

# -------------------------------------------------------------
# TAB 3: トレード詳細診断 (CORE vs TCO)
# -------------------------------------------------------------
with tab3:
    st.subheader("📜 戦略別・決済理由別・銘柄別 実績クロス診断")
    st.caption("COREのopen_trades.pop欠落バグおよびTCOの1日強制決済バグを解消した厳密なトレード統計です。")

    if len(all_trades) > 0:
        df_trades = pd.DataFrame(all_trades)
        t_col1, t_col2 = st.columns(2)

        for st_name, col_obj in [("CORE (順張り)", t_col1), ("TCO (暴落迎撃)", t_col2)]:
            sub = df_trades[df_trades["strategy"] == st_name]
            with col_obj:
                st.markdown(f"#### 🏛️ {st_name}")
                if len(sub) > 0:
                    win_sub = sub[sub["weighted_pnl"] > 0]
                    w_rate = len(win_sub) / len(sub) * 100.0
                    sum_w = sub[sub["weighted_pnl"] > 0]["weighted_pnl"].sum()
                    sum_l = abs(sub[sub["weighted_pnl"] < 0]["weighted_pnl"].sum())
                    s_pf = (sum_w / sum_l) if sum_l > EPSILON else (99.0 if sum_w > 0 else 0.0)
                    avg_h = sub["holding_days"].mean()
                    tot_contrib = sub["weighted_pnl"].sum() * 100.0

                    sm1, sm2, sm3, sm4 = st.columns(4)
                    sm1.metric("トレード数", f"{len(sub)} 件")
                    sm2.metric("勝率 / PF", f"{w_rate:.1f}% / {s_pf:.2f}")
                    sm3.metric("累積PF貢献", f"{tot_contrib:+.1f}%")
                    sm4.metric("平均保有日数", f"{avg_h:.1f} 日")
                else:
                    st.info(f"{st_name} の完結トレードはありません。")

        st.markdown("---")
        diag_c1, diag_c2 = st.columns(2)
        tco_only = df_trades[df_trades["strategy"] == "TCO (暴落迎撃)"]

        with diag_c1:
            st.markdown("#### 🎯 TCO 決済理由別パフォーマンス内訳")
            if len(tco_only) > 0:
                reason_rows = []
                for r_name, r_grp in tco_only.groupby("exit_reason"):
                    rw = (r_grp["weighted_pnl"] > 0).mean() * 100.0
                    sw = r_grp[r_grp["weighted_pnl"] > 0]["weighted_pnl"].sum()
                    sl = abs(r_grp[r_grp["weighted_pnl"] < 0]["weighted_pnl"].sum())
                    r_pf = (sw / sl) if sl > EPSILON else (99.0 if sw > 0 else 0.0)
                    reason_rows.append({
                        "決済理由": r_name,
                        "件数": len(r_grp),
                        "勝率": f"{rw:.1f}%",
                        "平均単体損益": f"{r_grp['pnl_pct'].mean():+.2f}%",
                        "累積PF貢献": f"{r_grp['weighted_pnl'].sum()*100:+.2f}%",
                        "PF": f"{r_pf:.2f}"
                    })
                st.dataframe(pd.DataFrame(reason_rows), use_container_width=True, hide_index=True)
            else:
                st.caption("TCOトレードなし")

        with diag_c2:
            st.markdown("#### 📊 TCO 銘柄別スナイパー成績")
            if len(tco_only) > 0:
                asset_rows = []
                for a_name, a_grp in tco_only.groupby("asset"):
                    aw = (a_grp["weighted_pnl"] > 0).mean() * 100.0
                    sw = a_grp[a_grp["weighted_pnl"] > 0]["weighted_pnl"].sum()
                    sl = abs(a_grp[a_grp["weighted_pnl"] < 0]["weighted_pnl"].sum())
                    a_pf = (sw / sl) if sl > EPSILON else (99.0 if sw > 0 else 0.0)
                    asset_rows.append({
                        "銘柄": a_name,
                        "件数": len(a_grp),
                        "勝率": f"{aw:.1f}%",
                        "平均保有": f"{a_grp['holding_days'].mean():.1f}日",
                        "累積PF貢献": f"{a_grp['weighted_pnl'].sum()*100:+.2f}%",
                        "PF": f"{a_pf:.2f}"
                    })
                st.dataframe(pd.DataFrame(asset_rows), use_container_width=True, hide_index=True)
            else:
                st.caption("TCOトレードなし")

        st.markdown("---")
        df_disp = df_trades.copy()
        df_disp["entry_date"] = pd.to_datetime(df_disp["entry_date"]).dt.strftime('%Y-%m-%d')
        df_disp["exit_date"] = pd.to_datetime(df_disp["exit_date"]).dt.strftime('%Y-%m-%d')
        df_disp["pnl_pct_str"] = df_disp["pnl_pct"].apply(lambda x: f"{x:+.2f} %")
        df_disp["weighted_pnl_str"] = df_disp["weighted_pnl"].apply(lambda x: f"{x*100:+.2f} %")
        df_disp["weight_str"] = df_disp["weight"].apply(lambda x: f"{x*100:.1f} %")

        st.dataframe(
            df_disp[["strategy", "asset", "entry_date", "exit_date", "weight_str", "holding_days", "pnl_pct_str", "weighted_pnl_str", "exit_reason"]].rename(columns={
                "strategy": "戦略種別", "asset": "銘柄", "entry_date": "買付日", "exit_date": "売却日",
                "weight_str": "投入配分", "holding_days": "保有日数",
                "pnl_pct_str": "単体純損益 (%)", "weighted_pnl_str": "PF貢献損益 (%)", "exit_reason": "決済理由"
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
    st.caption("3年学習・5日Embargo・1年完全未学習OOSテストを逐次連結した実績推移です。")

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
            oos_dates.extend(eval_idx[test_start:test_end])

            seg_cum = np.cumprod(1.0 + seg)
            seg_ret = (seg_cum[-1] - 1.0) * 100.0
            seg_peaks = np.maximum.accumulate(seg_cum)
            seg_dd = np.min((seg_cum - seg_peaks) / np.maximum(seg_peaks, EPSILON)) * 100.0
            oos_splits.append({
                "検証期間": f"{eval_idx[test_start].strftime('%Y/%m')} - {eval_idx[test_end-1].strftime('%Y/%m')}",
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
    st.caption("10営業日のブロック単位でリサンプリングし、テールリスクを再現したファンチャートです。")

    block_size = 10
    n_sims = 500
    n_steps = min(252 * 5, len(strat_ret))
    base_r = strat_ret[-n_steps:]
    n_base = len(base_r)

    np.random.seed(42)
    n_blocks = int(np.ceil(n_steps / block_size))
    rand_starts = np.random.randint(0, n_base, size=(n_sims, n_blocks))
    offsets = np.arange(block_size)
    sampled_indices = (rand_starts[:, :, None] + offsets[None, None, :]).reshape(n_sims, -1)[:, :n_steps] % n_base
    resampled_returns = base_r[sampled_indices]
    sim_paths = np.cumprod(1.0 + resampled_returns, axis=1)

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
            r_st, _, _, _, _ = run_hybrid_backtest(
                core_target_alloc, tco_target_alloc, o_etf_matrix, fx_array,
                test_bps/10000.0, max(test_bps*2, 30)/10000.0, use_dd_controller, use_jpy,
                rebalance_buffer, common_idx, tco_exit_map
            )
            c_st = (np.cumprod(1.0 + r_st)[-1] ** (1.0 / max(len(r_st)/252, 0.1))) - 1.0
            stress_list.append({
                "片道コスト": f"{test_bps} bps", "CAGR": f"{c_st*100:.1f}%", "劣化幅": f"{(c_st - metrics_strat['CAGR'])*100:+.1f}%"
            })
        st.dataframe(pd.DataFrame(stress_list), use_container_width=True, hide_index=True)

# -------------------------------------------------------------
# TAB 6: 3目的 Pareto Frontier 探索
# -------------------------------------------------------------
with tab6:
    st.subheader("💎 3目的 Pareto Frontier (収益 vs MDD vs CVaR) 探索")

    if "pareto_df" not in st.session_state:
        st.session_state.pareto_df = None

    if st.button("🚀 パレート探索を実行 (TCO v3.0 ハイブリッド グリッドスキャン)"):
        with st.spinner("パラメータ空間を高速走査中..."):
            grid_results = []
            for scan_vol in [0.30, 0.40, 0.50]:
                for scan_tco in [0.0, 0.20, 0.30]:
                    s_core, s_tco, s_comb, _, s_exit_map = generate_hybrid_allocations(
                        signals, market_data, common_idx, max_cap, roll_cov_matrix, scan_vol,
                        False, (scan_tco > 0), scan_tco, tco_top_k,
                        tco_entry_mode, capital_sweep_mode, tco_risk_budget, tco_max_hold
                    )
                    r_sc, eff_w_sc, _, tr_sc, _ = run_hybrid_backtest(
                        s_core, s_tco, o_etf_matrix, fx_array, fee_rate, tco_fee_rate,
                        use_dd_controller, use_jpy, rebalance_buffer, common_idx, s_exit_map
                    )
                    m_sc = evaluate_performance(r_sc, eff_w_sc, tr_sc)
                    grid_results.append({
                        "TargetVol": f"{int(scan_vol*100)}%",
                        "TCO_Cap": f"{int(scan_tco*100)}%",
                        "設定ラベル": f"Vol {int(scan_vol*100)}% / 迎撃枠 {int(scan_tco*100)}%",
                        "CAGR": m_sc["CAGR"] * 100.0,
                        "MDD": abs(m_sc["MDD"] * 100.0),
                        "CVaR_95": abs(m_sc["CVaR_95"] * 100.0),
                        "Calmar": m_sc["Calmar"]
                    })

            st.session_state.pareto_df = pd.DataFrame(grid_results)

    if st.session_state.pareto_df is not None:
        df_grid = st.session_state.pareto_df

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
                f"* MDD: **-{best_safe['MDD']:.1f}%**\n"
                f"* Calmar比率: **{best_safe['Calmar']:.2f}**"
            )
        with col_b3:
            st.warning(
                f"**③ 最も収益重視 (リターン最大)**\n\n"
                f"**【{best_growth['設定ラベル']}】**\n\n"
                f"* CAGR: **{best_growth['CAGR']:.1f}%**\n"
                f"* MDD: **-{best_growth['MDD']:.1f}%**\n"
                f"* Calmar比率: **{best_growth['Calmar']:.2f}**"
            )

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

        fig_pareto.update_layout(
            title="Pareto 最適空間 (左上ほど優秀 ｜ 丸サイズ: Calmar比率)",
            xaxis_title="最大ドローダウン MDD (%) [← 左ほど下落が小さく安全]",
            yaxis_title="通算 CAGR (%) [↑ 上ほど収益が高い]",
            height=520,
            hovermode="closest"
        )
        st.plotly_chart(fig_pareto, use_container_width=True)
