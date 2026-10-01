"""
SDE-Engine Pro: 動的資金配分・クオンツポートフォリオ管理システム
レバレッジETFのボラティリティコントロールおよび過熱抑制ロジックを統合した本番運用向け実装
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
EXPENSE_RATIO_ANNUAL = 0.0095      # レバレッジETF推定年間経費率
CASH_YIELD_ANNUAL = 0.035          # 米ドルMMF年間想定利回り
TRADING_DAYS_PER_YEAR = 252        # 年間営業日数
EPSILON = 1e-9                     # ゼロ除算防止用微小量

ASSETS: Dict[str, Dict[str, Any]] = {
    "TQQQ": {
        "name": "TQQQ (NASDAQ 3倍)", "underlying": "QQQ", "type": "ハイテク",
        "leverage": 3.0, "sigma_target": 55.0, "u_vol_norm": 20.0, "color": "#00ba38"
    },
    "SPXL": {
        "name": "SPXL (S&P500 3倍)", "underlying": "SPY", "type": "米国全体",
        "leverage": 3.0, "sigma_target": 45.0, "u_vol_norm": 16.0, "color": "#619cff"
    },
    "SOXL": {
        "name": "SOXL (半導体 3倍)", "underlying": "SOXX", "type": "半導体",
        "leverage": 3.0, "sigma_target": 65.0, "u_vol_norm": 28.0, "color": "#f5b041"
    },
    "FAS": {
        "name": "FAS (金融株 3倍)", "underlying": "XLF", "type": "金融",
        "leverage": 3.0, "sigma_target": 50.0, "u_vol_norm": 18.0, "color": "#9b59b6"
    },
    "UGL": {
        "name": "UGL (ゴールド 2倍)", "underlying": "GLD", "type": "ゴールド",
        "leverage": 2.0, "sigma_target": 35.0, "u_vol_norm": 13.0, "color": "#f1c40f"
    },
}

# =============================================================
# 2. ページ基本設定 & UIコントローラー
# =============================================================
st.set_page_config(
    page_title="SDE-Engine Pro 動的資金配分システム",
    page_icon="⚡",
    layout="wide"
)

st.sidebar.title("⚡ SDE-Engine Pro")

period_options = {
    "5y (直近5年間)": "5y",
    "10y (直近10年間)": "10y",
    "15y (直近15年間)": "15y",
    "20y (直近20年間・リーマンショック含む)": "20y",
    "25y (直近25年間・ITバブル崩壊含む)": "25y",
    "max (取得可能全期間)": "max"
}

selected_period_label = st.sidebar.selectbox(
    "📅 バックテスト検証期間の選択",
    list(period_options.keys()),
    index=1
)
period_code = period_options[selected_period_label]

st.sidebar.markdown("---")
st.sidebar.markdown("### 🛡️ リスク管理・集中投資上限")
max_cap_pct = st.sidebar.slider(
    "1銘柄あたりの最大投資上限 (推奨: 50%)",
    min_value=30, max_value=100, value=50, step=5
)
max_cap = max_cap_pct / 100.0

st.sidebar.info(f"""
**【動的配分・リスク抑制仕様】**
* 投資適格銘柄のみに動的集中（1銘柄上限 **{max_cap_pct}%**）。
* 50日EMA割れ時は即座に **0% (待機)** に遮断。
* 9日RSI > 70 または 50日EMA乖離 > +8% で過熱抑制（高値掴み防止・部分利確）。
""")

# =============================================================
# 3. データ取得 & 幾何整合合成データ生成エンジン
# =============================================================
def _sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    """MultiIndexの解除・タイムゾーン正規化・主要列の検証を行う"""
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    # tz-aware / tz-naive の不整合を完全解消
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in required_cols:
        if col not in df.columns and "Close" in df.columns:
            df[col] = df["Close"]
    return df.dropna(subset=["Close"])

@st.cache_data(ttl=3600, show_spinner=False)
def load_and_sync_market_data(period_str: str) -> Tuple[Dict[str, pd.DataFrame], pd.DatetimeIndex]:
    """
    全銘柄の取得、共通インデックス整流、未上場期間のフォワード幾何スケーリング合成
    """
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
        raise RuntimeError("基準母体データ (SPY) の取得に失敗しました。時間をおいて再試行してください。")

    # 全母体インデックスの交差を取得（基準期間の確定）
    base_idx = raw_data["SPY"].index
    for u in u_tickers:
        if u in raw_data and not raw_data[u].empty:
            base_idx = base_idx.intersection(raw_data[u].index)

    if len(base_idx) < 50:
        raise ValueError("検証に必要なデータ長が不足しています（50営業日未満）。")

    cleaned_data: Dict[str, pd.DataFrame] = {}
    for u in u_tickers:
        cleaned_data[u] = raw_data[u].loc[base_idx].copy()

    # レバレッジETFデータの構築（未上場期間の幾何スケーリング合成）
    for sym, cfg in ASSETS.items():
        u_sym = cfg["underlying"]
        lev = cfg["leverage"]
        df_u = cleaned_data[u_sym]
        
        # 母体の日次幾何リターンから信託報酬控除後のレバレッジリターンを算出
        u_ret = df_u["Close"].pct_change().fillna(0.0)
        daily_expense = EXPENSE_RATIO_ANNUAL / TRADING_DAYS_PER_YEAR
        syn_ret = (u_ret * lev) - daily_expense

        # 累積成長ファクター
        cum_growth = (1.0 + syn_ret).cumprod()

        has_real_etf = (sym in raw_data) and (not raw_data[sym].empty)
        real_idx = raw_data[sym].index.intersection(base_idx) if has_real_etf else pd.DatetimeIndex([])

        if len(real_idx) > 0:
            df_real = raw_data[sym].loc[real_idx]
            first_real_date = real_idx[0]
            first_real_price = float(df_real["Close"].iloc[0])
            growth_anchor = float(cum_growth.loc[first_real_date])
            scale = first_real_price / np.maximum(growth_anchor, EPSILON)

            # 未上場期間の合成価格を実データ起点に整合
            synth_close = cum_growth * scale
            full_close = synth_close.copy()
            full_close.loc[real_idx] = df_real["Close"]

            # High / Low の統合：実上場期間は本物の市場価格、未上場期間は母体のボラティリティ比率から近似
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
            # ETF実データが完全に存在しない場合は全期間合成
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
    """最新USD/JPYレートの安全取得"""
    try:
        fx = yf.download("USDJPY=X", period="5d", interval="1d", progress=False)
        fx = _sanitize_df(fx)
        if not fx.empty and "Close" in fx.columns:
            return round(float(fx["Close"].iloc[-1]), 2)
    except Exception:
        pass
    return 155.0

# データ初期化
with st.spinner(f"市場データ（{selected_period_label}）を取得・同期解析中..."):
    try:
        market_data, common_idx = load_and_sync_market_data(period_code)
        latest_fx_rate = get_usdjpy_rate()
    except Exception as e:
        st.error(f"データ取得中にエラーが発生しました: {str(e)}")
        st.stop()

latest_date_str = common_idx[-1].strftime('%Y年%m月%d日')

# =============================================================
# 4. 高速クオンツシグナル解析エンジン
# =============================================================
def calc_parkinson_vol_vectorized(df: pd.DataFrame, window: int = 10) -> np.ndarray:
    """Parkinson極値ボラティリティのベクトル計算（高速化・NaN安全処理）"""
    h = np.maximum(df["High"].values, EPSILON)
    l = np.maximum(df["Low"].values, EPSILON)
    hl_ratio_sq = (np.log(h / l)) ** 2
    factor = 1.0 / (4.0 * np.log(2.0))
    # 移動平均をNumPy畳み込み/Pandasで高速計算
    rolling_var = pd.Series(hl_ratio_sq * factor).rolling(window=window, min_periods=1).mean().values
    pv = np.sqrt(np.maximum(rolling_var, 0.0)) * np.sqrt(TRADING_DAYS_PER_YEAR) * 100.0
    return np.where(np.isnan(pv), 20.0, pv)

def calc_rsi_vectorized(series: pd.Series, period: int = 9) -> np.ndarray:
    """Wilder法に基づく修正RSIの完全ベクトル計算"""
    delta = series.diff().values
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    # 指数平滑移動平均 (Wilder's alpha = 1 / period)
    alpha = 1.0 / period
    avg_gain = pd.Series(gain).ewm(alpha=alpha, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=alpha, adjust=False).mean().values

    rs = avg_gain / np.maximum(avg_loss, EPSILON)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return np.nan_to_num(rsi, nan=50.0)

def run_asset_signals(sym: str) -> Dict[str, Any]:
    """銘柄ごとの確率モデル判定・過熱抑制・防衛ラインの統合評価"""
    cfg = ASSETS[sym]
    df_u = market_data[cfg["underlying"]].loc[common_idx]
    df_etf = market_data[sym].loc[common_idx]

    c_u = df_u["Close"].values
    c_etf = df_etf["Close"].values

    sigma_local = calc_parkinson_vol_vectorized(df_etf, window=10)

    # トレンド指標 (EMA50 / EMA200)
    ema50 = pd.Series(c_u).ewm(span=50, adjust=False).mean().values
    ema200 = pd.Series(c_u).ewm(span=200, adjust=False).mean().values
    diff_50 = (c_u - ema50) / np.maximum(ema50, EPSILON)
    diff_200 = (c_u - ema200) / np.maximum(ema200, EPSILON)
    combined_trend = 0.60 * diff_50 + 0.40 * diff_200

    # ボラティリティ適応スコア
    u_ret_series = pd.Series(c_u).pct_change().fillna(0.0)
    vol_20 = (u_ret_series.rolling(20, min_periods=1).std().values * np.sqrt(TRADING_DAYS_PER_YEAR) * 100.0)
    vol_20 = np.nan_to_num(vol_20, nan=cfg["u_vol_norm"])
    vol_score = -(vol_20 - cfg["u_vol_norm"]) / cfg["u_vol_norm"]
    adjusted_vol_score = np.where(combined_trend < 0, np.minimum(vol_score, 0.0), vol_score)

    # 9日RSI & 過熱ペナルティ
    rsi9 = calc_rsi_vectorized(pd.Series(c_u), period=9)
    penalty_rsi = np.where(rsi9 > 70.0, (rsi9 - 70.0) / 20.0, 0.0)
    penalty_ema = np.where(diff_50 > 0.08, (diff_50 - 0.08) / 0.10, 0.0)
    overheat_penalty = np.clip(penalty_rsi + penalty_ema, 0.0, 0.60)

    # シグモイド確率変換（オーバーフロー耐性付与）
    logit = 6.0 * combined_trend + 1.5 * adjusted_vol_score - 3.5 * overheat_penalty
    logit_clipped = np.clip(logit, -50.0, 50.0)
    p_bull = 1.0 / (1.0 + np.exp(-logit_clipped))

    # 防衛ライン・ハードキル判定
    below_50 = diff_50 < -0.01
    below_200_only = (c_u < ema200) & (~below_50)

    p_bull = np.where(below_50, np.minimum(p_bull, 0.20), p_bull)
    p_bull = np.where(below_200_only & (diff_50 < 0.03), np.minimum(p_bull, 0.30), p_bull)

    # 目標投資比率 W* 算出
    vol_adj = np.clip(cfg["sigma_target"] / np.maximum(sigma_local, 1e-4), 0.2, 1.2)
    w_star = np.clip(p_bull * vol_adj, 0.0, 1.0)
    w_star = w_star * (1.0 - overheat_penalty)
    w_star = np.where(below_50, 0.0, w_star)

    # フェーズゲート適用
    factor = np.where(
        (p_bull < 0.35) | (w_star < 0.10),
        0.0,
        np.where(p_bull < 0.55, 0.50, 1.0)
    )
    raw_desired_w = w_star * factor

    # ETF日次リターン
    etf_ret = np.zeros_like(c_etf)
    etf_ret[1:] = (c_etf[1:] - c_etf[:-1]) / np.maximum(c_etf[:-1], EPSILON)

    # 最新ステータス判定
    latest_p = p_bull[-1]
    latest_w = w_star[-1]
    latest_diff50 = diff_50[-1]
    latest_diff200 = diff_200[-1]
    latest_rsi = rsi9[-1]
    latest_penalty = overheat_penalty[-1]
    latest_below50 = below_50[-1]
    latest_below200 = (c_u[-1] < ema200[-1])

    if latest_below50:
        badge = "🔴 弱気防衛 (待機)"
        desc = "50日EMA割れ / 全面キャッシュ退避"
        phase_type = "Defense"
    elif latest_below200 and latest_p < 0.35:
        badge = "🔴 長期弱気 (待機)"
        desc = "200日線下での調整局面 / キャッシュ待機"
        phase_type = "Defense"
    elif latest_p < 0.35:
        badge = "🔴 弱気警戒 (待機)"
        desc = "ボラティリティ過大・下落リスク優勢"
        phase_type = "Defense"
    elif latest_penalty > 0.15:
        badge = "⚠️ 短期過熱警戒 (利確・抑制)"
        desc = f"RSI {latest_rsi:.0f}・買われすぎ抑制中 (-{latest_penalty*100:.0f}%)"
        phase_type = "Overheated"
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
        "sigma_local": sigma_local, "ema50": ema50, "ema200": ema200, "rsi9": rsi9,
        "c_u": c_u, "c_etf": c_etf, "etf_ret": etf_ret,
        "latest_p": latest_p, "latest_w": latest_w, "latest_sigma": sigma_local[-1],
        "latest_diff50": latest_diff50, "latest_diff200": latest_diff200,
        "latest_rsi": latest_rsi, "latest_penalty": latest_penalty,
        "badge": badge, "desc": desc, "phase_type": phase_type,
        "u_close": c_u[-1], "etf_price": c_etf[-1]
    }

asset_keys = list(ASSETS.keys())
signals = {s: run_asset_signals(s) for s in asset_keys}

# =============================================================
# 5. 完全ベクトル化 動的ポートフォリオ配分エンジン
# =============================================================
n_days = len(common_idx)
raw_matrix = np.column_stack([signals[k]["raw_w"] for k in asset_keys])  # shape: (n_days, 5)

# 動的キャップ・正規化演算のベクトル化
active_mask = raw_matrix > 0.05
filtered_matrix = np.where(active_mask, raw_matrix, 0.0)
sum_weights = np.sum(filtered_matrix, axis=1, keepdims=True)

# 合計が1.0を超える場合は正規化、1.0以下は未投資現金を維持
norm_scaler = np.maximum(sum_weights, 1.0)
cand_matrix = filtered_matrix / norm_scaler
daily_alloc_matrix = np.minimum(cand_matrix, max_cap)

# 前日終値確定配分による翌日執行（ルックアヘッドバイアス排除）
exec_matrix = np.zeros_like(daily_alloc_matrix)
exec_matrix[1:] = daily_alloc_matrix[:-1]

# バックテストリターン合成
daily_cash_rate = CASH_YIELD_ANNUAL / TRADING_DAYS_PER_YEAR
ret_matrix = np.column_stack([signals[k]["etf_ret"] for k in asset_keys])

strat_daily_ret = np.sum(exec_matrix * ret_matrix, axis=1)
cash_ratio_daily = np.maximum(0.0, 1.0 - np.sum(exec_matrix, axis=1))
strat_daily_ret += cash_ratio_daily * daily_cash_rate

# 等金額ベンチマーク
bm_daily_ret = np.mean(ret_matrix, axis=1)

cum_strat = np.cumprod(1.0 + strat_daily_ret)
cum_bm = np.cumprod(1.0 + bm_daily_ret)

# パフォーマンス統計指標の安全計算
years = max(n_days / TRADING_DAYS_PER_YEAR, 0.1)
strat_cagr = (cum_strat[-1] ** (1.0 / years)) - 1.0
bm_cagr = (cum_bm[-1] ** (1.0 / years)) - 1.0

peak_strat = np.maximum.accumulate(cum_strat)
dd_strat = (cum_strat - peak_strat) / np.maximum(peak_strat, EPSILON)
strat_mdd = float(np.min(dd_strat))

peak_bm = np.maximum.accumulate(cum_bm)
dd_bm = (cum_bm - peak_bm) / np.maximum(peak_bm, EPSILON)
bm_mdd = float(np.min(dd_bm))

strat_vol = float(np.std(strat_daily_ret, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
strat_calmar = (strat_cagr / abs(strat_mdd)) if abs(strat_mdd) > EPSILON else 0.0
bm_calmar = (bm_cagr / abs(bm_mdd)) if abs(bm_mdd) > EPSILON else 0.0

pos_rets = strat_daily_ret[strat_daily_ret > 0]
neg_rets = strat_daily_ret[strat_daily_ret < 0]
strat_pf = (np.sum(pos_rets) / abs(np.sum(neg_rets))) if len(neg_rets) > 0 and abs(np.sum(neg_rets)) > EPSILON else 0.0

latest_alloc = daily_alloc_matrix[-1]
tot_latest_invested = float(np.sum(latest_alloc))
tot_latest_cash = max(0.0, 1.0 - tot_latest_invested)

# =============================================================
# 6. ダッシュボード・プレゼンテーション層
# =============================================================
st.title("⚡ SDE-Engine Pro 動的資金配分ダッシュボード")
st.caption(f"検証範囲: **{selected_period_label}** （計 {n_days} 営業日 ｜ 最終確定: {latest_date_str}）")

with st.expander("📖 【運用・アルゴリズム仕様ガイド】", expanded=False):
    st.markdown("""
    * **EMA乖離フィルター**: 株価が50日EMA未満に転落した瞬間に即座にリスク資産を遮断（キャッシュ待機）。
    * **過熱ペナルティ**: 9日RSI > 70、または50日EMA乖離 > +8% で投資比率を最大60%圧縮。天井掴みを防止。
    * **動的集中 & キャップ**: 有望銘柄に比率を集中させつつ、1銘柄上限（設定値: 50%）で過度な集中リスクを統制。
    * **キャッシュ待機益**: 投資待機枠は年利3.5%の米ドルMMFで自動複利運用。
    """)

c_k1, c_k2, c_k3 = st.columns([1.5, 1, 1.2])
c_k1.markdown("#### 🛡️ カタストロフィ・キルスイッチ状態")
c_k2.success("● 正常稼働中 (Normal)")
c_k3.caption("※夜間先物 -4.5% 急落 または 日中NAV -8% 突破で手動強制全決済")

st.markdown("---")

tab1, tab2, tab3 = st.tabs([
    "🏛️ 今夜の最適配分 ＆ 執行シミュレーター",
    "🧪 長期バックテスト検証（5年〜最大期間）",
    "📊 銘柄別詳細分析（トレンド・RSI・過熱度）"
])

# -------------------------------------------------------------
# TAB 1: 今夜の最適配分 ＆ 楽天証券 執行シミュレーター
# -------------------------------------------------------------
with tab1:
    st.info(f"📌 **判定確定日: {latest_date_str}（NY終値ベース）**")
    st.subheader("📊 銘柄別シグナル判定マトリクス")

    cols = st.columns(len(asset_keys))
    for idx_c, sym in enumerate(asset_keys):
        sig = signals[sym]
        alloc_ratio = latest_alloc[idx_c]

        with cols[idx_c]:
            st.markdown(f"#### {sym}")
            st.caption(f"{ASSETS[sym]['name']}")
            st.metric(f"{ASSETS[sym]['underlying']} 終値", f"${sig['u_close']:.2f}")

            st.write(f"強気確率 $P(\\text{{Bull}})$: **{sig['latest_p']*100:.1f}%**")
            st.progress(float(np.clip(sig["latest_p"], 0.0, 1.0)))

            st.caption(f"**9日RSI:** {sig['latest_rsi']:.1f}")
            st.caption(f"**50日比:** {sig['latest_diff50']*100:+.1f}% ｜ **200日比:** {sig['latest_diff200']*100:+.1f}%")
            st.caption(f"Parkinson Vol: {sig['latest_sigma']:.1f}%")

            st.info(f"**{sig['badge']}**\n\n*{sig['desc']}*")

            if alloc_ratio > 0:
                st.success(f"**推奨比率:**\n\n### {alloc_ratio*100:.1f} %")
            else:
                st.warning("**配分:**\n\n### 0.0 % (待機)")

    st.markdown("---")

    c_s1, c_s2 = st.columns([1.5, 1])
    with c_s1:
        st.markdown("### 💼 ポートフォリオ全体の実効配分")
        m_c1, m_c2 = st.columns(2)
        m_c1.metric("総株式・ゴールド投資比率", f"{tot_latest_invested*100:.1f} %")
        m_c2.metric("米ドルMMF待機比率", f"{tot_latest_cash*100:.1f} %")
        st.caption(f"※1銘柄上限: {max_cap_pct}%。シグナル未充足分は米ドルMMF（年利約3.5%想定）へ配分。")

    with c_s2:
        active_labels = [asset_keys[i] for i in range(len(asset_keys)) if latest_alloc[i] > 0] + ["米ドルMMF"]
        active_vals = [latest_alloc[i] * 100 for i in range(len(asset_keys)) if latest_alloc[i] > 0] + [tot_latest_cash * 100]
        fig_pie = go.Figure(data=[go.Pie(
            labels=active_labels, values=active_vals, hole=.45,
            marker_colors=['#00ba38', '#619cff', '#f5b041', '#9b59b6', '#f1c40f', '#95a5a6']
        )])
        fig_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=220)
        st.plotly_chart(fig_pie, use_container_width=True)

    # 執行シミュレーター
    st.markdown("---")
    st.subheader("💡 証券会社 寄り付き発注シミュレーター")

    curr_c1, curr_c2, curr_c3 = st.columns([1.2, 1.5, 1.3])
    with curr_c1:
        in_curr = st.radio("入力通貨単位", ["日本円 (万円)", "米ドル (USD)"], horizontal=True)
    with curr_c2:
        if in_curr == "日本円 (万円)":
            f_jpy_man = st.number_input("運用総資金額（万円）", min_value=10, value=300, step=10)
        else:
            f_usd_in = st.number_input("運用総資金額（USD）", min_value=1000, value=20000, step=1000)
    with curr_c3:
        fx_val = st.number_input("適用為替レート (USD/JPY)", value=latest_fx_rate, min_value=50.0, max_value=300.0, step=0.5, format="%.2f")

    if in_curr == "日本円 (万円)":
        total_usd = (f_jpy_man * 10000.0) / fx_val
        total_jpy = f_jpy_man * 10000.0
    else:
        total_usd = float(f_usd_in)
        total_jpy = total_usd * fx_val

    st.success(f"💰 **運用総資産**: **${total_usd:,.2f}** ＝ **約 {total_jpy:,.0f} 円** （{total_jpy/10000:,.1f} 万円）")

    sim_rows = []
    for idx_c, sym in enumerate(asset_keys):
        sig = signals[sym]
        alloc_ratio = latest_alloc[idx_c]
        p = sig["etf_price"]

        target_usd = total_usd * alloc_ratio
        target_jpy = target_usd * fx_val
        target_shares = int(target_usd // p) if p > 0 else 0

        if alloc_ratio == 0:
            action = "待機・買付なし (保有中は全売却)"
        elif sig["phase_type"] == "Overheated":
            action = f"⚠️ 過熱警戒: 目標 {target_shares} 株へ調整 (部分利確)"
        elif sig["phase_type"] == "Scout":
            action = f"🟡 探査打診: {target_shares} 株を買付"
        else:
            action = f"🟢 本玉巡航: {target_shares} 株に調整"

        sim_rows.append({
            "銘柄": ASSETS[sym]["name"],
            "判定ステータス": sig["badge"],
            "最適配分比率": f"{alloc_ratio*100:.1f} %",
            "目標投資額 (USD)": f"${target_usd:,.2f}",
            "目標投資額 (JPY)": f"約 {target_jpy:,.0f} 円" if alloc_ratio > 0 else "0 円",
            "参考株価": f"${p:.2f}",
            "目標保有株数": f"{target_shares} 株",
            "推奨執行アクション": action
        })

    c_usd = total_usd * tot_latest_cash
    c_jpy = c_usd * fx_val
    sim_rows.append({
        "銘柄": "米ドル現金 / MMF",
        "判定ステータス": "🛡️ 安全待機",
        "最適配分比率": f"{tot_latest_cash*100:.1f} %",
        "目標投資額 (USD)": f"${c_usd:,.2f}",
        "目標投資額 (JPY)": f"約 {c_jpy:,.0f} 円",
        "参考株価": "-",
        "目標保有株数": "-",
        "推奨執行アクション": "外貨MMFで利息享受 (待機)"
    })

    st.dataframe(pd.DataFrame(sim_rows), use_container_width=True, hide_index=True)

# -------------------------------------------------------------
# TAB 2: パフォーマンス検証 ＆ 統計指標
# -------------------------------------------------------------
with tab2:
    st.subheader(f"🧪 バックテスト検証結果 （{selected_period_label}）")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("年平均リターン (CAGR)", f"{strat_cagr*100:.1f} %", f"ベンチマーク: {bm_cagr*100:.1f}%")
    m2.metric("年率ボラティリティ", f"{strat_vol*100:.1f} %")
    m3.metric("最大下落率 (MDD)", f"{strat_mdd*100:.1f} %", f"ベンチマーク: {bm_mdd*100:.1f}%")
    m4.metric("カルマーレシオ", f"{strat_calmar:.2f}", f"ベンチマーク: {bm_calmar:.2f}")
    m5.metric("プロフィットファクター", f"{strat_pf:.2f}")

    st.markdown("---")
    fig_bt = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.7, 0.3]
    )
    fig_bt.add_trace(
        go.Scatter(x=common_idx, y=cum_strat, name="SDE Pro 戦略", line=dict(color="#00ba38", width=2.5)),
        row=1, col=1
    )
    fig_bt.add_trace(
        go.Scatter(x=common_idx, y=cum_bm, name="5銘柄等金額ホールド", line=dict(color="#888888", width=1.5, dash="dot")),
        row=1, col=1
    )
    fig_bt.add_trace(
        go.Scatter(x=common_idx, y=dd_strat * 100, name="戦略DD (%)", fill="tozeroy", line=dict(color="#d62728", width=1)),
        row=2, col=1
    )
    fig_bt.add_trace(
        go.Scatter(x=common_idx, y=dd_bm * 100, name="ベンチマークDD (%)", line=dict(color="#7f7f7f", width=1, dash="dash")),
        row=2, col=1
    )
    fig_bt.update_layout(height=520, margin=dict(t=20, b=20, l=10, r=10), hovermode="x unified")
    fig_bt.update_yaxes(title_text="資産成長倍率", row=1, col=1)
    fig_bt.update_yaxes(title_text="下落率 (%)", row=2, col=1)
    st.plotly_chart(fig_bt, use_container_width=True)

# -------------------------------------------------------------
# TAB 3: 銘柄別詳細分析
# -------------------------------------------------------------
with tab3:
    st.subheader("📊 個別銘柄トレンド・モメンタム・過熱度分析")
    selected_asset = st.radio(
        "分析対象銘柄",
        asset_keys,
        format_func=lambda x: f"{x} ({ASSETS[x]['name']})",
        horizontal=True
    )

    sig = signals[selected_asset]
    cfg = ASSETS[selected_asset]
    u_sym = cfg["underlying"]

    fig_single = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.45, 0.30, 0.25],
        subplot_titles=(
            f"① 母体指数 {u_sym} 終値 ＆ 50日/200日 EMA",
            f"② 母体指数 9日RSI (買われすぎ: 70 ｜ 売られすぎ: 30)",
            f"③ 強気確率 P(Bull) ＆ 目標保有比率 W*"
        )
    )

    fig_single.add_trace(
        go.Scatter(x=common_idx, y=sig["c_u"], name=f"{u_sym} 終値", line=dict(color=cfg["color"], width=1.5)),
        row=1, col=1
    )
    fig_single.add_trace(
        go.Scatter(x=common_idx, y=sig["ema50"], name="50日 EMA", line=dict(color="#17becf", width=1.5)),
        row=1, col=1
    )
    fig_single.add_trace(
        go.Scatter(x=common_idx, y=sig["ema200"], name="200日 EMA", line=dict(color="#ff7f0e", width=1.8)),
        row=1, col=1
    )

    fig_single.add_trace(
        go.Scatter(x=common_idx, y=sig["rsi9"], name="9日 RSI", line=dict(color="#9467bd", width=1.5)),
        row=2, col=1
    )
    fig_single.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig_single.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    fig_single.add_trace(
        go.Scatter(x=common_idx, y=sig["p_bull"] * 100, name="強気確率 P(Bull) %", line=dict(color="#1f77b4", width=1.5)),
        row=3, col=1
    )
    fig_single.add_trace(
        go.Scatter(x=common_idx, y=sig["w_star"] * 100, name="目標比率 W* %", line=dict(color="#d62728", width=1.5)),
        row=3, col=1
    )

    fig_single.update_layout(height=720, margin=dict(t=30, b=20, l=10, r=10), hovermode="x unified")
    fig_single.update_yaxes(title_text="価格 ($)", row=1, col=1)
    fig_single.update_yaxes(title_text="RSI", row=2, col=1)
    fig_single.update_yaxes(title_text="比率 (%)", row=3, col=1)
    st.plotly_chart(fig_single, use_container_width=True)
