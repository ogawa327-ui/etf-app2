import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# PCワイド画面設定
st.set_page_config(
    page_title="米国レバレッジETF 科学的検証・自動最適化ダッシュボード",
    page_icon="🔬",
    layout="wide"
)

# -------------------------------------------------------------
# 1. データ取得（キャッシュ保持）
# -------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_all_market_data():
    tickers = ["QQQ", "SOXX", "TQQQ", "SOXL"]
    data = {}
    for t in tickers:
        df = yf.download(t, period="5y", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        data[t] = df.dropna()
    return data

with st.spinner("市場データを読み込み中..."):
    raw_data = load_all_market_data()

qqq_raw = raw_data["QQQ"]
soxx_raw = raw_data["SOXX"]
tqqq_raw = raw_data["TQQQ"]
soxl_raw = raw_data["SOXL"]

common_idx = qqq_raw.index.intersection(soxx_raw.index).intersection(tqqq_raw.index).intersection(soxl_raw.index)

# -------------------------------------------------------------
# 2. セッション状態の初期化（スライダーと連動）
# -------------------------------------------------------------
if "tqqq_weight" not in st.session_state:
    st.session_state["tqqq_weight"] = 70
if "ema_span" not in st.session_state:
    st.session_state["ema_span"] = 200
if "qqq_hv_thresh" not in st.session_state:
    st.session_state["qqq_hv_thresh"] = 28.0
if "soxx_hv_thresh" not in st.session_state:
    st.session_state["soxx_hv_thresh"] = 40.0
if "regime2_alloc" not in st.session_state:
    st.session_state["regime2_alloc"] = 50
if "opt_msg" not in st.session_state:
    st.session_state["opt_msg"] = None

# -------------------------------------------------------------
# 3. 高速バックテスト & 自動最適化エンジン
# -------------------------------------------------------------
tqqq_ret_arr = tqqq_raw.loc[common_idx, "Close"].pct_change().fillna(0).values
soxl_ret_arr = soxl_raw.loc[common_idx, "Close"].pct_change().fillna(0).values
q_close_arr = qqq_raw.loc[common_idx, "Close"].values
s_close_arr = soxx_raw.loc[common_idx, "Close"].values

# 事前ボラティリティ計算
q_hv_arr = (np.log(qqq_raw.loc[common_idx, "Close"] / qqq_raw.loc[common_idx, "Close"].shift(1)).rolling(20).std() * np.sqrt(252) * 100).fillna(0).values
s_hv_arr = (np.log(soxx_raw.loc[common_idx, "Close"] / soxx_raw.loc[common_idx, "Close"].shift(1)).rolling(20).std() * np.sqrt(252) * 100).fillna(0).values

def fast_eval(tw, span, q_hvt, s_hvt, r2):
    sw = 1.0 - tw
    # EMA計算
    alpha = 2.0 / (span + 1.0)
    # 高速指数平滑化
    q_ema = pd.Series(q_close_arr).ewm(alpha=alpha, adjust=False).mean().values
    s_ema = pd.Series(s_close_arr).ewm(alpha=alpha, adjust=False).mean().values
    
    q_bull = (q_close_arr > q_ema) & (q_hv_arr < q_hvt)
    q_warn = (q_close_arr > q_ema) & (q_hv_arr >= q_hvt)
    q_pos = np.where(q_bull, 1.0, np.where(q_warn, r2, 0.0))
    
    s_bull = (s_close_arr > s_ema) & (s_hv_arr < s_hvt)
    s_warn = (s_close_arr > s_ema) & (s_hv_arr >= s_hvt)
    s_pos = np.where(s_bull, 1.0, np.where(s_warn, r2, 0.0))
    
    # 翌日執行 (shift 1)
    q_exec = np.zeros_like(q_pos)
    q_exec[1:] = q_pos[:-1]
    s_exec = np.zeros_like(s_pos)
    s_exec[1:] = s_pos[:-1]
    
    strat_ret = (tw * q_exec * tqqq_ret_arr) + (sw * s_exec * soxl_ret_arr)
    cum = np.cumprod(1.0 + strat_ret)
    
    n = len(strat_ret)
    yrs = n / 252.0
    cagr = cum[-1] ** (1.0 / yrs) - 1.0
    
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    mdd = np.min(dd)
    
    calmar = cagr / abs(mdd) if mdd != 0 else 0
    gains = np.sum(strat_ret[strat_ret > 0])
    losses = np.abs(np.sum(strat_ret[strat_ret < 0]))
    pf = gains / losses if losses != 0 else 0
    vol = np.std(strat_ret) * np.sqrt(252)
    t_stat = (np.mean(strat_ret) / (np.std(strat_ret) / np.sqrt(n))) if np.std(strat_ret) != 0 else 0
    
    return {
        "CAGR": cagr, "Vol": vol, "MDD": mdd, "PF": pf, "Calmar": calmar, "t_stat": t_stat,
        "cum": cum, "dd": dd, "q_pos_last": q_pos[-1], "s_pos_last": s_pos[-1],
        "q_ema_last": q_ema[-1], "s_ema_last": s_ema[-1],
        "q_hv_last": q_hv_arr[-1], "s_hv_last": s_hv_arr[-1]
    }

def run_grid_search(target_metric="カルマーレシオ"):
    # 探索パラメータ空間
    t_weights = [1.0, 0.85, 0.70, 0.60, 0.50]
    spans = [150, 175, 200, 220]
    q_hvs = [24.0, 27.0, 30.0]
    s_hvs = [35.0, 40.0, 45.0]
    r2_vals = [0.0, 0.3, 0.5]
    
    best_score = -9999
    best_params = None
    best_metrics = None
    
    for tw in t_weights:
        for sp in spans:
            for qh in q_hvs:
                for sh in s_hvs:
                    for r2 in r2_vals:
                        res_temp = fast_eval(tw, sp, qh, sh, r2)
                        score = res_temp["Calmar"] if target_metric == "カルマーレシオ" else res_temp["PF"]
                        if score > best_score:
                            best_score = score
                            best_params = (tw, sp, qh, sh, r2)
                            best_metrics = res_temp
                            
    return best_params, best_metrics

# -------------------------------------------------------------
# 4. サイドバー：自動最適化 & パラメータ調整
# -------------------------------------------------------------
st.sidebar.title("🔬 パラメータ最適化ラボ")

st.sidebar.markdown("### 🤖 自動最適化 (Auto-Optimizer)")
opt_target = st.sidebar.selectbox(
    "最適化の目標指標",
    ["カルマーレシオ（下落抑制＆リターン効率）", "プロフィットファクター（利益損失比）"]
)

if st.sidebar.button("🚀 最適パラメータを自動探索・適用", type="primary", use_container_width=True):
    with st.spinner("全パラメータ空間を一括解析中..."):
        target_name = "カルマーレシオ" if "カルマー" in opt_target else "PF"
        best_p, best_m = run_grid_search(target_name)
        
        # セッションに適用
        st.session_state["tqqq_weight"] = int(best_p[0] * 100)
        st.session_state["ema_span"] = int(best_p[1])
        st.session_state["qqq_hv_thresh"] = float(best_p[2])
        st.session_state["soxx_hv_thresh"] = float(best_p[3])
        st.session_state["regime2_alloc"] = int(best_p[4] * 100)
        
        st.session_state["opt_msg"] = f"""
        **✅ 最適化完了！（目標: {target_name}最大化）**
        * TQQQ比率: **{int(best_p[0]*100)}%** ｜ SOXL比率: **{int((1-best_p[0])*100)}%**
        * EMA日数: **{best_p[1]}日**
        * ボラ閾値: QQQ **{best_p[2]}%** ｜ SOXX **{best_p[3]}%**
        * 警戒時比率: **{int(best_p[4]*100)}%**
        * 達成数値: **カルマー {best_m['Calmar']:.2f}** ｜ **PF {best_m['PF']:.2f}** ｜ **MDD {best_m['MDD']*100:.1f}%**
        """
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ 手動微調整スライダー")

# スライダー（セッション状態と連動）
tqqq_w = st.sidebar.slider("① TQQQ 配分比率 (%)", 0, 100, step=5, key="tqqq_weight") / 100.0
soxl_w = 1.0 - tqqq_w
st.sidebar.caption(f"配分: TQQQ **{int(tqqq_w*100)}%** ｜ SOXL **{int(soxl_w*100)}%**")

ema_s = st.sidebar.slider("② EMA 計算日数 (共通)", 100, 250, step=5, key="ema_span")
q_hvt = st.sidebar.slider("③ QQQ 警戒閾値 (%)", 20.0, 35.0, step=1.0, key="qqq_hv_thresh")
s_hvt = st.sidebar.slider("  SOXX 警戒閾値 (%)", 30.0, 50.0, step=1.0, key="soxx_hv_thresh")
r2_a = st.sidebar.slider("④ レジーム2（警戒時）の投資比率 (%)", 0, 100, step=10, key="regime2_alloc") / 100.0

# -------------------------------------------------------------
# 5. 現在選択パラメータのバックテスト評価
# -------------------------------------------------------------
res = fast_eval(tqqq_w, ema_s, q_hvt, s_hvt, r2_a)

# ベンチマーク
bm_strat = (tqqq_w * tqqq_ret_arr) + (soxl_w * soxl_ret_arr)
bm_cum = np.cumprod(1.0 + bm_strat)
bm_cagr = bm_cum[-1] ** (1.0 / (len(common_idx)/252.0)) - 1.0
bm_peak = np.maximum.accumulate(bm_cum)
bm_mdd = np.min((bm_cum - bm_peak) / bm_peak)

# -------------------------------------------------------------
# 6. メイン画面レイアウト
# -------------------------------------------------------------
st.title("🛡️ 米国レバレッジETF 科学的最適化ダッシュボード")
st.caption(f"データ検証期間: 直近5年 ｜ データ更新日: {common_idx[-1].strftime('%Y/%m/%d')}")

# 最適化実行時の完了メッセージ表示
if st.session_state["opt_msg"]:
    st.success(st.session_state["opt_msg"])

tab1, tab2, tab3 = st.tabs([
    "🧪 パフォーマンス検証 & 統計指標", 
    "🏛️ 現在のシグナル & 楽天証券発注計算",
    "📈 テクニカルチャート"
])

# =============================================================
# TAB 1: パフォーマンス検証
# =============================================================
with tab1:
    st.subheader("📊 現在設定での統計パフォーマンス")
    
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("期待年利 (CAGR)", f"{res['CAGR']*100:.1f} %", f"単主持: {bm_cagr*100:.1f}%")
    m2.metric("年率リスク (Vol)", f"{res['Vol']*100:.1f} %")
    m3.metric("最大下落率 (MDD)", f"{res['MDD']*100:.1f} %", f"単主持: {bm_mdd*100:.1f}%")
    m4.metric("プロフィットファクター (PF)", f"{res['PF']:.2f}")
    m5.metric("カルマーレシオ", f"{res['Calmar']:.2f}")
    m6.metric("t値 (有意性)", f"{res['t_stat']:.2f}")

    st.markdown("---")
    
    # 資産推移 & ドローダウン
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.7, 0.3])
    fig.add_trace(go.Scatter(x=common_idx, y=res["cum"], name="戦略資産曲線", line=dict(color="#00ba38", width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=common_idx, y=bm_cum, name="バイ＆ホールド (放置)", line=dict(color="#888888", width=1.5, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=common_idx, y=res["dd"] * 100, name="ドローダウン (%)", fill="tozeroy", line=dict(color="#d62728", width=1)), row=2, col=1)
    fig.update_layout(height=480, margin=dict(t=20, b=20, l=10, r=10), hovermode="x unified")
    fig.update_yaxes(title_text="資産倍率", row=1, col=1)
    fig.update_yaxes(title_text="下落率 (%)", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

# =============================================================
# TAB 2: 現在シグナル & 発注計算
# =============================================================
with tab2:
    st.subheader("🏛️ 現在の判定シグナルと推奨発注")
    q_latest_c = qqq_raw["Close"].iloc[-1]
    s_latest_c = soxx_raw["Close"].iloc[-1]
    
    col1, col2, col3 = st.columns([1.2, 1.2, 1.6])
    with col1:
        st.markdown(f"### TQQQ (枠: {int(tqqq_w*100)}%)")
        st.metric("QQQ 終値", f"${q_latest_c:.2f}")
        st.write(f"EMA({ema_s}日): **${res['q_ema_last']:.2f}**")
        st.write(f"HV20: **{res['q_hv_last']:.1f}%** (閾値: {q_hvt}%)")
        st.write(f"推奨比率: **{int(res['q_pos_last']*100)}%**")
        st.progress(res['q_pos_last'])
        
    with col2:
        st.markdown(f"### SOXL (枠: {int(soxl_w*100)}%)")
        st.metric("SOXX 終値", f"${s_latest_c:.2f}")
        st.write(f"EMA({ema_s}日): **${res['s_ema_last']:.2f}**")
        st.write(f"HV20: **{res['s_hv_last']:.1f}%** (閾値: {s_hvt}%)")
        st.write(f"推奨比率: **{int(res['s_pos_last']*100)}%**")
        st.progress(res['s_pos_last'])
        
    with col3:
        tot_eq = (tqqq_w * res['q_pos_last']) + (soxl_w * res['s_pos_last'])
        c_ratio = 1.0 - tot_eq
        st.markdown("### ポートフォリオ全体")
        st.metric("株式エクスポージャー", f"{tot_eq*100:.1f} %", f"待機キャッシュ: {c_ratio*100:.1f}%")
        fig_pie = go.Figure(data=[go.Pie(
            labels=['TQQQ', 'SOXL', '現金/MMF'],
            values=[tqqq_w * res['q_pos_last'] * 100, soxl_w * res['s_pos_last'] * 100, c_ratio * 100],
            hole=.4,
            marker_colors=['#00ba38', '#f5b041', '#b0b0b0']
        )])
        fig_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=170)
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")
    st.subheader("💡 楽天証券 寄り付き発注シミュレーター")
    c_f1, c_f2 = st.columns([1, 2])
    with c_f1:
        funds = st.number_input("運用総資金額（米ドル：USD）", min_value=1000, value=30000, step=1000)
    with c_f2:
        t_price = tqqq_raw["Close"].iloc[-1]
        s_price = soxl_raw["Close"].iloc[-1]
        t_val = funds * tqqq_w * res['q_pos_last']
        s_val = funds * soxl_w * res['s_pos_last']
        c_val = funds * c_ratio
        
        sim_data = {
            "銘柄": ["TQQQ", "SOXL", "米ドル現金 / MMF"],
            "目標金額": [f"${t_val:,.2f}", f"${s_val:,.2f}", f"${c_val:,.2f}"],
            "目標保有株数": [f"{int(t_val // t_price)} 株", f"{int(s_val // s_price)} 株", "-"],
            "アクション": [
                f"{int(t_val // t_price)} 株に調整" if t_val > 0 else "全売却 (0株)",
                f"{int(s_val // s_price)} 株に調整" if s_val > 0 else "全売却 (0株)",
                "MMF等で安全待機"
            ]
        }
        st.table(pd.DataFrame(sim_data))

# =============================================================
# TAB 3: テクニカルチャート
# =============================================================
with tab3:
    st.subheader("QQQ & SOXX テクニカル分析チャート")
    c_sub1, c_sub2 = st.columns(2)
    with c_sub1:
        st.markdown(f"**QQQ (NASDAQ100) ＆ {ema_s}日EMA**")
        fig_q = go.Figure()
        fig_q.add_trace(go.Scatter(x=common_idx, y=qqq_raw.loc[common_idx, "Close"], name="QQQ 終値", line=dict(color="#1f77b4")))
        fig_q.add_trace(go.Scatter(x=common_idx, y=pd.Series(q_close_arr).ewm(span=ema_s, adjust=False).mean(), name=f"{ema_s}日EMA", line=dict(color="#ff7f0e", width=2)))
        fig_q.update_layout(height=400, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_q, use_container_width=True)
    with c_sub2:
        st.markdown(f"**SOXX (半導体) ＆ {ema_s}日EMA**")
        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(x=common_idx, y=soxx_raw.loc[common_idx, "Close"], name="SOXX 終値", line=dict(color="#9467bd")))
        fig_s.add_trace(go.Scatter(x=common_idx, y=pd.Series(s_close_arr).ewm(span=ema_s, adjust=False).mean(), name=f"{ema_s}日EMA", line=dict(color="#ff7f0e", width=2)))
        fig_s.update_layout(height=400, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_s, use_container_width=True)
