import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# PCワイド画面設定
st.set_page_config(
    page_title="米国レバレッジETF 科学的検証ダッシュボード (TQQQ / SOXL)",
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

# 共通インデックス
common_idx = qqq_raw.index.intersection(soxx_raw.index).intersection(tqqq_raw.index).intersection(soxl_raw.index)

# -------------------------------------------------------------
# 2. サイドバー：科学的検証パラメータ設定（対話型ラボ）
# -------------------------------------------------------------
st.sidebar.title("🔬 科学的検証ラボ")
st.sidebar.markdown("各パラメータを動かし、統計指標（PF・カルマー等）の感度を検証します。")

st.sidebar.subheader("① 資産配分比率（基本枠）")
tqqq_weight = st.sidebar.slider("TQQQ 配分比率 (%)", min_value=0, max_value=100, value=70, step=5) / 100.0
soxl_weight = 1.0 - tqqq_weight
st.sidebar.caption(f"配分: TQQQ **{int(tqqq_weight*100)}%** ｜ SOXL **{int(soxl_weight*100)}%**")

st.sidebar.subheader("② トレンド判定 (EMA)")
ema_span = st.sidebar.slider("EMA 計算日数 (共通)", min_value=100, max_value=250, value=200, step=10)

st.sidebar.subheader("③ ボラティリティ警戒閾値 (HV20)")
qqq_hv_thresh = st.sidebar.slider("QQQ 警戒閾値 (%)", min_value=20.0, max_value=35.0, value=28.0, step=1.0)
soxx_hv_thresh = st.sidebar.slider("SOXX 警戒閾値 (%)", min_value=30.0, max_value=50.0, value=40.0, step=1.0)

st.sidebar.subheader("④ レジーム2（警戒時）の投資比率")
regime2_alloc = st.sidebar.slider("波乱相場での株式比率 (%)", min_value=0, max_value=100, value=50, step=10) / 100.0

# -------------------------------------------------------------
# 3. バックテスト計算関数（ルックアヘッドバイアス完全排除）
# -------------------------------------------------------------
def calculate_backtest(t_w, s_w, span, q_hvt, s_hvt, r2_alloc):
    df_q = qqq_raw.loc[common_idx].copy()
    df_s = soxx_raw.loc[common_idx].copy()
    
    # 指標算出
    df_q["EMA"] = df_q["Close"].ewm(span=span, adjust=False).mean()
    df_q["HV20"] = np.log(df_q["Close"] / df_q["Close"].shift(1)).rolling(20).std() * np.sqrt(252) * 100
    
    df_s["EMA"] = df_s["Close"].ewm(span=span, adjust=False).mean()
    df_s["HV20"] = np.log(df_s["Close"] / df_s["Close"].shift(1)).rolling(20).std() * np.sqrt(252) * 100
    
    # レジーム判定 (当日の判定)
    q_cond_bull = (df_q["Close"] > df_q["EMA"]) & (df_q["HV20"] < q_hvt)
    q_cond_warn = (df_q["Close"] > df_q["EMA"]) & (df_q["HV20"] >= q_hvt)
    q_pos = np.where(q_cond_bull, 1.0, np.where(q_cond_warn, r2_alloc, 0.0))
    
    s_cond_bull = (df_s["Close"] > df_s["EMA"]) & (df_s["HV20"] < s_hvt)
    s_cond_warn = (df_s["Close"] > df_s["EMA"]) & (df_s["HV20"] >= s_hvt)
    s_pos = np.where(s_cond_bull, 1.0, np.where(s_cond_warn, r2_alloc, 0.0))
    
    # 翌日執行 (shift 1)
    q_exec = pd.Series(q_pos, index=common_idx).shift(1).fillna(0)
    s_exec = pd.Series(s_pos, index=common_idx).shift(1).fillna(0)
    
    tqqq_ret = tqqq_raw.loc[common_idx, "Close"].pct_change().fillna(0)
    soxl_ret = soxl_raw.loc[common_idx, "Close"].pct_change().fillna(0)
    
    strat_ret = (t_w * q_exec * tqqq_ret) + (s_w * s_exec * soxl_ret)
    bm_ret = (t_w * tqqq_ret) + (s_w * soxl_ret)
    
    equity = (1 + strat_ret).cumprod()
    bm_equity = (1 + bm_ret).cumprod()
    
    # 統計指標計算
    n_days = len(strat_ret)
    years = n_days / 252.0
    cagr = (equity.iloc[-1]) ** (1.0 / years) - 1.0
    vol = strat_ret.std() * np.sqrt(252)
    
    roll_max = equity.cummax()
    dd = (equity - roll_max) / roll_max
    mdd = dd.min()
    
    calmar = cagr / abs(mdd) if mdd != 0 else 0
    gains = strat_ret[strat_ret > 0].sum()
    losses = abs(strat_ret[strat_ret < 0].sum())
    pf = gains / losses if losses != 0 else 0
    t_stat = (strat_ret.mean() / (strat_ret.std() / np.sqrt(n_days))) if strat_ret.std() != 0 else 0
    
    return {
        "CAGR": cagr, "Vol": vol, "MDD": mdd, "PF": pf, "Calmar": calmar, "t_stat": t_stat,
        "Equity": equity, "BM_Equity": bm_equity, "DD": dd,
        "Current_Q_Pos": q_pos[-1], "Current_S_Pos": s_pos[-1],
        "Q_EMA": df_q["EMA"].iloc[-1], "Q_HV": df_q["HV20"].iloc[-1],
        "S_EMA": df_s["EMA"].iloc[-1], "S_HV": df_s["HV20"].iloc[-1]
    }

# 選択中パラメータの実行
res = calculate_backtest(tqqq_weight, soxl_weight, ema_span, qqq_hv_thresh, soxx_hv_thresh, regime2_alloc)

# -------------------------------------------------------------
# 4. メインダッシュボード画面
# -------------------------------------------------------------
st.title("🛡️ 米国レバレッジETF 科学的検証・統合運用ダッシュボード")
st.caption(f"データ期間: 直近5年（{common_idx[0].strftime('%Y/%m/%d')} 〜 {common_idx[-1].strftime('%Y/%m/%d')}）")

tab1, tab2, tab3 = st.tabs([
    "🧪 リアルタイム検証結果 & 統計比較", 
    "🏛️ 現在シグナル & 楽天証券発注計算",
    "📈 テクニカルチャート"
])

# =============================================================
# TAB 1: リアルタイム検証 & 統計比較
# =============================================================
with tab1:
    st.subheader("📊 現在の設定における統計的パフォーマンス")
    st.caption("左側のサイドバーでスライダーを動かすと、リアルタイムに全期間のバックテストが再計算されます。")
    
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("期待年利 (CAGR)", f"{res['CAGR']*100:.1f} %")
    m2.metric("年率リスク (Vol)", f"{res['Vol']*100:.1f} %")
    m3.metric("最大下落率 (MDD)", f"{res['MDD']*100:.1f} %")
    m4.metric("プロフィットファクター (PF)", f"{res['PF']:.2f}")
    m5.metric("カルマーレシオ", f"{res['Calmar']:.2f}")
    m6.metric("t値 (統計的有意性)", f"{res['t_stat']:.2f}")

    st.markdown("---")
    
    # 資産推移 & ドローダウン
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.7, 0.3])
    fig.add_trace(go.Scatter(x=common_idx, y=res["Equity"], name="戦略資産曲線", line=dict(color="#00ba38", width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=common_idx, y=res["BM_Equity"], name="バイ＆ホールド", line=dict(color="#888888", width=1.5, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=common_idx, y=res["DD"] * 100, name="ドローダウン (%)", fill="tozeroy", line=dict(color="#d62728", width=1)), row=2, col=1)
    fig.update_layout(height=480, margin=dict(t=20, b=20, l=10, r=10), hovermode="x unified")
    fig.update_yaxes(title_text="資産倍率", row=1, col=1)
    fig.update_yaxes(title_text="下落率 (%)", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # 科学的比較マトリクス（固定パラメータ下での配分比率別比較）
    st.markdown("#### 🔬 ポートフォリオ配分比率別の客観的比較マトリクス")
    st.caption(f"（※現在の設定：EMA={ema_span}日, QQQ_HV={qqq_hv_thresh}%, SOXX_HV={soxx_hv_thresh}%, 警戒時比率={int(regime2_alloc*100)}% にて一括試算）")
    
    alloc_patterns = [
        ("TQQQ 100% : SOXL 0%", 1.0, 0.0),
        ("TQQQ 85% : SOXL 15%", 0.85, 0.15),
        ("TQQQ 70% : SOXL 30%", 0.70, 0.30),
        ("TQQQ 50% : SOXL 50%", 0.50, 0.50),
    ]
    matrix_rows = []
    for label, tw, sw in alloc_patterns:
        m = calculate_backtest(tw, sw, ema_span, qqq_hv_thresh, soxx_hv_thresh, regime2_alloc)
        matrix_rows.append({
            "配分パターン": label,
            "CAGR (年利)": f"{m['CAGR']*100:.1f} %",
            "リスク (Vol)": f"{m['Vol']*100:.1f} %",
            "最大下落 (MDD)": f"{m['MDD']*100:.1f} %",
            "PF": f"{m['PF']:.2f}",
            "カルマーレシオ": f"{m['Calmar']:.2f}",
            "t値": f"{m['t_stat']:.2f}"
        })
    st.table(pd.DataFrame(matrix_rows))

# =============================================================
# TAB 2: 現在シグナル & 発注計算
# =============================================================
with tab2:
    st.subheader("🏛️ 現在のシグナルと今夜の推奨発注")
    q_latest_c = qqq_raw["Close"].iloc[-1]
    s_latest_c = soxx_raw["Close"].iloc[-1]
    
    col1, col2, col3 = st.columns([1.2, 1.2, 1.6])
    with col1:
        st.markdown(f"### TQQQ (枠: {int(tqqq_weight*100)}%)")
        st.metric("QQQ 終値", f"${q_latest_c:.2f}")
        st.write(f"EMA({ema_span}日): **${res['Q_EMA']:.2f}**")
        st.write(f"HV20: **{res['Q_HV']:.1f}%** (閾値: {qqq_hv_thresh}%)")
        st.write(f"推奨ポジション: **{int(res['Current_Q_Pos']*100)}%**")
        st.progress(res['Current_Q_Pos'])
        
    with col2:
        st.markdown(f"### SOXL (枠: {int(soxl_weight*100)}%)")
        st.metric("SOXX 終値", f"${s_latest_c:.2f}")
        st.write(f"EMA({ema_span}日): **${res['S_EMA']:.2f}**")
        st.write(f"HV20: **{res['S_HV']:.1f}%** (閾値: {soxx_hv_thresh}%)")
        st.write(f"推奨ポジション: **{int(res['Current_S_Pos']*100)}%**")
        st.progress(res['Current_S_Pos'])
        
    with col3:
        tot_eq = (tqqq_weight * res['Current_Q_Pos']) + (soxl_weight * res['Current_S_Pos'])
        c_ratio = 1.0 - tot_eq
        st.markdown("### ポートフォリオ全体")
        st.metric("株式エクスポージャー", f"{tot_eq*100:.1f} %", f"待機キャッシュ: {c_ratio*100:.1f}%")
        fig_pie = go.Figure(data=[go.Pie(
            labels=['TQQQ', 'SOXL', '現金/MMF'],
            values=[tqqq_weight * res['Current_Q_Pos'] * 100, soxl_weight * res['Current_S_Pos'] * 100, c_ratio * 100],
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
        t_val = funds * tqqq_weight * res['Current_Q_Pos']
        s_val = funds * soxl_weight * res['Current_S_Pos']
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
        st.markdown("**QQQ (NASDAQ100)**")
        fig_q = go.Figure()
        fig_q.add_trace(go.Scatter(x=common_idx, y=qqq_raw.loc[common_idx, "Close"], name="QQQ 終値", line=dict(color="#1f77b4")))
        fig_q.add_trace(go.Scatter(x=common_idx, y=qqq_raw.loc[common_idx, "Close"].ewm(span=ema_span, adjust=False).mean(), name=f"{ema_span}日EMA", line=dict(color="#ff7f0e", width=2)))
        fig_q.update_layout(height=400, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_q, use_container_width=True)
    with c_sub2:
        st.markdown("**SOXX (半導体)**")
        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(x=common_idx, y=soxx_raw.loc[common_idx, "Close"], name="SOXX 終値", line=dict(color="#9467bd")))
        fig_s.add_trace(go.Scatter(x=common_idx, y=soxx_raw.loc[common_idx, "Close"].ewm(span=ema_span, adjust=False).mean(), name=f"{ema_span}日EMA", line=dict(color="#ff7f0e", width=2)))
        fig_s.update_layout(height=400, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_s, use_container_width=True)
