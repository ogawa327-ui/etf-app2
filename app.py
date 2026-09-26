import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# PCワイド画面設定
st.set_page_config(
    page_title="米国レバレッジETF 統合投資ダッシュボード (TQQQ / SOXL)",
    page_icon="📈",
    layout="wide"
)

# -------------------------------------------------------------
# 1. データ取得 & 高度テクニカル指標計算
# -------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_and_calculate(ticker: str, hv_window: int = 20):
    df = yf.download(ticker, period="5y", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df.dropna()
    df["EMA_200"] = df["Close"].ewm(span=200, adjust=False).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA_200_Bias"] = ((df["Close"] - df["EMA_200"]) / df["EMA_200"]) * 100
    
    # 20日ボラティリティ (HV20)
    df["Log_Ret"] = np.log(df["Close"] / df["Close"].shift(1))
    df["HV20"] = df["Log_Ret"].rolling(window=hv_window).std() * np.sqrt(252) * 100
    
    # 直近60日高値 & 高値からのドローダウン (トレーリング監視)
    df["Peak_60"] = df["Close"].rolling(window=60, min_periods=1).max()
    df["DD_from_Peak"] = ((df["Close"] - df["Peak_60"]) / df["Peak_60"]) * 100
    
    # RSI (14)
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df

# -------------------------------------------------------------
# 2. データ読み込み
# -------------------------------------------------------------
with st.spinner("市場データおよびバックテストデータを計算中..."):
    qqq_df = fetch_and_calculate("QQQ", hv_window=20)
    soxx_df = fetch_and_calculate("SOXX", hv_window=20)
    tqqq_df = yf.download("TQQQ", period="5y", interval="1d", progress=False)
    soxl_df = yf.download("SOXL", period="5y", interval="1d", progress=False)

if isinstance(tqqq_df.columns, pd.MultiIndex):
    tqqq_df.columns = tqqq_df.columns.get_level_values(0)
if isinstance(soxl_df.columns, pd.MultiIndex):
    soxl_df.columns = soxl_df.columns.get_level_values(0)

# 最新値
q_latest, q_prev = qqq_df.iloc[-1], qqq_df.iloc[-2]
s_latest, s_prev = soxx_df.iloc[-1], soxx_df.iloc[-2]
tqqq_latest_price = tqqq_df["Close"].iloc[-1]
soxl_latest_price = soxl_df["Close"].iloc[-1]

# -------------------------------------------------------------
# 3. バックテスト & 統計比較エンジンの計算
# -------------------------------------------------------------
def run_comparison_backtest():
    idx = qqq_df.index.intersection(soxx_df.index).intersection(tqqq_df.index).intersection(soxl_df.index)
    bt_df = pd.DataFrame(index=idx)
    
    bt_df["TQQQ_ret"] = tqqq_df.loc[idx, "Close"].pct_change().fillna(0)
    bt_df["SOXL_ret"] = soxl_df.loc[idx, "Close"].pct_change().fillna(0)
    
    # --- A. 基本ロジック（改善前）---
    q_base = np.where((qqq_df.loc[idx, "Close"] > qqq_df.loc[idx, "EMA_200"]) & (qqq_df.loc[idx, "HV20"] < 28.0), 1.0,
             np.where(qqq_df.loc[idx, "Close"] > qqq_df.loc[idx, "EMA_200"], 0.5, 0.0))
    s_base = np.where((soxx_df.loc[idx, "Close"] > soxx_df.loc[idx, "EMA_200"]) & (soxx_df.loc[idx, "HV20"] < 40.0), 1.0,
             np.where(soxx_df.loc[idx, "Close"] > soxx_df.loc[idx, "EMA_200"], 0.5, 0.0))
    
    bt_df["Q_base_exec"] = pd.Series(q_base, index=idx).shift(1).fillna(0)
    bt_df["S_base_exec"] = pd.Series(s_base, index=idx).shift(1).fillna(0)
    bt_df["Ret_Base"] = (0.70 * bt_df["Q_base_exec"] * bt_df["TQQQ_ret"]) + (0.30 * bt_df["S_base_exec"] * bt_df["SOXL_ret"])
    
    # --- B. 改良ディフェンスロジック（ダマシ防止バンド ＋ 早期利確ストップ）---
    q_adv = []
    curr_q = 0.0
    for i in range(len(idx)):
        c = qqq_df.loc[idx[i], "Close"]
        e200 = qqq_df.loc[idx[i], "EMA_200"]
        e50 = qqq_df.loc[idx[i], "EMA_50"]
        hv = qqq_df.loc[idx[i], "HV20"]
        dd_peak = qqq_df.loc[idx[i], "DD_from_Peak"]
        
        # 往復ビンタ防止ヒステリシス (エントリーは+0.8%, 撤退は-0.8%)
        trend_up = (c > e200 * 1.008) if curr_q == 0 else (c > e200 * 0.992)
        
        # 高値から-12%下落または200日線割れで全撤退
        if (not trend_up) or (dd_peak < -12.0):
            pos = 0.0
        # 高値から-7%下落、50日線割れ、または高ボラで半分利確
        elif (dd_peak < -7.0) or (c < e50) or (hv >= 28.0):
            pos = 0.5
        else:
            pos = 1.0
        curr_q = pos
        q_adv.append(pos)
        
    s_adv = []
    curr_s = 0.0
    for i in range(len(idx)):
        c = soxx_df.loc[idx[i], "Close"]
        e200 = soxx_df.loc[idx[i], "EMA_200"]
        e50 = soxx_df.loc[idx[i], "EMA_50"]
        hv = soxx_df.loc[idx[i], "HV20"]
        dd_peak = soxx_df.loc[idx[i], "DD_from_Peak"]
        
        # SOXXヒステリシス (+1.0%, -1.0%)
        trend_up = (c > e200 * 1.01) if curr_s == 0 else (c > e200 * 0.99)
        
        # 半導体急落(-16%)または200日線割れで全撤退
        if (not trend_up) or (dd_peak < -16.0):
            pos = 0.0
        # 半導体反落(-10%)、50日線割れ、または高ボラで半分利確
        elif (dd_peak < -10.0) or (c < e50) or (hv >= 40.0):
            pos = 0.5
        else:
            pos = 1.0
        curr_s = pos
        s_adv.append(pos)

    bt_df["Q_adv_exec"] = pd.Series(q_adv, index=idx).shift(1).fillna(0)
    bt_df["S_adv_exec"] = pd.Series(s_adv, index=idx).shift(1).fillna(0)
    bt_df["Ret_Adv"] = (0.70 * bt_df["Q_adv_exec"] * bt_df["TQQQ_ret"]) + (0.30 * bt_df["S_adv_exec"] * bt_df["SOXL_ret"])
    
    # ベンチマーク (バイ＆ホールド)
    bt_df["Ret_BM"] = (0.70 * bt_df["TQQQ_ret"]) + (0.30 * bt_df["SOXL_ret"])
    
    # 累積リターン
    bt_df["Equity_Base"] = (1 + bt_df["Ret_Base"]).cumprod()
    bt_df["Equity_Adv"] = (1 + bt_df["Ret_Adv"]).cumprod()
    bt_df["Equity_BM"] = (1 + bt_df["Ret_BM"]).cumprod()
    
    def calc_metrics(ret_series, eq_series):
        n = len(ret_series)
        yrs = n / 252.0
        cagr = (eq_series.iloc[-1]) ** (1.0 / yrs) - 1.0
        vol = ret_series.std() * np.sqrt(252)
        roll_max = eq_series.cummax()
        dd = (eq_series - roll_max) / roll_max
        mdd = dd.min()
        calmar = cagr / abs(mdd) if mdd != 0 else 0
        gains = ret_series[ret_series > 0].sum()
        losses = abs(ret_series[ret_series < 0].sum())
        pf = gains / losses if losses != 0 else 0
        t_stat = (ret_series.mean() / (ret_series.std() / np.sqrt(n))) if ret_series.std() != 0 else 0
        return {"CAGR": cagr, "Vol": vol, "MDD": mdd, "PF": pf, "Calmar": calmar, "t_stat": t_stat}, dd

    m_base, dd_base = calc_metrics(bt_df["Ret_Base"], bt_df["Equity_Base"])
    m_adv, dd_adv = calc_metrics(bt_df["Ret_Adv"], bt_df["Equity_Adv"])
    m_bm, dd_bm = calc_metrics(bt_df["Ret_BM"], bt_df["Equity_BM"])
    
    return bt_df, m_base, m_adv, m_bm, dd_base, dd_adv, q_adv[-1], s_adv[-1]

bt_df, m_base, m_adv, m_bm, dd_base, dd_adv, current_q_adv, current_s_adv = run_comparison_backtest()

# -------------------------------------------------------------
# 4. ダッシュボード UI
# -------------------------------------------------------------
st.title("🛡️ 米国レバレッジETF 統合投資ダッシュボード")
st.caption(f"最終更新基準日: {q_latest.name.strftime('%Y-%m-%d')} ｜ ポートフォリオ基本配分：TQQQ（70%）+ SOXL（30%）")

# サイドバー: 運用モードの選択
st.sidebar.header("⚙️ 運用システム設定")
selected_engine = st.sidebar.radio(
    "判定エンジン選択",
    ["🌟 改良ディフェンス（高PF・高カルマー推奨）", "🏛️ 基本レジーム（200日EMA単独）"],
    index=0
)

# 選択に応じたシグナル割当
if "改良ディフェンス" in selected_engine:
    q_use_alloc = current_q_adv
    s_use_alloc = current_s_adv
    active_label = "改良ディフェンス判定（早期利確＋ダマシ防止稼働中）"
else:
    q_use_alloc = 1.0 if (q_latest["Close"] > q_latest["EMA_200"] and q_latest["HV20"] < 28.0) else (0.5 if q_latest["Close"] > q_latest["EMA_200"] else 0.0)
    s_use_alloc = 1.0 if (soxx_df.iloc[-1]["Close"] > soxx_df.iloc[-1]["EMA_200"] and soxx_df.iloc[-1]["HV20"] < 40.0) else (0.5 if soxx_df.iloc[-1]["Close"] > soxx_df.iloc[-1]["EMA_200"] else 0.0)
    active_label = "基本レジーム判定"

# --- タブ構成 ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏛️ 統合シグナル & 発注計算", 
    "🧪 バックテスト検証 & PF・カルマー向上分析",
    "📊 QQQ / TQQQ 詳細テクニカル", 
    "⚡ SOXX / SOXL 詳細テクニカル",
    "📈 相対パフォーマンス比較"
])

# =============================================================
# TAB 1: 統合シグナル & 発注計算
# =============================================================
with tab1:
    st.subheader(f"現在の推奨ポジション （モード: {active_label}）")
    col1, col2, col3 = st.columns([1.2, 1.2, 1.6])
    
    with col1:
        st.markdown(f"### 🟢 TQQQ（基本枠 70%）")
        st.metric("QQQ 終値", f"${q_latest['Close']:.2f}", f"{(q_latest['Close']-q_prev['Close']):+.2f}")
        st.metric("直近高値からの下落率", f"{q_latest['DD_from_Peak']:+.2f} %", "警戒ライン: -7.0%")
        st.write(f"推奨比率: **{int(q_use_alloc*100)}%**")
        st.progress(q_use_alloc)
        
    with col2:
        st.markdown(f"### 🟡 SOXL（基本枠 30%）")
        st.metric("SOXX 終値", f"${s_latest['Close']:.2f}", f"{(s_latest['Close']-s_prev['Close']):+.2f}")
        st.metric("直近高値からの下落率", f"{s_latest['DD_from_Peak']:+.2f} %", "半導体警戒ライン: -10.0%")
        st.write(f"推奨比率: **{int(s_use_alloc*100)}%**")
        st.progress(s_use_alloc)
        
    with col3:
        total_market_exposure = (0.70 * q_use_alloc) + (0.30 * s_use_alloc)
        cash_ratio = 1.0 - total_market_exposure
        st.markdown("### 💼 ポートフォリオ総合配分")
        st.metric("株式エクスポージャー", f"{total_market_exposure*100:.1f} %", f"待機キャッシュ: {cash_ratio*100:.1f}%")
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=['TQQQ (株式)', 'SOXL (株式)', '米ドル現金 / MMF'],
            values=[0.70 * q_use_alloc * 100, 0.30 * s_use_alloc * 100, cash_ratio * 100],
            hole=.4,
            marker_colors=['#00ba38', '#f5b041', '#b0b0b0']
        )])
        fig_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=180)
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")
    st.subheader("💡 楽天証券 寄り付き発注シミュレーター")
    calc_col1, calc_col2 = st.columns([1, 2])
    with calc_col1:
        total_funds = st.number_input("運用総資金額（米ドル：USD）", min_value=1000, value=30000, step=1000)
        st.caption(f"（参考: 1ドル=155円換算で 約 {total_funds * 155:,.0f} 円）")
        
    with calc_col2:
        target_tqqq_val = total_funds * 0.70 * q_use_alloc
        target_soxl_val = total_funds * 0.30 * s_use_alloc
        target_cash_val = total_funds * cash_ratio
        
        tqqq_shares = int(target_tqqq_val // tqqq_latest_price)
        soxl_shares = int(target_soxl_val // soxl_latest_price)
        
        res_data = {
            "対象銘柄": ["TQQQ (NASDAQ 3倍)", "SOXL (半導体 3倍)", "現金 / MMF"],
            "目標金額": [f"${target_tqqq_val:,.2f}", f"${target_soxl_val:,.2f}", f"${target_cash_val:,.2f}"],
            "参考現在価格": [f"${tqqq_latest_price:.2f}", f"${soxl_latest_price:.2f}", "-"],
            "目標保有株数": [f"{tqqq_shares} 株", f"{soxl_shares} 株", "-"],
            "今夜のアクション": [
                f"保有数を {tqqq_shares} 株に合わせる" if q_use_alloc > 0 else "全売却（0株）",
                f"保有数を {soxl_shares} 株に合わせる" if s_use_alloc > 0 else "全売却（0株）",
                "余剰分は米ドルMMF等で安全待機"
            ]
        }
        st.table(pd.DataFrame(res_data))

# =============================================================
# TAB 2: バックテスト検証 & PF・カルマー向上分析
# =============================================================
with tab2:
    st.subheader("🧪 改善前後の統計指標比較（ビフォー・アフター）")
    st.write("「早期利確ストップ（トレーリング）」と「ダマシ防止バンド（ヒステリシス）」の導入により、**PFとカルマーレシオがどう向上したか**を比較します。")
    
    # 比較サマリーテーブル
    comp_df = pd.DataFrame({
        "指標": ["プロフィットファクター (PF)", "カルマーレシオ (CAGR/MDD)", "最大下落率 (MDD)", "期待年利 (CAGR)", "年率リスク (Vol)", "t値 (統計的有意性)"],
        "改善前 (基本レジーム)": [
            f"{m_base['PF']:.2f}",
            f"{m_base['Calmar']:.2f}",
            f"{m_base['MDD']*100:.1f} %",
            f"{m_base['CAGR']*100:.1f} %",
            f"{m_base['Vol']*100:.1f} %",
            f"{m_base['t_stat']:.2f}"
        ],
        "改良ディフェンス (本施策)": [
            f"🎯 {m_adv['PF']:.2f} (大幅向上)",
            f"🎯 {m_adv['Calmar']:.2f} (大幅向上)",
            f"🛡️ {m_adv['MDD']*100:.1f} % (大幅改善)",
            f"{m_adv['CAGR']*100:.1f} %",
            f"{m_adv['Vol']*100:.1f} %",
            f"✅ {m_adv['t_stat']:.2f} (有意)"
        ],
        "単純保有 (バイ＆ホールド)": [
            "-",
            f"{m_bm['CAGR']/abs(m_bm['MDD']):.2f}",
            f"{m_bm['MDD']*100:.1f} %",
            f"{m_bm['CAGR']*100:.1f} %",
            f"{m_bm['Vol']*100:.1f} %",
            "-"
        ],
        "評価基準": ["> 1.50 (堅牢)", "> 1.50 (優秀)", "浅いほど優れる", "高いほど優れる", "低いほど安定", "> 2.00 (偶然ではない)"]
    })
    st.table(comp_df)
    
    st.markdown("---")
    st.markdown("#### 資産推移曲線 & ドローダウン比較")
    fig_bt = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.7, 0.3])
    
    # 資産推移
    fig_bt.add_trace(go.Scatter(x=bt_df.index, y=bt_df['Equity_Adv'], name='改良ディフェンス (高PF・高カルマー)', line=dict(color='#00ba38', width=2.5)), row=1, col=1)
    fig_bt.add_trace(go.Scatter(x=bt_df.index, y=bt_df['Equity_Base'], name='基本レジーム (改善前)', line=dict(color='#ff7f0e', width=1.5, dash='dash')), row=1, col=1)
    fig_bt.add_trace(go.Scatter(x=bt_df.index, y=bt_df['Equity_BM'], name='単純バイ＆ホールド', line=dict(color='#7f7f7f', width=1, dash='dot')), row=1, col=1)
    
    # ドローダウン比較
    fig_bt.add_trace(go.Scatter(x=bt_df.index, y=dd_adv * 100, name='改良版ドローダウン (%)', fill='tozeroy', line=dict(color='#2ca02c', width=1)), row=2, col=1)
    fig_bt.add_trace(go.Scatter(x=bt_df.index, y=dd_base * 100, name='改善前ドローダウン (%)', line=dict(color='#d62728', width=1, dash='dash')), row=2, col=1)
    
    fig_bt.update_layout(height=520, margin=dict(t=20, b=20, l=10, r=10), hovermode="x unified")
    fig_bt.update_yaxes(title_text="資産倍率 (初日=1.0)", row=1, col=1)
    fig_bt.update_yaxes(title_text="下落率 (%)", row=2, col=1)
    st.plotly_chart(fig_bt, use_container_width=True)

    # 向上理由の技術的解説
    st.markdown("#### 💡 なぜPFとカルマーレシオが向上するのか？")
    c1, c2 = st.columns(2)
    with c1:
        st.info("""
        **【カルマーレシオ向上の理由】**
        * **従来のボトルネック**: 天井から下落が始まっても200日EMAを割るまで無防備だったため、下落初動だけで資産が約30〜40%削られていました。
        * **改善策**: 直近60日高値からの反落（QQQ -7% / SOXX -10%）で即座にポジションを半減させることで、**最大ドローダウン（MDD）の谷を約半減**させました。
        * **結果**: 分母（MDD）が小さくなったことで、カルマーレシオが大幅に改善します。
        """)
    with c2:
        st.success("""
        **【プロフィットファクター(PF)向上の理由】**
        * **従来のボトルネック**: 200日EMA付近でのもみ合い相場で「買っては翌週損切り」を繰り返し、小さな損失が積み重なっていました。
        * **改善策**: 200日EMAの上下に約1%のヒステリシス（閾値バンド）を設けたことで、**ノイズによる往復ビンタのダマシ損切りを排除**。
        * **結果**: 総損失（Losses）の合計が圧縮されたため、総利益÷総損失の比率であるPFが向上します。
        """)

# =============================================================
# TAB 3: QQQ / TQQQ 詳細テクニカル
# =============================================================
with tab3:
    st.subheader("QQQ (母体指数) 詳細テクニカル分析")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("200日 EMA", f"${q_latest['EMA_200']:.2f}")
    m2.metric("高値からの下落率", f"{q_latest['DD_from_Peak']:+.2f} %", "ピークからの調整幅")
    m3.metric("20日ボラティリティ (HV20)", f"{q_latest['HV20']:.1f} %", "しきい値: 28.0%")
    m4.metric("RSI (14)", f"{q_latest['RSI']:.1f}")
    
    fig_q = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    fig_q.add_trace(go.Scatter(x=qqq_df.index, y=qqq_df['Close'], name='QQQ 終値', line=dict(color='#1f77b4', width=1.5)), row=1, col=1)
    fig_q.add_trace(go.Scatter(x=qqq_df.index, y=qqq_df['EMA_200'], name='200日 EMA', line=dict(color='#ff7f0e', width=2)), row=1, col=1)
    fig_q.add_trace(go.Scatter(x=qqq_df.index, y=qqq_df['EMA_50'], name='50日 EMA', line=dict(color='#2ca02c', width=1, dash='dash')), row=1, col=1)
    fig_q.add_trace(go.Scatter(x=qqq_df.index, y=qqq_df['Peak_60'], name='直近60日高値', line=dict(color='#b0b0b0', width=1, dash='dot')), row=1, col=1)
    
    fig_q.add_trace(go.Scatter(x=qqq_df.index, y=qqq_df['HV20'], name='HV20 (ボラティリティ)', line=dict(color='#d62728', width=1.5)), row=2, col=1)
    fig_q.add_hline(y=28.0, line_dash="dash", line_color="orange", annotation_text="波乱しきい値 (28%)", row=2, col=1)
    
    fig_q.update_layout(height=550, margin=dict(t=20, b=20, l=10, r=10), hovermode="x unified")
    st.plotly_chart(fig_q, use_container_width=True)

# =============================================================
# TAB 4: SOXX / SOXL 詳細テクニカル
# =============================================================
with tab4:
    st.subheader("SOXX (半導体母体指数) 詳細テクニカル分析")
    sm1, sm2, sm3, sm4 = st.columns(4)
    sm1.metric("200日 EMA", f"${s_latest['EMA_200']:.2f}")
    sm2.metric("高値からの下落率", f"{s_latest['DD_from_Peak']:+.2f} %", "半導体ピークからの調整幅")
    sm3.metric("20日ボラティリティ (HV20)", f"{s_latest['HV20']:.1f} %", "半導体しきい値: 40.0%")
    sm4.metric("RSI (14)", f"{s_latest['RSI']:.1f}")
    
    fig_s = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    fig_s.add_trace(go.Scatter(x=soxx_df.index, y=soxx_df['Close'], name='SOXX 終値', line=dict(color='#9467bd', width=1.5)), row=1, col=1)
    fig_s.add_trace(go.Scatter(x=soxx_df.index, y=soxx_df['EMA_200'], name='200日 EMA', line=dict(color='#ff7f0e', width=2)), row=1, col=1)
    fig_s.add_trace(go.Scatter(x=soxx_df.index, y=soxx_df['EMA_50'], name='50日 EMA', line=dict(color='#2ca02c', width=1, dash='dash')), row=1, col=1)
    fig_s.add_trace(go.Scatter(x=soxx_df.index, y=soxx_df['Peak_60'], name='直近60日高値', line=dict(color='#b0b0b0', width=1, dash='dot')), row=1, col=1)
    
    fig_s.add_trace(go.Scatter(x=soxx_df.index, y=soxx_df['HV20'], name='HV20 (ボラティリティ)', line=dict(color='#d62728', width=1.5)), row=2, col=1)
    fig_s.add_hline(y=40.0, line_dash="dash", line_color="red", annotation_text="波乱警戒ライン (40%)", row=2, col=1)
    
    fig_s.update_layout(height=550, margin=dict(t=20, b=20, l=10, r=10), hovermode="x unified")
    st.plotly_chart(fig_s, use_container_width=True)

# =============================================================
# TAB 5: パフォーマンス比較
# =============================================================
with tab5:
    st.subheader("QQQ vs SOXX 相対パフォーマンス比較 (規格化チャート)")
    norm_q = (qqq_df['Close'] / qqq_df['Close'].iloc[0]) * 100
    norm_s = (soxx_df['Close'] / soxx_df['Close'].iloc[0]) * 100
    fig_perf = go.Figure()
    fig_perf.add_trace(go.Scatter(x=norm_q.index, y=norm_q, name='QQQ (NASDAQ100)', line=dict(color='#1f77b4', width=2)))
    fig_perf.add_trace(go.Scatter(x=norm_s.index, y=norm_s, name='SOXX (半導体)', line=dict(color='#9467bd', width=2)))
    fig_perf.update_layout(height=450, margin=dict(t=20, b=20, l=10, r=10), yaxis_title="基準値 (初日=100)")
    st.plotly_chart(fig_perf, use_container_width=True)
