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
# 1. データ取得 & 指標計算
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
    
    df["Log_Ret"] = np.log(df["Close"] / df["Close"].shift(1))
    df["HV20"] = df["Log_Ret"].rolling(window=hv_window).std() * np.sqrt(252) * 100
    
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df

def evaluate_regime(close_price, ema_200, hv20, hv_threshold):
    if close_price > ema_200:
        if hv20 < hv_threshold:
            return 1, "レジーム1：強気巡航", "🟢", 1.0, "安定上昇中。100%全力投資・継続保有"
        else:
            return 2, "レジーム2：波乱警戒", "🟡", 0.5, "高ボラティリティ。保有比率を50%に半減"
    else:
        return 3, "レジーム3：弱気防衛", "🔴", 0.0, "下落トレンド。全売却してキャッシュ（MMF）退避"

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

# 最新指標値
q_latest, q_prev = qqq_df.iloc[-1], qqq_df.iloc[-2]
s_latest, s_prev = soxx_df.iloc[-1], soxx_df.iloc[-2]
tqqq_latest_price = tqqq_df["Close"].iloc[-1]
soxl_latest_price = soxl_df["Close"].iloc[-1]

q_reg, q_title, q_icon, q_alloc, q_desc = evaluate_regime(q_latest["Close"], q_latest["EMA_200"], q_latest["HV20"], 28.0)
s_reg, s_title, s_icon, s_alloc, s_desc = evaluate_regime(s_latest["Close"], s_latest["EMA_200"], s_latest["HV20"], 40.0)

# -------------------------------------------------------------
# 3. バックテスト & 統計評価エンジンの計算
# -------------------------------------------------------------
def run_backtest():
    # 日付インデックスの共通化
    idx = qqq_df.index.intersection(soxx_df.index).intersection(tqqq_df.index).intersection(soxl_df.index)
    bt_df = pd.DataFrame(index=idx)
    
    # 日次リターン
    bt_df["TQQQ_ret"] = tqqq_df.loc[idx, "Close"].pct_change().fillna(0)
    bt_df["SOXL_ret"] = soxl_df.loc[idx, "Close"].pct_change().fillna(0)
    
    # レジーム比率の時系列算出
    # QQQ判定
    q_cond1 = qqq_df.loc[idx, "Close"] > qqq_df.loc[idx, "EMA_200"]
    q_cond2 = qqq_df.loc[idx, "HV20"] < 28.0
    bt_df["Q_pos"] = np.where(q_cond1 & q_cond2, 1.0, np.where(q_cond1, 0.5, 0.0))
    
    # SOXX判定
    s_cond1 = soxx_df.loc[idx, "Close"] > soxx_df.loc[idx, "EMA_200"]
    s_cond2 = soxx_df.loc[idx, "HV20"] < 40.0
    bt_df["S_pos"] = np.where(s_cond1 & s_cond2, 1.0, np.where(s_cond1, 0.5, 0.0))
    
    # 【重要】ルックアヘッドバイアス排除：当日のシグナルで「翌日寄り付き〜引け」を取引
    bt_df["Q_exec_pos"] = bt_df["Q_pos"].shift(1).fillna(0)
    bt_df["S_exec_pos"] = bt_df["S_pos"].shift(1).fillna(0)
    
    # 戦略ポートフォリオ日次リターン (TQQQ 70% + SOXL 30%)
    bt_df["Strategy_ret"] = (0.70 * bt_df["Q_exec_pos"] * bt_df["TQQQ_ret"]) + \
                            (0.30 * bt_df["S_exec_pos"] * bt_df["SOXL_ret"])
    
    # ベンチマーク（単純バイ＆ホールド 70:30）
    bt_df["BM_ret"] = (0.70 * bt_df["TQQQ_ret"]) + (0.30 * bt_df["SOXL_ret"])
    
    # 累積資産推移
    bt_df["Strategy_equity"] = (1 + bt_df["Strategy_ret"]).cumprod()
    bt_df["BM_equity"] = (1 + bt_df["BM_ret"]).cumprod()
    
    # 統計指標計算
    n_days = len(bt_df)
    years = n_days / 252.0
    
    cagr = (bt_df["Strategy_equity"].iloc[-1]) ** (1.0 / years) - 1.0
    bm_cagr = (bt_df["BM_equity"].iloc[-1]) ** (1.0 / years) - 1.0
    
    vol = bt_df["Strategy_ret"].std() * np.sqrt(252)
    bm_vol = bt_df["BM_ret"].std() * np.sqrt(252)
    
    # ドローダウン
    roll_max = bt_df["Strategy_equity"].cummax()
    dd = (bt_df["Strategy_equity"] - roll_max) / roll_max
    mdd = dd.min()
    
    bm_roll_max = bt_df["BM_equity"].cummax()
    bm_dd = (bt_df["BM_equity"] - bm_roll_max) / bm_roll_max
    bm_mdd = bm_dd.min()
    
    # カルマーレシオ
    calmar = cagr / abs(mdd) if mdd != 0 else 0
    
    # プロフィットファクター (PF)
    gains = bt_df["Strategy_ret"][bt_df["Strategy_ret"] > 0].sum()
    losses = abs(bt_df["Strategy_ret"][bt_df["Strategy_ret"] < 0].sum())
    pf = gains / losses if losses != 0 else np.nan
    
    # t値 (平均リターンが0と有意に異なるか)
    mean_ret = bt_df["Strategy_ret"].mean()
    std_ret = bt_df["Strategy_ret"].std()
    t_stat = (mean_ret / (std_ret / np.sqrt(n_days))) if std_ret != 0 else 0
    
    # 勝率
    active_days = bt_df[(bt_df["Q_exec_pos"] > 0) | (bt_df["S_exec_pos"] > 0)]
    win_rate = (active_days["Strategy_ret"] > 0).mean() * 100
    
    metrics = {
        "CAGR": cagr, "BM_CAGR": bm_cagr,
        "Vol": vol, "BM_Vol": bm_vol,
        "MDD": mdd, "BM_MDD": bm_mdd,
        "Calmar": calmar,
        "PF": pf,
        "t_stat": t_stat,
        "Win_Rate": win_rate,
        "Years": years,
        "Days": n_days
    }
    return bt_df, metrics, dd

bt_df, metrics, dd_series = run_backtest()

# -------------------------------------------------------------
# 4. ダッシュボード UI
# -------------------------------------------------------------
st.title("🛡️ 米国レバレッジETF 統合投資ダッシュボード")
st.caption(f"最終更新基準日: {q_latest.name.strftime('%Y-%m-%d')} ｜ ポートフォリオ基本配分：TQQQ（70%）+ SOXL（30%）")

# --- タブ構成 ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏛️ 統合シグナル & 発注計算", 
    "🧪 バックテスト検証 & 統計指標",
    "📊 QQQ / TQQQ 詳細テクニカル", 
    "⚡ SOXX / SOXL 詳細テクニカル",
    "📈 相対パフォーマンス比較"
])

# =============================================================
# TAB 1: 統合シグナル & 発注計算
# =============================================================
with tab1:
    st.subheader("現在の市場レジームと推奨ポジション")
    col1, col2, col3 = st.columns([1.2, 1.2, 1.6])
    
    with col1:
        st.markdown(f"### {q_icon} TQQQ（配分 70%）")
        st.metric("QQQ 終値", f"${q_latest['Close']:.2f}", f"{(q_latest['Close']-q_prev['Close']):+.2f}")
        st.write(f"**判定**: {q_title}")
        st.progress(q_alloc)
        st.write(f"推奨比率: **{int(q_alloc*100)}%** ({q_desc})")
        
    with col2:
        st.markdown(f"### {s_icon} SOXL（配分 30%）")
        st.metric("SOXX 終値", f"${s_latest['Close']:.2f}", f"{(s_latest['Close']-s_prev['Close']):+.2f}")
        st.write(f"**判定**: {s_title}")
        st.progress(s_alloc)
        st.write(f"推奨比率: **{int(s_alloc*100)}%** ({s_desc})")
        
    with col3:
        total_market_exposure = (0.70 * q_alloc) + (0.30 * s_alloc)
        cash_ratio = 1.0 - total_market_exposure
        st.markdown("### 💼 ポートフォリオ総合配分")
        st.metric("株式エクスポージャー", f"{total_market_exposure*100:.1f} %", f"待機キャッシュ: {cash_ratio*100:.1f}%")
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=['TQQQ (株式)', 'SOXL (株式)', '米ドル現金 / MMF'],
            values=[0.70 * q_alloc * 100, 0.30 * s_alloc * 100, cash_ratio * 100],
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
        target_tqqq_val = total_funds * 0.70 * q_alloc
        target_soxl_val = total_funds * 0.30 * s_alloc
        target_cash_val = total_funds * cash_ratio
        
        tqqq_shares = int(target_tqqq_val // tqqq_latest_price)
        soxl_shares = int(target_soxl_val // soxl_latest_price)
        
        res_data = {
            "対象銘柄": ["TQQQ (NASDAQ 3倍)", "SOXL (半導体 3倍)", "現金 / MMF"],
            "目標金額": [f"${target_tqqq_val:,.2f}", f"${target_soxl_val:,.2f}", f"${target_cash_val:,.2f}"],
            "参考現在価格": [f"${tqqq_latest_price:.2f}", f"${soxl_latest_price:.2f}", "-"],
            "目標保有株数": [f"{tqqq_shares} 株", f"{soxl_shares} 株", "-"],
            "今夜のアクション": [
                f"保有数を {tqqq_shares} 株に合わせる" if q_alloc > 0 else "全売却（0株）",
                f"保有数を {soxl_shares} 株に合わせる" if s_alloc > 0 else "全売却（0株）",
                "余剰分は米ドルMMF等で安全待機"
            ]
        }
        st.table(pd.DataFrame(res_data))

# =============================================================
# TAB 2: バックテスト検証 & 統計指標 (★新規追加★)
# =============================================================
with tab2:
    st.subheader("🧪 運用戦略の統計的パフォーマンス評価")
    
    # 統計指標カードの横並び表示
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("期待年利 (CAGR)", f"{metrics['CAGR']*100:.1f} %", f"単主持: {metrics['BM_CAGR']*100:.1f}%")
    m2.metric("年率リスク (Vol)", f"{metrics['Vol']*100:.1f} %", f"単主持: {metrics['BM_Vol']*100:.1f}%")
    m3.metric("最大下落 (MDD)", f"{metrics['MDD']*100:.1f} %", f"単主持: {metrics['BM_MDD']*100:.1f}%")
    m4.metric("プロフィットファクター (PF)", f"{metrics['PF']:.2f}", "基準: > 1.50")
    m5.metric("カルマーレシオ", f"{metrics['Calmar']:.2f}", "基準: > 1.50")
    m6.metric("t値 (統計的有意性)", f"{metrics['t_stat']:.2f}", "有意水準: > 2.00")

    st.markdown("---")
    
    # 資産曲線とドローダウンの可視化
    st.markdown("#### 資産推移曲線（本戦略 vs 単純バイ＆ホールド）")
    fig_bt = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.7, 0.3])
    
    # 資産推移
    fig_bt.add_trace(go.Scatter(x=bt_df.index, y=bt_df['Strategy_equity'], name='本戦略 (レジーム制御 70:30)', line=dict(color='#2ca02c', width=2.5)), row=1, col=1)
    fig_bt.add_trace(go.Scatter(x=bt_df.index, y=bt_df['BM_equity'], name='単純保有 (バイ＆ホールド 70:30)', line=dict(color='#7f7f7f', width=1.5, dash='dot')), row=1, col=1)
    
    # ドローダウン推移
    fig_bt.add_trace(go.Scatter(x=bt_df.index, y=dd_series * 100, name='戦略ドローダウン (%)', fill='tozeroy', line=dict(color='#d62728', width=1)), row=2, col=1)
    
    fig_bt.update_layout(height=520, margin=dict(t=20, b=20, l=10, r=10), hovermode="x unified")
    fig_bt.update_yaxes(title_text="資産倍率 (初日=1.0)", row=1, col=1)
    fig_bt.update_yaxes(title_text="下落率 (%)", row=2, col=1)
    st.plotly_chart(fig_bt, use_container_width=True)

    # バックテスト前提条件の明示（カード）
    st.markdown("#### 📋 バックテスト検証条件・前提仕様")
    c_cond1, c_cond2 = st.columns(2)
    with c_cond1:
        st.info("""
        **【システム設定・銘柄配分】**
        * **検証期間**: 直近約5年間（日足データ、データ取得日全期間）
        * **基本資産配分**: TQQQ 70%（コア） ＋ SOXL 30%（サテライト）
        * **シグナル判定指標**:
          * TQQQ枠: 母体指数 `QQQ` の 200日EMA ＆ 20日HV（しきい値: 28.0%）
          * SOXL枠: 母体指数 `SOXX` の 200日EMA ＆ 20日HV（しきい値: 40.0%）
        """)
    with c_cond2:
        st.success("""
        **【執行ルール・バイアス排除】**
        * **発注タイミング**: 米国市場クローズ後（日本時間朝）に判定し、**翌営業日寄り付き（成行）**にて売買執行（ルックアヘッドバイアスを完全排除）。
        * **ポジションサイズ**:
          * レジーム1（強気巡航）: 100% 投資
          * レジーム2（波乱警戒）: 50% 投資（半分キャッシュ化）
          * レジーム3（弱気防衛）: 0%（全額キャッシュ退避）
        * **キャッシュ運用**: 退避資金の金利収益・売買手数料・税金は考慮外（保守的評価）。
        """)

# =============================================================
# TAB 3: QQQ / TQQQ 詳細テクニカル
# =============================================================
with tab3:
    st.subheader("QQQ (母体指数) 詳細テクニカル分析")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("200日 EMA", f"${q_latest['EMA_200']:.2f}")
    m2.metric("200日 EMA 乖離率", f"{q_latest['EMA_200_Bias']:+.2f} %")
    m3.metric("20日ボラティリティ (HV20)", f"{q_latest['HV20']:.1f} %", "しきい値: 28.0%")
    m4.metric("RSI (14)", f"{q_latest['RSI']:.1f}")
    
    fig_q = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    fig_q.add_trace(go.Scatter(x=qqq_df.index, y=qqq_df['Close'], name='QQQ 終値', line=dict(color='#1f77b4', width=1.5)), row=1, col=1)
    fig_q.add_trace(go.Scatter(x=qqq_df.index, y=qqq_df['EMA_200'], name='200日 EMA', line=dict(color='#ff7f0e', width=2)), row=1, col=1)
    fig_q.add_trace(go.Scatter(x=qqq_df.index, y=qqq_df['EMA_50'], name='50日 EMA', line=dict(color='#2ca02c', width=1, dash='dash')), row=1, col=1)
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
    sm2.metric("200日 EMA 乖離率", f"{s_latest['EMA_200_Bias']:+.2f} %")
    sm3.metric("20日ボラティリティ (HV20)", f"{s_latest['HV20']:.1f} %", "半導体しきい値: 40.0%")
    sm4.metric("RSI (14)", f"{s_latest['RSI']:.1f}")
    
    fig_s = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    fig_s.add_trace(go.Scatter(x=soxx_df.index, y=soxx_df['Close'], name='SOXX 終値', line=dict(color='#9467bd', width=1.5)), row=1, col=1)
    fig_s.add_trace(go.Scatter(x=soxx_df.index, y=soxx_df['EMA_200'], name='200日 EMA', line=dict(color='#ff7f0e', width=2)), row=1, col=1)
    fig_s.add_trace(go.Scatter(x=soxx_df.index, y=soxx_df['EMA_50'], name='50日 EMA', line=dict(color='#2ca02c', width=1, dash='dash')), row=1, col=1)
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
