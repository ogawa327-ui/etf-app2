import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="ETF Regime Engine", layout="wide")
st.title("🛡️ 米国レバレッジETF 3レジーム判定アプリ")

pair = st.sidebar.selectbox("取引対象", ["QQQ (原指数) / TQQQ (3倍)", "SPY (原指数) / UPRO (3倍)"])
u_sym = "QQQ" if "QQQ" in pair else "SPY"
l_sym = "TQQQ" if "QQQ" in pair else "UPRO"

user_capital_jpy = st.sidebar.number_input("運用資金 (円)", value=3000000, step=500000)
usdjpy_rate = st.sidebar.number_input("為替 (USD/JPY)", value=150.0, step=1.0)

@st.cache_data(ttl=3600)
def load_clean_data(underlying, leveraged):
    u = yf.Ticker(underlying).history(period="10y")
    l = yf.Ticker(leveraged).history(period="10y")
    u.index = u.index.tz_localize(None)
    l.index = l.index.tz_localize(None)
    return pd.DataFrame({"U_Close": u["Close"], "L_Close": l["Close"]}).dropna()

df = load_clean_data(u_sym, l_sym)
df["EMA200"] = df["U_Close"].ewm(span=200, adjust=False).mean()
log_ret = np.log(df["U_Close"] / df["U_Close"].shift(1))
df["HV20"] = log_ret.rolling(window=20).std() * np.sqrt(252)
df = df.dropna()

latest = df.iloc[-1]
close, ema, hv = latest["U_Close"], latest["EMA200"], latest["HV20"]
lev_price = latest["L_Close"]

if close < ema * 0.99:
    status_title = "🔴 レジーム3: ベア退避"
    weight = 0.0
    desc = "200EMAを下回っています。全額現金（SGOV等）へ退避します。"
elif close > ema * 1.01:
    if hv < 0.25:
        status_title = "🟢 レジーム1: 強気巡航"
        weight = 1.0
        desc = "200EMAの上でボラティリティも安定。100%保有します。"
    else:
        status_title = "🟡 レジーム2: 波乱警戒"
        weight = 0.5
        desc = "上昇中ですがボラティリティ警戒。リスク抑制のため50%保有にします。"
else:
    status_title = "⚪ 中立（現状維持）"
    weight = 0.0
    desc = "判断の分かれ目です。前日のポジションを維持します。"

target_usd = (user_capital_jpy / usdjpy_rate) * weight
shares = int(target_usd // lev_price)

st.subheader("🎯 本日のシグナル")
col1, col2, col3 = st.columns(3)
col1.metric("市場レジーム", status_title)
col2.metric("HV20 (ボラティリティ)", f"{hv*100:.1f}%", "25%以上で警戒")
col3.metric(f"推奨 {l_sym} 保有数", f"{shares:,} 株", f"${target_usd:,.0f}分")

st.info(desc)
st.markdown("#### 原指数と200日EMAの推移")
st.line_chart(df[["U_Close", "EMA200"]].tail(250))
