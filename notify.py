import os
import sys
import time
import requests
import yfinance as yf
import pandas as pd
import numpy as np

LINE_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN")
LINE_USER_ID = os.environ.get("LINE_USER_ID")

UNDERLYING = "QQQ"
LEVERAGED = "TQQQ"
EMA_PERIOD = 200
HV_THRESHOLD = 0.25
BUFFER_PCT = 0.01

def fetch_data_robust(ticker, retries=3):
    for attempt in range(retries):
        try:
            df = yf.Ticker(ticker).history(period="1y")
            if not df.empty and len(df) > 200:
                df.index = df.index.tz_localize(None)
                return df
        except Exception as e:
            time.sleep(3)
    return None

def calculate_regime(df_u):
    df = df_u.copy()
    df["EMA200"] = df["Close"].ewm(span=EMA_PERIOD, adjust=False).mean()
    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    df["HV20"] = log_ret.rolling(window=20).std() * np.sqrt(252)
    df = df.dropna()
    
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    def get_reg(row):
        c, ema, hv = row["Close"], row["EMA200"], row["HV20"]
        if c < ema * (1 - BUFFER_PCT):
            return "BEAR", "🔴 レジーム3: ベア退避 (0% 現金)", "#E74C3C"
        elif c > ema * (1 + BUFFER_PCT):
            if hv < HV_THRESHOLD:
                return "BULL", "🟢 レジーム1: 強気巡航 (100% 投資)", "#2ECC71"
            else:
                return "WARNING", "🟡 レジーム2: 波乱警戒 (50% 保有)", "#F1C40F"
        return "NEUTRAL", "⚪ 中立・維持", "#95A5A6"
    
    curr_key, curr_title, curr_color = get_reg(latest)
    prev_key, _, _ = get_reg(prev)
    
    return {
        "is_changed": curr_key != prev_key,
        "curr_key": curr_key,
        "prev_key": prev_key,
        "title": curr_title,
        "color": curr_color,
        "close": float(latest["Close"]),
        "ema": float(latest["EMA200"]),
        "hv": float(latest["HV20"]) * 100,
        "date": latest.name.strftime("%Y-%m-%d")
    }

def send_line_flex(info):
    if not LINE_ACCESS_TOKEN or not LINE_USER_ID:
        print("LINEキーが未設定です。")
        return

    flex_payload = {
        "to": LINE_USER_ID,
        "messages": [{
            "type": "flex",
            "altText": f"【判定通知】{info['title']}",
            "contents": {
                "type": "bubble",
                "header": {
                    "type": "box",
                    "layout": "vertical",
                    "backgroundColor": info["color"],
                    "contents": [{
                        "type": "text",
                        "text": info["title"],
                        "weight": "bold",
                        "size": "md",
                        "color": "#FFFFFF"
                    }]
                },
                "body": {
                    "type": "box",
                    "layout": "vertical",
                    "contents": [
                        {"type": "text", "text": f"状態: {info['prev_key']} ➔ {info['curr_key']}", "size": "xs", "color": "#888888"},
                        {"type": "separator", "margin": "md"},
                        {
                            "type": "box",
                            "layout": "vertical",
                            "margin": "md",
                            "spacing": "sm",
                            "contents": [
                                {
                                    "type": "box",
                                    "layout": "baseline",
                                    "contents": [
                                        {"type": "text", "text": f"{UNDERLYING} 終値", "color": "#aaaaaa", "size": "sm", "flex": 2},
                                        {"type": "text", "text": f"${info['close']:.2f}", "weight": "bold", "size": "sm", "flex": 3, "align": "end"}
                                    ]
                                },
                                {
                                    "type": "box",
                                    "layout": "baseline",
                                    "contents": [
                                        {"type": "text", "text": "200日EMA", "color": "#aaaaaa", "size": "sm", "flex": 2},
                                        {"type": "text", "text": f"${info['ema']:.2f}", "weight": "bold", "size": "sm", "flex": 3, "align": "end"}
                                    ]
                                },
                                {
                                    "type": "box",
                                    "layout": "baseline",
                                    "contents": [
                                        {"type": "text", "text": "HV20 (ボラ)", "color": "#aaaaaa", "size": "sm", "flex": 2},
                                        {"type": "text", "text": f"{info['hv']:.1f}%", "weight": "bold", "size": "sm", "flex": 3, "align": "end"}
                                    ]
                                }
                            ]
                        }
                    ]
                },
                "footer": {
                    "type": "box",
                    "layout": "vertical",
                    "contents": [
                        {"type": "text", "text": f"基準日: {info['date']} | 寄り付き前注文推奨", "size": "xxs", "color": "#aaaaaa", "align": "center"}
                    ]
                }
            }
        }]
    }
    
    url = "https://api.line.me/v2/bot/message/push"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {LINE_ACCESS_TOKEN}"}
    requests.post(url, headers=headers, json=flex_payload)

def main():
    u_df = fetch_data_robust(UNDERLYING)
    if u_df is None:
        return
    info = calculate_regime(u_df)
    force = "--force" in sys.argv
    if not force and not info["is_changed"]:
        return
    send_line_flex(info)

if __name__ == "__main__":
    main()
