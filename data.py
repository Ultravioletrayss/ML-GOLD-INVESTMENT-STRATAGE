# data.py: 获取现货黄金数据
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime

API_KEY = "YOUR_KEY_HERE"  # 替换为你的 Alpha Vantage Key


def get_gold_data_av():
    if API_KEY == "YOUR_KEY_HERE":
        print("⚠️ 未设置 API Key，使用 yfinance 备用方案")
        return None

    url = f"https://www.alphavantage.co/query"
    params = {
        "function": "FX_DAILY",
        "from_symbol": "XAU",
        "to_symbol": "USD",
        "apikey": API_KEY,
        "outputsize": "full",
        "datatype": "json"
    }
    r = requests.get(url, params=params)
    data = r.json()

    if "Time Series FX (Daily)" not in data:
        print(f"⚠️ API 错误: {data.get('Note', 'Key 无效或限额')}")
        return None

    df = pd.DataFrame.from_dict(data["Time Series FX (Daily)"], orient="index")
    df = df.astype(float)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df.columns = ["open", "high", "low", "close"]
    return df


def get_gold_data_yf():
    df = yf.download("GLD", period="max", interval="1d")
    df = df[['Open', 'High', 'Low', 'Close']].rename(columns={
        'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'
    })
    return df


def load_data():
    """加载并返回数据 DataFrame"""
    df = get_gold_data_av()
    if df is None:
        df = get_gold_data_yf()
    print(f"✅ 数据获取成功！时间范围: {df.index.min().date()} ~ {df.index.max().date()}")
    print(f"数据量: {len(df)} 天")
    return df.dropna()


if __name__ == "__main__":
    df = load_data()
    print(df.tail(3))