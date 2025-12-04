# features.py: 特征工程
import pandas as pd
import pandas_ta as ta
from data import load_data  # 导入数据模块


def engineer_features(df):
    """添加技术指标、目标变量等特征"""
    data = df.copy()

    # 目标: 明天涨=1, 跌=0
    data['returns'] = data['close'].pct_change()
    data['target'] = (data['close'].shift(-1) > data['close']).astype(int)

    # 技术指标
    data.ta.rsi(close='close', length=14, append=True)
    data.ta.macd(close='close', append=True)
    data.ta.bbands(close='close', append=True)
    data.ta.ema(close='close', length=10, append=True)
    data.ta.ema(close='close', length=30, append=True)
    data.ta.sma(close='close', length=50, append=True)
    data.ta.sma(close='close', length=200, append=True)
    data.ta.stoch(append=True)
    data.ta.adx(append=True)
    data.ta.obv(append=True)

    # 额外特征
    data['volatility_20'] = data['returns'].rolling(20).std()
    data['close_to_high_20'] = data['close'] / data['high'].rolling(20).max()
    data['close_to_low_20'] = data['close'] / data['low'].rolling(20).min()
    data['dayofweek'] = data.index.dayofweek
    data['month'] = data.index.month

    data = data.dropna()
    print(f"✅ 特征工程完成！可用数据: {len(data)} 天")
    return data


def get_features_and_target(data):
    """返回 X (特征) 和 y (目标)"""
    feature_cols = [col for col in data.columns if col not in ['open', 'high', 'low', 'close', 'returns', 'target']]
    X = data[feature_cols]
    y = data['target']
    return X, y, feature_cols


if __name__ == "__main__":
    df = load_data()
    data = engineer_features(df)
    X, y, feature_cols = get_features_and_target(data)
    print(f"特征数量: {len(feature_cols)}")
    print(data.head(3))