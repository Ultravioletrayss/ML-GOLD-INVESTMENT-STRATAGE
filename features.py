# features.py: 特征工程模块
# 文件描述：本模块基于原始黄金数据，添加技术指标、目标变量、时间特征等，用于机器学习模型输入。
# 作者：基于 Grok (xAI) 生成
# 版本：1.0 (2025-12-04)
# 依赖：pip install pandas_ta（用于技术指标计算）

import pandas as pd  # 导入 pandas，用于数据处理（如复制 DataFrame、计算 pct_change 等）
import pandas_ta as ta  # 导入 pandas_ta，用于计算技术指标（如 RSI、MACD 等）
from data import load_data  # 从 data.py 导入 load_data 函数，用于获取原始数据


def engineer_features(df):
    """添加技术指标、目标变量等特征"""
    # 函数描述：输入原始 OHLC DataFrame，输出带特征的 DataFrame（包含 target）
    data = df.copy()  # 创建数据副本，避免修改原始数据

    # 目标变量：计算每日回报率，并创建二分类标签（明天 close > 今天 close 则 1=涨，0=跌）
    data['returns'] = data['close'].pct_change()  # 计算每日回报率：(close - prev_close) / prev_close
    data['target'] = (data['close'].shift(-1) > data['close']).astype(int)  # 移位 tomorrow close，与今天比较，转换为 int（1/0）

    # 技术指标：使用 pandas_ta 库自动计算并追加到 DataFrame（append=True 表示直接添加列）
    data.ta.rsi(close='close', length=14, append=True)  # RSI 相对强弱指数（14 期）
    data.ta.macd(close='close', append=True)  # MACD 移动平均收敛散度（默认参数：12,26,9）
    data.ta.bbands(close='close', append=True)  # Bollinger Bands 布林带（默认 20 期，2 倍标准差）
    data.ta.ema(close='close', length=10, append=True)  # EMA 指数移动平均（10 期）
    data.ta.ema(close='close', length=30, append=True)  # EMA 指数移动平均（30 期）
    data.ta.sma(close='close', length=50, append=True)  # SMA 简单移动平均（50 期）
    data.ta.sma(close='close', length=200, append=True)  # SMA 简单移动平均（200 期，长线趋势）
    data.ta.stoch(append=True)  # Stochastic 随机指标（默认 %K=14, %D=3）
    data.ta.adx(append=True)  # ADX 平均方向指数（默认 14 期，趋势强度）
    data.ta.obv(append=True)  # OBV 能量潮指标（基于成交量，但我们数据无 volume，可选）

    # 额外自定义特征：波动率与价格位置
    data['volatility_20'] = data['returns'].rolling(20).std()  # 20 日滚动回报率标准差（波动率）
    data['close_to_high_20'] = data['close'] / data['high'].rolling(20).max()  # 收盘价相对 20 日最高价的位置（0-1 归一化）
    data['close_to_low_20'] = data['close'] / data['low'].rolling(20).min()  # 收盘价相对 20 日最低价的位置（>1 表示强势）

    # 时间特征：提取日期的星期和月份（季节性效应）
    data['dayofweek'] = data.index.dayofweek  # 星期几（0=周一，6=周日）
    data['month'] = data.index.month  # 月份（1-12）

    data = data.dropna()  # 删除任何包含 NaN 的行（技术指标计算需历史数据，会产生 NaN）
    print(f"✅ 特征工程完成！可用数据: {len(data)} 天")  # 输出处理后数据量
    return data  # 返回带特征的 DataFrame


def get_features_and_target(data):
    """返回 X (特征矩阵) 和 y (目标向量)，以及特征列列表"""
    # 函数描述：从带特征的数据中提取 X（输入特征）和 y（输出标签），排除非特征列
    feature_cols = [col for col in data.columns if
                    col not in ['open', 'high', 'low', 'close', 'returns', 'target']]  # 筛选特征列（排除 OHLC、returns、target）
    X = data[feature_cols]  # X：特征 DataFrame（行=日期，列=特征）
    y = data['target']  # y：目标 Series（二分类标签）
    return X, y, feature_cols  # 返回 X, y 和特征列名列表（用于模型训练）


if __name__ == "__main__":  # 条件执行：仅在直接运行此文件时执行（python features.py）
    df = load_data()  # 从 data.py 加载原始数据
    data = engineer_features(df)  # 执行特征工程
    X, y, feature_cols = get_features_and_target(data)  # 提取 X 和 y
    print(f"特征数量: {len(feature_cols)}")  # 输出总特征数（通常 20+ 个）
    print(data.head(3))  # 打印前 3 行数据（包含所有新特征，用于检查）
    data.to_csv('features_data.csv', index=True)  # 保存完整特征数据到 CSV（日期索引包含在内）
    print("✅ 特征数据已保存到 features_data.csv")  # 输出保存成功提示