# predict.py: 最新预测（最常用！）
from lightgbm import Booster
import pandas as pd
from datetime import datetime
from data import load_data
from features import engineer_features, get_features_and_target


def load_model():
    """加载已训练模型"""
    try:
        model = Booster(model_file='gold_model.txt')
        return model
    except:
        print("⚠️ 未找到模型文件 gold_model.txt，请先运行 model.py 训练模型")
        return None


def predict_tomorrow(model):
    """预测明天涨跌"""
    data = engineer_features(load_data())
    X, y, feature_cols = get_features_and_target(data)

    latest_data = data[feature_cols].iloc[-1:]
    prob = model.predict(latest_data)[0]  # Booster 的 predict 返回概率数组
    current_price = data['close'].iloc[-1]
    today = datetime.now().strftime('%Y-%m-%d')

    print("\n" + "=" * 50)
    print(f"【最新预测】 {today}")
    print(f"当前黄金价格: ${current_price:.2f}")
    print(f"模型预测明天上涨概率: {prob * 100:.2f}%")

    if prob > 0.58:
        signal = "🚀 强烈看涨！"
    elif prob > 0.55:
        signal = "📈 偏向看涨"
    elif prob < 0.45:
        signal = "📉 偏向看跌"
    else:
        signal = "⚡ 震荡，观望为主"

    print(signal)
    print("=" * 50)


if __name__ == "__main__":
    model = load_model()
    if model:
        predict_tomorrow(model)