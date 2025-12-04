# main.py: 全功能入口（可选）
from model import train_model
from backtest import run_backtest
from predict import load_model, predict_tomorrow

if __name__ == "__main__":
    print("🚀 开始全量运行黄金预测系统...")
    train_model()  # 训练模型
    run_backtest()  # 回测
    model = load_model()
    if model:
        predict_tomorrow(model)  # 最新预测
    print("\n🎉 全运行完毕！")