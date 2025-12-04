# backtest.py: 策略回测
import pandas as pd
import matplotlib.pyplot as plt
from model import train_model
from features import engineer_features, get_features_and_target


def run_backtest():
    """运行回测并绘图"""
    model, test_index, pred_proba, feature_cols = train_model()  # 训练模型（如果已保存，可跳过）

    data = engineer_features(load_data())
    X, y, _ = get_features_and_target(data)
    test_returns = data.loc[X.loc[test_index].index, 'returns'].shift(-1).fillna(0)

    threshold = 0.55
    signals = (pred_proba > threshold).astype(int)
    strategy_returns = signals * test_returns
    cum_strategy = (1 + strategy_returns).cumprod()
    cum_benchmark = (1 + test_returns).cumprod()

    plt.figure(figsize=(12, 6))
    plt.plot(cum_strategy.index, cum_strategy, label='策略（做多信号）', linewidth=2)
    plt.plot(cum_benchmark.index, cum_benchmark, label='持有黄金', linewidth=2)
    plt.title('黄金涨跌预测策略回测（2023至今）')
    plt.ylabel('累计收益')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('backtest.png', dpi=300, bbox_inches='tight')
    plt.show()

    days = len(cum_strategy)
    strategy_annual = (cum_strategy.iloc[-1] ** (252 / days) - 1) * 100
    benchmark_annual = (cum_benchmark.iloc[-1] ** (252 / days) - 1) * 100
    print(f"\n✅ 回测结果:")
    print(f"策略年化收益: {strategy_annual:.2f}%")
    print(f"持有年化收益: {benchmark_annual:.2f}%")


if __name__ == "__main__":
    run_backtest()