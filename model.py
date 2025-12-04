# model.py: 模型训练模块（扩展版：多步价格预测）
# 文件描述：本模块训练 LightGBM 回归模型，用于预测黄金未来 30 天每日收盘价（多输出回归）。
#           原二分类（涨跌）升级为回归（价格），支持单步/多步预测。直接多输出生成 30 天序列（无递归）。
# 作者：基于 Grok (xAI) + ChatGPT 联合整理
# 版本：2.5 (2025-12-04) - 使用 MultiOutputRegressor 包裹 LGBMRegressor 支持多输出
# 依赖：pip install lightgbm scikit-learn matplotlib seaborn pandas numpy joblib

from lightgbm import LGBMRegressor  # 导入 LightGBM 回归器（用于连续值预测）
from sklearn.multioutput import MultiOutputRegressor  # 导入 MultiOutputRegressor（sklearn 官方多输出包装器）
from sklearn.metrics import mean_absolute_error, mean_squared_error  # 导入回归评估指标（MAE、MSE）
import pandas as pd  # 导入 pandas，用于数据处理（如索引对齐、预测生成）
import numpy as np  # 导入 numpy，用于数值计算（如 np.sqrt 用于 RMSE）
import matplotlib
import matplotlib.pyplot as plt  # 导入 matplotlib，用于绘图（预测 vs 实际曲线）
import seaborn as sns  # 导入 seaborn（暂未使用，可后续扩展热图）
from features import engineer_features, get_features_and_target  # 从 features.py 导入特征工程函数
from data import load_data  # 从 data.py 导入数据加载函数
import warnings  # 导入 warnings，用于忽略非关键警告
import joblib  # 用于保存模型

# === 解决中文乱码和负号显示问题（Windows + 微软雅黑） ===
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 使用微软雅黑显示中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示为方块的问题

warnings.filterwarnings("ignore")  # 忽略所有警告（包括 OBV 无 volume 的警告）


def train_model(horizon=30):
    """
    训练 LightGBM 回归模型，支持多步价格预测（多输出回归）。

    参数:
        horizon (int): 预测天数（默认 30）。

    返回:
        model: 训练好的 MultiOutputRegressor(LGBMRegressor)
        test_index: 测试集索引（DatetimeIndex）
        pred_df: 测试集预测结果 DataFrame，列为 close_1 ... close_horizon
        feature_cols: 使用的特征列名列表
        mae_scores: 每个 horizon 的 MAE 列表
        rmse_scores: 每个 horizon 的 RMSE 列表
    """
    # 1. 加载并做特征工程
    df = engineer_features(load_data())  # 从 features.py 加载并工程化数据（包含 'close' 和特征）
    X, y_binary, feature_cols = get_features_and_target(df)  # 获取特征 X（旧二分类 y_binary 不再使用）

    # 2. 构造多步回归目标：未来 1~horizon 天的收盘价
    for t in range(1, horizon + 1):
        # 例如：close_1 = 明天、close_2 = 后天...
        df[f'close_{t}'] = df['close'].shift(-t)

    # 目标列名
    target_cols = [f'close_{t}' for t in range(1, horizon + 1)]
    y = df[target_cols]  # DataFrame: (n_samples, horizon)

    # 3. 去掉末尾未来数据不足的行（NaN）
    valid_mask = y.notna().all(axis=1)  # 仅保留所有目标列都有值的行
    df = df[valid_mask]
    X = X[valid_mask]
    y = y[valid_mask]

    if len(X) < 100:
        raise ValueError(f"❌ 数据不足！可用样本: {len(X)}，需至少 100 条。检查数据范围。")

    # 4. 时间序列切分：前 70% 作为训练集，后 30% 作为测试集
    split_idx = int(0.7 * len(X))
    split_date = X.index[split_idx]  # 对应的实际日期（便于打印查看）
    mask_train = X.index < split_date
    mask_test = X.index >= split_date

    X_train = X[mask_train]
    y_train = y[mask_train]
    X_test = X[mask_test]
    y_test = y[mask_test]

    print(f"训练集: {len(X_train)} 条，测试集: {len(X_test)} 条")
    print(f"动态 split_date: {split_date.date()}")
    print(f"预测 horizon: {horizon} 天（多输出回归）")

    # 5. 定义基础 LightGBM 回归模型
    base_model = LGBMRegressor(
        n_estimators=1000,       # 树的数量（提升迭代次数）
        learning_rate=0.01,      # 学习率（小一点，拟合更平滑）
        max_depth=6,             # 最大树深，控制复杂度
        subsample=0.8,           # 行采样比例（防止过拟合）
        colsample_bytree=0.8,    # 列采样比例（防止过拟合）
        random_state=42,         # 随机种子
        objective='regression',  # 回归任务
        verbosity=-1             # 不打印 LightGBM 自己的日志
    )

    # 6. 用 MultiOutputRegressor 包装，实现多输出回归（30 天一起预测）
    model = MultiOutputRegressor(base_model, n_jobs=1)  # n_jobs=1 防止某些环境下并行问题

    # 7. 训练模型
    # 注意：MultiOutputRegressor.fit 不支持 early_stopping_rounds, eval_set 等参数
    model.fit(X_train, y_train)

    # 8. 在测试集上做预测
    pred = model.predict(X_test)  # ndarray: (n_test, horizon)
    pred_df = pd.DataFrame(pred, index=X_test.index, columns=target_cols)

    # 9. 评估指标：逐 horizon 计算 MAE & RMSE
    mae_scores = []
    rmse_scores = []
    for t in range(horizon):
        mae = mean_absolute_error(y_test.iloc[:, t], pred_df.iloc[:, t])
        rmse = np.sqrt(mean_squared_error(y_test.iloc[:, t], pred_df.iloc[:, t]))
        mae_scores.append(mae)
        rmse_scores.append(rmse)
        print(f"Horizon {t + 1:2d} 天 - MAE: ${mae:.2f}, RMSE: ${rmse:.2f}")

    avg_mae = np.mean(mae_scores)
    avg_rmse = np.mean(rmse_scores)
    print(f"\n✅ 模型训练完成！平均 MAE: ${avg_mae:.2f}, 平均 RMSE: ${avg_rmse:.2f}")

    # 10. 可视化：前 10 个 horizon 的真实 vs 预测
    plt.figure(figsize=(18, 10))
    plt.suptitle("前 10 个预测步长（Horizon）真实 vs 预测价格对比", fontsize=16)

    max_plots = min(10, horizon)
    sample_len = min(120, len(y_test))  # 最多显示 120 天，避免曲线太挤

    for t in range(max_plots):
        plt.subplot(2, 5, t + 1)
        plt.plot(
            y_test.index[:sample_len],
            y_test.iloc[:sample_len, t],
            label="实际价格 Actual",
            alpha=0.8
        )
        plt.plot(
            pred_df.index[:sample_len],
            pred_df.iloc[:sample_len, t],
            label="预测价格 Predicted",
            alpha=0.8
        )
        plt.title(f"Horizon {t + 1} 天", fontsize=12)
        plt.xlabel("日期 Date", fontsize=9)
        plt.ylabel("价格 Price (USD)", fontsize=9)
        plt.grid(True, linestyle='--', alpha=0.3)
        if t == 0:
            plt.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('multi_step_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ 测试集多步预测图已保存为 multi_step_predictions.png")

    # 11. 保存整个多输出模型
    joblib.dump(model, 'gold_regressor_model.pkl')
    print("✅ 模型已保存到 gold_regressor_model.pkl")

    # 12. 保存测试集预测结果
    pred_df.to_csv('test_predictions.csv')
    print("✅ 测试预测已保存到 test_predictions.csv")

    return model, X_test.index, pred_df, feature_cols, mae_scores, rmse_scores


def predict_future(model, feature_cols, horizon=30, steps=30):
    """
    使用训练好的模型预测未来价格序列（多输出，不递归）。

    参数:
        model: 训练好的 MultiOutputRegressor 模型
        feature_cols: 训练时使用的特征列名列表
        horizon (int): 模型实际支持的预测步数（一般为 30）
        steps (int): 想要输出的预测天数（不应大于 horizon）

    返回:
        future_df: DataFrame，索引为未来日期，列为 predicted_close
    """
    # 1. 用最新数据构造当前特征
    data = engineer_features(load_data())
    latest_features = data[feature_cols].iloc[-1:]  # DataFrame (1, n_features)

    # 2. 模型直接输出多步预测（shape: (1, horizon)）
    future_pred_full = model.predict(latest_features)[0]

    # 保证 steps 不超过模型 horizon
    steps = min(steps, len(future_pred_full))
    future_pred = future_pred_full[:steps]

    # 3. 当前真实价格（最后一天收盘价）
    current_close = data['close'].iloc[-1]

    # 4. 构造未来日期索引（工作日频率）
    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                 periods=steps, freq='B')

    # 5. 生成预测结果 DataFrame
    future_df = pd.DataFrame(
        {'predicted_close': future_pred},
        index=future_dates
    )

    # 6. 可视化未来价格预测曲线
    plt.figure(figsize=(12, 6))
    plt.plot(future_df.index, future_df['predicted_close'],
             marker='o', label='预测价格 Predicted', linewidth=2)
    plt.axhline(
        y=current_close,
        color='r',
        linestyle='--',
        label=f'当前价格 Current: ${current_close:.2f}'
    )
    plt.title(f'黄金未来 {steps} 个交易日价格预测', fontsize=14)
    plt.xlabel('日期 Date')
    plt.ylabel('收盘价 Close (USD)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('future_30d_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ 未来价格预测图已保存为 future_30d_predictions.png")

    # 7. 保存未来预测结果
    future_df.to_csv('future_30d_predictions.csv')
    print("✅ 未来 30 天预测已保存到 future_30d_predictions.csv")

    # 控制台打印前 10 天变化情况
    print(f"\n📈 当前黄金价格: ${current_close:.2f}")
    print("未来若干天预测趋势（前 10 行）：")
    for i, (date, price) in enumerate(future_df.head(10).itertuples(), 1):
        change_pct = (price - current_close) / current_close * 100
        print(f"{i:2d} 天后 ({date.date()}): ${price:.2f} ({change_pct:+.2f}%)")

    return future_df


if __name__ == "__main__":
    # 训练模型（默认 horizon=30）
    model, test_index, pred_df, feature_cols, mae_scores, rmse_scores = train_model(horizon=30)

    # 使用训练好的模型预测未来 30 个交易日的价格
    future_predictions = predict_future(model, feature_cols, horizon=30, steps=30)
