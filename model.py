# model.py — Keras LSTM 序列预测（预测未来7天）
# 保持原有CSV输入逻辑，无需修改 data.py 或 features.py

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from data import load_data
from features import engineer_features, get_features_and_target
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

SEQ_LEN = 30
HORIZON = 7


# ================= 生成滑动窗口 =================
def create_sequences(X, y):
    Xs, ys = [], []
    for i in range(len(X) - SEQ_LEN - HORIZON):
        Xs.append(X[i:i+SEQ_LEN])
        ys.append(y[i+SEQ_LEN:i+SEQ_LEN+HORIZON])
    return np.array(Xs), np.array(ys)


# ================= 训练 ==================
def train_model(horizon=HORIZON):
    df = engineer_features(load_data())
    X, _, feature_cols = get_features_and_target(df)

    y = df["close"].values.reshape(-1, 1)

    # 滑窗
    X_seq, y_seq = create_sequences(X.values, y)

    split = int(0.7 * len(X_seq))
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    model = Sequential([
        LSTM(64, return_sequences=False, input_shape=(SEQ_LEN, X.shape[1])),
        Dense(HORIZON)     # 直接输出7个预测值
    ])

    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=40, batch_size=16, verbose=1)

    # 预测
    pred = model.predict(X_test)

    # 性能评估
    mae_list, rmse_list = [], []
    for i in range(HORIZON):
        mae = mean_absolute_error(y_test[:, i], pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_test[:, i], pred[:, i]))
        mae_list.append(mae)
        rmse_list.append(rmse)
        print(f"Horizon {i+1} 日: MAE={mae:.2f}, RMSE={rmse:.2f}")

    print("\n平均 MAE=", np.mean(mae_list))
    print("平均 RMSE=", np.mean(rmse_list))

    # 保存
    model.save("lstm_keras_model.h5")

    return model, feature_cols


# =============== 未来预测 ================
def predict_future(model, feature_cols, steps=HORIZON):

    df = engineer_features(load_data())
    X = df[feature_cols].values

    seq = X[-SEQ_LEN:]  # 最后30天
    seq = np.expand_dims(seq, axis=0)

    pred = model.predict(seq)[0]

    last_date = df.index[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=steps, freq='B')

    future_df = pd.DataFrame({"predicted_close": pred}, index=future_dates)

    # plot
    plt.plot(future_df.index, pred, marker='o')
    plt.title("Future 7d forecast")
    plt.grid()
    plt.show()

    return future_df


if __name__ == "__main__":
    model, feature_cols = train_model()
    future = predict_future(model, feature_cols)
    print(future)
