# model.py: 训练 LightGBM 模型
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from features import engineer_features, get_features_and_target


def train_model():
    """训练并返回模型、测试预测"""
    df = engineer_features(load_data())  # 从 features 导入数据
    X, y, feature_cols = get_features_and_target(df)

    # 时间分割
    split_date = '2023-01-01'
    X_train = X[X.index < split_date]
    X_test = X[X.index >= split_date]
    y_train = y[y.index < split_date]
    y_test = y[y.index >= split_date]

    print(f"训练集: {len(X_train)} 条，测试集: {len(X_test)} 条")

    # 模型
    model = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              early_stopping_rounds=100,
              verbose=50)

    # 评估
    pred = model.predict(X_test)
    pred_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, pred)
    print(f"\n✅ 模型训练完成！测试准确率: {acc:.2%}")
    print("\n分类报告:")
    print(classification_report(y_test, pred))

    # 混淆矩阵
    cm = confusion_matrix(y_test, pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 保存模型
    model.save_model('gold_model.txt')
    print("✅ 模型已保存到 gold_model.txt")

    return model, X_test.index, pred_proba, feature_cols


if __name__ == "__main__":
    train_model()