# data.py: 获取现货黄金数据（简化版：仅 Polygon.io）
# 文件描述：本模块负责从 Polygon.io API 获取 XAU/USD 现货黄金的 OHLC 日线历史数据。
# 作者：基于 Grok (xAI) 生成
# 版本：2.0 (2025-12-04)
# 依赖：pip install polygon-api-client pandas

import pandas as pd  # 导入 pandas，用于数据处理（如 DataFrame 创建、索引设置、排序等）
import requests  # 用于发送 HTTP 请求（备用，但本版不需）
from datetime import datetime  # 用于处理日期时间（如格式化当前日期为字符串）
from polygon import RESTClient  # Polygon.io 客户端（pip install polygon-api-client，用于 API 调用）

# 配置：替换你的 Polygon Key（免费注册 https://polygon.io/）
# 变量说明：POLYGON_KEY 是 Polygon.io 的 API 密钥，用于认证请求
POLYGON_KEY = "SfCtIm5CJiwpjvak_kJDV1bZr8QmOx0m"  # 替换为你的真实 Key


def get_gold_data_polygon():
    """通过 Polygon.io API 获取 XAU/USD 现货黄金数据（唯一来源）"""
    # 函数描述：从 Polygon.io 获取黄金现货日线数据，返回 DataFrame 或 None（失败时）
    if POLYGON_KEY == "YOUR_POLYGON_KEY_HERE":  # 检查 Key 是否已设置（字符串比较）
        print("❌ Polygon Key 未设置！请去 https://polygon.io/ 免费注册并替换 Key")  # 输出错误提示
        return None  # 返回 None 表示失败

    try:  # 开始异常处理块，捕获 API 调用中的潜在错误
        client = RESTClient(api_key=POLYGON_KEY)  # 初始化 Polygon REST 客户端，传入 API Key
        end_date = datetime.now().strftime('%Y-%m-%d')  # 获取当前日期，并格式化为 YYYY-MM-DD 字符串

        # 获取日线聚合数据（从 2015 开始，足够用）
        # 调用 API：get_aggs 获取聚合数据（ticker、时间跨度、日期范围、限制条数）
        aggs = client.get_aggs(
            ticker="C:XAUUSD",  # Polygon 的黄金现货 ticker（C: 表示商品）
            multiplier=1,  # 时间乘数：1（日线）
            timespan="day",  # 时间跨度：day（日频）
            from_="2015-01-01",  # 开始日期：2015-01-01（历史起点）
            to=end_date,  # 结束日期：当前日期
            limit=50000  # 拉全历史：最大 50000 条（足够覆盖所有数据）
        )

        if len(aggs) == 0:  # 检查返回的聚合对象列表是否为空
            print("❌ Polygon 无数据！检查 Key 或网络")  # 输出无数据错误
            return None  # 返回 None 表示失败

        # 转换为 DataFrame
        # 初始化空列表，用于存储每个聚合条目的字典
        df_list = []
        for agg in aggs:  # 遍历聚合对象列表
            df_list.append({  # 添加字典到列表（每个字典代表一行数据）
                'timestamp': pd.to_datetime(agg.timestamp, unit='ms'),  # 时间戳：毫秒转 datetime
                'open': agg.open,  # 开盘价：从聚合对象提取
                'high': agg.high,  # 最高价：从聚合对象提取
                'low': agg.low,  # 最低价：从聚合对象提取
                'close': agg.close,  # 收盘价：从聚合对象提取
                'volume': agg.volume  # 可选，成交量：从聚合对象提取
            })

        df = pd.DataFrame(df_list)  # 从列表创建 DataFrame
        df.set_index('timestamp', inplace=True)  # 设置时间戳为索引（inplace=True 修改原 DataFrame）
        df = df.sort_index()  # 按索引（日期）升序排序
        df = df[['open', 'high', 'low', 'close']]  # 只保留 OHLC 列（丢弃 volume）
        print(f"✅ Polygon 数据：{len(df)} 天，从 {df.index.min().date()} 到 {df.index.max().date()}")  # 输出数据统计
        return df  # 返回处理后的 DataFrame

    except Exception as e:  # 捕获所有异常（API 错误、网络问题等）
        print(f"❌ Polygon API 错误: {e}！检查 Key、网络或 Polygon 状态")  # 输出异常细节
        return None  # 返回 None 表示失败


def load_data():
    """加载黄金数据：仅使用 Polygon.io"""
    # 函数描述：主加载函数，调用 get_gold_data_polygon 并处理输出
    df = get_gold_data_polygon()  # 调用数据获取函数

    if df is None:  # 检查 DataFrame 是否为 None
        print("❌ 数据获取失败！无法继续。请修复 Polygon Key 或网络")  # 输出失败提示
        return pd.DataFrame()  # 返回空 DataFrame，避免下游崩溃

    # 统一输出
    # 输出成功信息：时间范围和数据量
    print(f"✅ 数据获取成功！时间范围: {df.index.min().date()} ~ {df.index.max().date()}")
    print(f"数据量: {len(df)} 天")
    return df.dropna()  # 删除任何 NA 行，并返回数据


if __name__ == "__main__":  # 条件执行：仅在直接运行此文件时执行（python data.py）
    df = load_data()  # 执行加载数据函数
    if not df.empty:  # 检查 DataFrame 是否非空
        print(df.tail(3))  # 打印最近 3 行数据（用于测试）
        df.to_csv('gold_data.csv', index=True)  # 保存 DataFrame 到 CSV 文件（日期索引包含在内）
        print("✅ 数据已保存到 gold_data.csv")  # 输出保存成功提示
    else:
        print("测试失败：无数据可用")  # 输出测试失败提示