import pandas as pd
from alpha_vantage.timeseries import TimeSeries

# 初始化TimeSeries对象，并传入你的API密钥
ts = TimeSeries(key='ZCJDY6UJ8LHE4XV9')

# 获取日内交易数据，这里以'GOOGL'（谷歌的股票代码）为例
# 注意：Alpha Vantage提供的免费端点可能不包括所有股票的所有时间范围内的日内数据
# 你需要根据你的具体需求和API的限制来调整股票代码和时间参数
data, meta_data = ts.get_intraday('GOOGL', interval='5min', outputsize='full')

# 将数据转换为Pandas DataFrame
df = pd.DataFrame.from_dict(data, orient='index')

# 打印数据（可选）
print(df)

# 将DataFrame保存到CSV文件中
csv_file_path = 'googl_intraday_data.csv'
df.to_csv(csv_file_path)

print(f'Data saved to {csv_file_path}')
