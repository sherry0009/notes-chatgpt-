import tushare as ts
import pandas as pd

# 设置tushare的token
token = ''
ts.set_token(token)
pro = ts.pro_api()


def get_stock_data(ts_code, start_date, end_date):
    # 使用tushare的pro.daily接口获取日K线数据
    df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
    return df


def save_to_csv(stock_data, filename):
    # 确保DataFrame不为空再保存
    if not stock_data.empty:
        stock_data.to_csv(filename, index=False)
        print(f'Stock data saved to {filename}')
    else:
        print('No data to save.')

    # 比亚迪的股票代码（请根据实际情况确认）


# 假设是002594.SZ，但请确保这是正确的股票代码
ts_code = '002594.SZ'
start_date = '20240301'  # tushare的日期格式通常是YYYYMMDD
end_date = '20240315'

stock_data = get_stock_data(ts_code, start_date, end_date)

save_to_csv(stock_data, 'byd_stock_data.csv')
