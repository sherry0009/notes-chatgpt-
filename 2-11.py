import requests  # 导入requests库，用于发送HTTP请求
from bs4 import BeautifulSoup  # 导入BeautifulSoup库，用于解析HTML文档
import csv  # 导入csv库，用于将数据写入CSV文件


def get_weather_data(city_url):
    """
    从给定的城市天气预报URL获取天气数据。

    参数:
    city_url (str): 城市天气预报的URL地址。

    返回:
    list of dict: 包含天气数据的列表，每个元素是一个字典，包含日期、天气、最高温度、最低温度、风向和风力。
    """
    response = requests.get(city_url)  # 发送GET请求到指定的URL
    response.encoding = 'utf-8'  # 设置响应内容的编码为utf-8
    soup = BeautifulSoup(response.text, 'html.parser')  # 使用BeautifulSoup解析HTML内容

    weather_data = []  # 初始化一个空列表，用于存储天气数据
    for day in soup.find('ul', class_='t clearfix').find_all('li'):  # 查找包含天气信息的<li>元素列表
        date = day.find('h1').text  # 提取日期
        weather = day.find('p', class_='wea').text  # 提取天气状况
        temp_range = day.find('p', class_='tem')  # 提取温度范围信息
        temp_high = temp_range.find('span').text  # 提取最高温度
        temp_low = temp_range.find('i').text  # 提取最低温度
        wind = day.find('p', class_='win')  # 提取风力风向信息
        wind_direction = wind.find('em').find_all('span')[0]['title']  # 提取风向
        wind_force = wind.find('i').text  # 提取风力

        # 将提取的信息组织成一个字典，并添加到天气数据列表中
        weather_data.append({
            'date': date,
            'weather': weather,
            'temp_high': temp_high,
            'temp_low': temp_low,
            'wind_direction': wind_direction,
            'wind_force': wind_force,
        })

    return weather_data  # 返回天气数据列表


def save_to_csv(weather_data, filename):
    """
    将天气数据保存到CSV文件中。

    参数:
    weather_data (list of dict): 天气数据列表，每个元素是一个包含天气信息的字典。
    filename (str): 要保存的CSV文件的名称。
    """
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:  # 打开文件以写入，设置正确的编码和换行符
        fieldnames = ['date', 'weather', 'temp_high', 'temp_low', 'wind_direction', 'wind_force']  # 定义CSV文件的列名
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)  # 创建一个DictWriter对象，用于写入字典数据
        writer.writeheader()  # 写入列名作为CSV文件的头部
        for row in weather_data:  # 遍历天气数据列表
            writer.writerow(row)  # 将每个字典写入一行


# 杭州天气预报的URL
hangzhou_url = "http://www.weather.com.cn/weather/101210101.shtml"
weather_data = get_weather_data(hangzhou_url)  # 调用函数获取天气数据
save_to_csv(weather_data, 'hangzhou_weather.csv')  # 调用函数将天气数据保存到CSV文件
print('Hangzhou weather data for the next week saved to hangzhou_weather.csv')  # 输出提示信息