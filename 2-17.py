import requests  # 导入requests库，用于发送HTTP请求
import csv  # 导入csv库，用于处理CSV文件


def get_douyin_data():
    """
    从抖音热榜API获取数据

    Returns:
        list: 包含抖音热榜数据的列表，每个元素是一个字典，包含标题和热度值
    """
    url = "https://www.iesdouyin.com/web/api/v2/hotsearch/billboard/word/"  # 抖音热榜API的URL
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36"
    }  # 设置请求头，模拟浏览器访问
    response = requests.get(url, headers=headers)  # 发送GET请求
    response.encoding = "utf-8"  # 设置响应编码为UTF-8

    if response.status_code == 200:  # 检查请求是否成功
        data = response.json()["word_list"]  # 从响应的JSON数据中获取热榜列表
        douyin_data = []  # 初始化一个空列表来存储处理后的数据
        for item in data:  # 遍历热榜列表中的每个项目
            douyin_data.append({
                "title": item["word"],  # 提取标题
                "hot_value": item["hot_value"],  # 提取热度值
            })
        return douyin_data  # 返回处理后的数据列表
    else:
        return []  # 如果请求失败，返回空列表


def save_to_csv(douyin_data, filename):
    """
    将抖音热榜数据保存到CSV文件中

    Args:
        douyin_data (list): 包含抖音热榜数据的列表
        filename (str): 要保存的CSV文件的名称
    """
    with open(filename, "w", newline="", encoding="utf-8") as csvfile:  # 打开文件准备写入
        fieldnames = ["title", "hot_value"]  # 定义CSV文件的列名
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)  # 创建DictWriter对象
        writer.writeheader()  # 写入列名作为文件头
        for row in douyin_data:  # 遍历数据列表
            writer.writerow(row)  # 将每个字典写入一行


# 调用函数获取抖音热榜数据
douyin_data = get_douyin_data()
if douyin_data:  # 检查是否成功获取到数据
    save_to_csv(douyin_data, "douyin_top.csv")  # 保存数据到CSV文件
    print("Douyin top headlines data saved to douyin_top.csv")  # 打印成功消息
else:
    print("Failed to fetch Douyin top headlines data.")  # 打印失败消息