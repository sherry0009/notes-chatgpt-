import requests
from bs4 import BeautifulSoup
import csv


# 定义获取新闻数据的函数
def get_news():
    url = 'https://news.sina.com.cn/world/'  # 设置目标URL为新浪国际新闻页面

    # 发送HTTP GET请求到目标URL
    response = requests.get(url)
    # 设置响应内容的编码为UTF-8
    response.encoding = 'utf-8'

    # 使用BeautifulSoup解析HTML内容
    soup = BeautifulSoup(response.text, 'html.parser')

    news_data = []  # 初始化一个空列表来存储新闻数据

    # 遍历所有具有特定类名的新闻项（注意：这里的'news-item'需要根据实际网页结构调整）
    for item in soup.find_all(class_='news-item'):  # 假设每个新闻项都有'news-item'这个类名
        headline_element = item.find('h2')  # 查找新闻标题
        if headline_element:  # 如果找到了标题
            headline = headline_element.text.strip()  # 清理并获取标题文本
            a_tag = item.find('a')  # 查找包含新闻链接的<a>标签
            if a_tag:  # 如果找到了<a>标签
                news_url = a_tag['href']  # 提取新闻链接
                # 从新闻链接获取时间和摘要
                time, summary = get_time_and_summary_from_url(news_url)

                # 将新闻数据添加到列表中
                news_data.append({
                    'headline': headline,  # 新闻标题
                    'time': time,  # 发布时间
                    'summary': summary  # 新闻摘要
                })

                # 返回包含所有新闻数据的列表
    return news_data


# 定义从新闻URL获取时间和摘要的函数
def get_time_and_summary_from_url(news_url):
    # 发送HTTP GET请求到新闻URL
    response = requests.get(news_url)
    # 设置响应内容的编码为UTF-8
    response.encoding = 'utf-8'

    # 使用BeautifulSoup解析HTML内容
    soup = BeautifulSoup(response.text, 'html.parser')

    # 查找时间元素（注意：这里的'date'需要根据实际网页结构调整）
    time_element = soup.find('span', class_='date')
    time = time_element.text.strip() if time_element else ''  # 如果找到了时间元素，则清理并获取文本，否则设置为空字符串

    # 查找摘要元素（注意：这里的'summary'需要根据实际网页结构调整）
    summary_element = soup.find('p', class_='summary')
    summary = summary_element.text.strip() if summary_element else ''  # 如果找到了摘要元素，则清理并获取文本，否则设置为空字符串

    # 返回时间和摘要
    return time, summary


# 定义将新闻数据保存到CSV文件的函数
def save_to_csv(news_data, filename):
    # 以写入模式打开CSV文件，设置编码为UTF-8
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        # 初始化CSV写入器，并设置字段名
        fieldnames = ['headline', 'time', 'summary']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # 写入表头
        writer.writeheader()

        # 遍历新闻数据列表，并将每条新闻写入CSV文件
        for row in news_data:
            writer.writerow(row)

        # 调用函数获取新闻数据，并将其保存到CSV文件


news_data = get_news()
save_to_csv(news_data, 'sina_news_today.csv')
print('News data saved to sina_news_today.csv')
