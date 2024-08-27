# 导入必要的库
import matplotlib.pyplot as plt  # 用于绘图
from scipy.cluster.hierarchy import dendrogram, linkage  # 用于层次聚类的树状图和链接
from scipy.spatial.distance import pdist, squareform  # 用于计算样本之间的距离矩阵
import matplotlib.font_manager as fm  # 用于管理字体
from sklearn.cluster import AgglomerativeClustering  # 用于执行层次聚类
from sklearn.metrics import silhouette_score  # 用于计算轮廓系数，评估聚类效果
import pandas as pd  # 用于数据处理
import numpy as np

# 获取 SimHei 字体的路径，以便在图表中使用中文
font_paths = [f.fname for f in fm.fontManager.ttflist if 'SimHei' in f.name]
simhei_path = font_paths[0] if font_paths else None

if not simhei_path:
    raise ValueError("SimHei 字体未找到，请确保已安装 SimHei 字体。")

my_font = fm.FontProperties(fname=simhei_path)  # 设置字体属性

# 读取数据
data = pd.read_csv("Mall_Customers.csv")  # 假设Mall_Customers.csv包含用于聚类的数据


# 层次聚类函数
def hierarchical_clustering(data, method='ward'):
    """
    执行层次聚类并绘制树状图。

    参数:
    data (DataFrame): 包含聚类特征的数据集。
    method (str): 层次聚类的合并方法，默认为'ward'。
    """
    # 计算距离矩阵
    distance_matrix = squareform(pdist(data, metric='euclidean'))  # 使用欧氏距离
    # 提取下三角矩阵（不包括对角线）
    #triu_indices = np.triu_indices(distance_matrix, k=1)
    #compressed_distance_matrix = distance_matrix[triu_indices]

    # 层次聚类
    linked = linkage(distance_matrix, method=method)  # 执行层次聚类

    # 绘制树状图
    plt.figure(figsize=(10, 7))  # 设置图表大小
    dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)  # 绘制树状图
    plt.title("层次聚类树状图", fontproperties=my_font, fontsize=30)  # 设置图表标题
    plt.xlabel("样本编号", fontproperties=my_font, fontsize=30)  # 设置x轴标签
    plt.ylabel("距离", fontproperties=my_font, fontsize=30)  # 设置y轴标签
    plt.show()  # 显示图表


# 绘制肘部法则曲线函数
def plot_elbow_curve(data, max_clusters=10):
    """
    绘制肘部法则曲线以选择最佳聚类数。

    参数:
    data (DataFrame): 包含聚类特征的数据集。
    max_clusters (int): 考虑的最大聚类数，默认为10。
    """
    cluster_range = range(2, max_clusters + 1)  # 生成聚类数范围
    silhouette_scores = []  # 存储不同聚类数的轮廓系数

    for n_clusters in cluster_range:
        clusterer = AgglomerativeClustering(n_clusters=n_clusters)  # 创建层次聚类器
        cluster_labels = clusterer.fit_predict(data)  # 执行聚类并获取标签
        silhouette_avg = silhouette_score(data, cluster_labels)  # 计算轮廓系数
        silhouette_scores.append(silhouette_avg)  # 存储轮廓系数

    plt.figure(figsize=(10, 7))  # 设置图表大小
    plt.plot(cluster_range, silhouette_scores, 'o-')  # 绘制曲线
    plt.title("肘部法则曲线", fontproperties=my_font, fontsize=20)  # 设置图表标题
    plt.xlabel("簇的数量", fontproperties=my_font, fontsize=20)  # 设置x轴标签
    plt.ylabel("轮廓系数", fontproperties=my_font, fontsize=20)  # 设置y轴标签
    plt.show()  # 显示图表


# 选择要聚类的特征
selected_features = ["Annual Income (k$)", "Spending Score (1-100)"]  # 假设这些特征对聚类很重要

# 绘制层次聚类树状图
hierarchical_clustering(data[selected_features])  # 调用层次聚类函数并绘制树状图

# 绘制肘部法则曲线
plot_elbow_curve(data[selected_features])