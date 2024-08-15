# 导入必要的库
import matplotlib.pyplot as plt  # 用于数据可视化
from matplotlib.font_manager import FontProperties  # 用于设置绘图中的中文字体
from sklearn.datasets import load_iris  # 用于加载鸢尾花数据集
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # 用于线性判别分析
import pandas as pd  # 用于数据处理和分析

# 加载鸢尾花数据集
iris = load_iris()

# 查看数据的描述
print(iris.DESCR)  # 打印数据集的详细描述

# 查看特征名称
print(iris.feature_names)  # 打印特征（列）的名称

# 查看目标名称（类别）
print(iris.target_names)  # 打印目标（类别）的名称

# 查看数据（特征矩阵）
print(iris.data)  # 打印特征矩阵（不包含目标值）

# 查看目标值（类别标签）
print(iris.target)  # 打印每个样本的类别标签

# 将特征和目标值转换为DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)  # 创建DataFrame，包含特征数据
df['target'] = iris.target_names[iris.target]  # 添加一列，包含样本的类别名称

# 显示前几行数据
print(df.head())  # 打印DataFrame的前几行，用于查看数据结构

# 使用线性判别分析（LDA）对数据进行降维
lda = LinearDiscriminantAnalysis(n_components=2)  # 创建一个LDA实例，指定降维后的特征数为2
X_lda = lda.fit_transform(iris.data, iris.target)  # 对特征矩阵和目标值进行拟合和转换，得到降维后的数据

# 设置中文字体，以便在图表中显示中文
font = FontProperties(fname='C:/Windows/Fonts/msyh.ttc', size=14)  # 指定字体路径和大小

# 绘制原始数据散点图
plt.figure(figsize=(12, 5))  # 设置图表大小
plt.subplot(121)  # 创建子图，1行2列中的第1个
plt.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target, cmap=plt.cm.Set1, edgecolor='k')  # 绘制散点图，使用前两个特征
plt.xlabel('花萼长度 (cm)', fontproperties=font)  # 设置x轴标签
plt.ylabel('花萼宽度 (cm)', fontproperties=font)  # 设置y轴标签
plt.title('原始数据', fontproperties=font)  # 设置图表标题
plt.xticks(fontproperties=font)  # 设置x轴刻度字体
plt.yticks(fontproperties=font)  # 设置y轴刻度字体
plt.tight_layout()  # 自动调整子图参数, 使之填充整个图像区域

# 绘制LDA降维后的散点图
plt.subplot(122)  # 创建子图，1行2列中的第2个
plt.scatter(X_lda[:, 0], X_lda[:, 1], c=iris.target, cmap=plt.cm.Set1, edgecolor='k')  # 绘制LDA降维后的散点图
plt.xlabel('LDA特征1', fontproperties=font)  # 设置x轴标签
plt.ylabel('LDA特征2', fontproperties=font)  # 设置y轴标签
plt.title('LDA降维后的数据', fontproperties=font)  # 设置图表标题
plt.xticks(fontproperties=font)  # 设置x轴刻度字体
plt.yticks(fontproperties=font)  # 设置y轴刻度字体
plt.tight_layout()  # 自动调整子图参数

plt.show()  # 显示图表