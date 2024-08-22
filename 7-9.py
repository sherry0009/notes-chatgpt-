# 导入所需的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib.font_manager import FontManager
from sklearn.manifold import TSNE

# 获取 SimHei 字体的路径
fm = FontManager()
font_paths = [f.fname for f in fm.ttflist if 'SimHei' in f.name]
simhei_path = font_paths[0] if font_paths else None

if not simhei_path:
    raise ValueError("SimHei 字体未找到，请确保已安装 SimHei 字体。")

# 设置字体为 SimHei
my_font = {'family': 'SimHei', 'size': 12}

# 获取数据集
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
X, y = newsgroups.data, newsgroups.target  # X 是文本数据，y 是对应的类别标签

# 数据预处理 - 使用 CountVectorizer 将文本数据转换为词频矩阵
vectorizer = CountVectorizer(stop_words='english', max_df=0.5, min_df=2)
X_vec = vectorizer.fit_transform(X)  # X_vec 是转换后的词频矩阵

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# 朴素贝叶斯分类器 - 多项式朴素贝叶斯
clf = MultinomialNB()
clf.fit(X_train, y_train)  # 使用训练集对分类器进行训练

# 预测
y_pred = clf.predict(X_test)  # 对测试集进行预测

# 分类报告 - 显示每个类别的精确度、召回率、F1分数等
print(classification_report(y_test, y_pred, target_names=newsgroups.target_names))

# 混淆矩阵 - 显示实际类别与预测类别之间的比较
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=newsgroups.target_names, columns=newsgroups.target_names)

# 绘制各类别预测准确率
accuracy_per_class = np.diag(cm) / np.sum(cm, axis=1)  # 计算每个类别的准确率
# 绘制条形图
fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(newsgroups.target_names, accuracy_per_class)
ax.set_title('各类别预测准确率', fontdict=my_font)
ax.set_xlabel('类别', fontdict=my_font)
ax.set_ylabel('准确率', fontdict=my_font)
ax.set_xticks(np.arange(len(newsgroups.target_names)))  # 设置 x 轴刻度位置
ax.set_xticklabels(newsgroups.target_names, rotation=90, fontsize=8)  # 设置 x 轴刻度标签
ax.tick_params(axis='y', labelsize=8)  # 设置 y 轴刻度标签大小
plt.show()

# 对测试集进行 t-SNE 降维
tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
X_test_2d = tsne.fit_transform(X_test.toarray())  # 将测试集数据降至二维

# 绘制分类图
fig, ax = plt.subplots(figsize=(8, 6))
# 遍历每个类别并绘制散点图
for i, target_name in enumerate(newsgroups.target_names):
    # 这里是针对每个类别 `i` 和对应的类别名称 `target_name` 来绘制散点图。
    #
    # 1. `X_test_2d`：这是经过 t-SNE 降维后得到的二维数据集，其中每个样本由两个坐标值表示，即 `(x, y)`。
    #
    # 2. `y_test == i`：这是一个布尔掩码，用于筛选出测试集中属于类别 `i` 的所有样本。当 `y_test` 中的值等于 `i` 时，该位置的值为 `True`，否则为 `False`。
    #
    # 3. `X_test_2d[y_test == i, 0]`：这会选择出属于类别 `i` 的所有样本的第一维坐标值（即 t-SNE 特征1的值）。这里的 `0` 表示选取第一列的数据。
    #
    # 4. `X_test_2d[y_test == i, 1]`：这会选择出属于类别 `i` 的所有样本的第二维坐标值（即 t-SNE 特征2的值）。这里的 `1` 表示选取第二列的数据。
    #
    # 5. `label=target_name`：这为每个类别设置了一个标签，用于图例中标识各个类别。
    #
    # 综上所述，这一行代码的作用是，对于每个类别 `i`，它会绘制出属于该类别的所有样本在 t-SNE 降维后的二维坐标系中的位置，并且为每个类别设置了标签，以便在图例中区分不同的类别。
    #
    # 这样做的目的是为了直观地展示不同类别在降维后的空间中的分布情况，有助于我们理解模型是如何区分不同类别的。
    ax.scatter(X_test_2d[y_test == i, 0], X_test_2d[y_test == i, 1], label=target_name)
ax.set_title('t-SNE降维后的分类图', fontdict=my_font)
ax.set_xlabel('t-SNE特征1', fontdict=my_font)
ax.set_ylabel('t-SNE特征2', fontdict=my_font)
ax.legend(prop={'family': 'SimHei', 'size': 12})  # 设置图例
ax.tick_params(axis='x', labelsize=8)  # 设置 x 轴刻度标签大小
ax.tick_params(axis='y', labelsize=8)  # 设置 y 轴刻度标签大小
plt.show()
