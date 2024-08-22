# 导入必要的库  
import pandas as pd  # 数据处理库  
import seaborn as sns  # 基于matplotlib的高级绘图库  
import matplotlib.pyplot as plt  # 绘图库  
import matplotlib.font_manager as fm  # 用于管理matplotlib中的字体  
from sklearn import datasets  # 加载数据集的库  
from sklearn.model_selection import train_test_split  # 划分训练集和测试集的函数  
from sklearn.neighbors import KNeighborsClassifier  # K-近邻分类器  
from sklearn.metrics import classification_report, confusion_matrix  # 评估分类性能的函数  
from matplotlib.font_manager import FontProperties  # 用于设置中文字体的类  

# 设置中文字体，以便在图表中显示中文  
my_font = FontProperties(fname='C:/Windows/Fonts/msyh.ttc', size=12)  # 指定中文字体路径和大小

# 加载 Iris 数据集  
iris = datasets.load_iris()  # 加载Iris数据集  
X = iris.data  # 数据集中的特征  
y = iris.target  # 数据集中的目标值（类别标签）  

# 将数据集划分为训练集和测试集  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 划分比例为80%训练集，20%测试集  

# 使用 K-近邻算法训练模型  
k = 3  # 设置K-近邻算法的邻居数为3  
knn = KNeighborsClassifier(n_neighbors=k)  # 创建K-近邻分类器实例  
knn.fit(X_train, y_train)  # 使用训练集数据训练模型  

# 预测测试集  
y_pred = knn.predict(X_test)  # 使用训练好的模型对测试集进行预测  

# 输出分类结果和混淆矩阵  
print("分类报告：")
print(classification_report(y_test, y_pred))  # 输出分类的详细报告，包括精确度、召回率等  
print("混淆矩阵：")
print(confusion_matrix(y_test, y_pred))  # 输出混淆矩阵，展示预测结果与实际结果的对比  

# 数据可视化  
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)  # 将特征数据转换为DataFrame，并设置列名  
iris_df['species'] = iris.target  # 添加目标值列  
iris_df['species'] = iris_df['species'].map({0: '山鸢尾', 1: '杂色鸢尾', 2: '维吉尼亚鸢尾'})  # 将目标值映射为中文类别名  

# 修改列名为中文  
iris_df.columns = ['萼片长度', '萼片宽度', '花瓣长度', '花瓣宽度', '种类']  # 将列名设置为中文  

sns.set(style="whitegrid", palette="husl")  # 设置Seaborn的绘图风格和颜色方案  
g = sns.pairplot(iris_df, hue='种类', markers=['o', 's', 'D'], diag_kind='kde')  # 绘制成对关系图，按种类着色，并设置标记和对角线分布类型  

# 设置字体  
for ax in g.axes.flat:  # 遍历所有子图  
    ax.set_xlabel(ax.get_xlabel(), fontproperties=my_font)  # 设置x轴标签字体  
    ax.set_ylabel(ax.get_ylabel(), fontproperties=my_font)  # 设置y轴标签字体  

# 由于Seaborn的pairplot默认图例位置可能不合适，我们需要手动设置图例  
handles = g._legend_data.values()  # 获取图例句柄  
labels = g._legend_data.keys()  # 获取图例标签  
g._legend.remove()  # 移除默认图例  
legend = g.fig.legend(handles=handles, labels=labels, loc='upper center', ncol=3, prop=my_font)  # 添加新的图例，并设置位置和字体  
legend.set_title('种类', prop=my_font)  # 设置图例标题和字体  

plt.show()  # 显示图表