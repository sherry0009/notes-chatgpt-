import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.font_manager as fm

# 获取 SimHei 字体的路径，用于在图表中显示中文
font_paths = [f.fname for f in fm.fontManager.ttflist if 'SimHei' in f.name]
simhei_path = font_paths[0] if font_paths else None

if not simhei_path:
    raise ValueError("SimHei 字体未找到，请确保已安装 SimHei 字体。")

my_font = fm.FontProperties(fname=simhei_path)  # 设置字体属性

# 加载MNIST手写数字数据集
digits = datasets.load_digits()

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.5, random_state=42)

# 数据预处理：使用标准化方法，将数据缩放到均值为0，方差为1
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 创建SVM分类器，使用RBF（径向基函数）核
classifier = svm.SVC(kernel='rbf', gamma=0.01, C=10)

# 训练分类器
classifier.fit(X_train, y_train)

# 对测试集进行预测
y_pred = classifier.predict(X_test)

# 计算并打印分类器的准确率
accuracy = metrics.accuracy_score(y_test, y_pred)
print("准确率:", accuracy)

# 绘制混淆矩阵
# 对于多分类问题，混淆矩阵的维度会相应增加，每一行代表一个实际类别，每一列代表一个预测类别，表格中的元素i,j表示实际为第i类但被预测为第j类的样本数。
cm = metrics.confusion_matrix(y_test, y_pred)  # 计算混淆矩阵
plt.figure(figsize=(10, 10))  # 设置图表大小
sns.heatmap(cm, annot=True, fmt=".0f", linewidths=1, square=True, cmap="Blues", annot_kws={"fontsize": 14})  # 绘制热力图
plt.ylabel('实际类别', fontproperties=my_font, fontsize=14)  # 设置y轴标签
plt.xlabel('预测类别', fontproperties=my_font, fontsize=14)  # 设置x轴标签
plt.title('混淆矩阵', fontproperties=my_font, fontsize=20)  # 设置图表标题
plt.show()  # 显示图表