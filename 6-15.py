# 导入必要的库
import pandas as pd  # 用于数据处理和分析
import matplotlib.pyplot as plt  # 用于数据可视化
from sklearn.model_selection import train_test_split  # 用于划分训练集和测试集
from sklearn.tree import DecisionTreeRegressor  # 导入决策树回归模型
from sklearn.ensemble import RandomForestRegressor  # 导入随机森林回归模型
from sklearn.metrics import mean_squared_error  # 用于计算均方误差
from matplotlib.font_manager import FontProperties  # 用于设置中文字体

# 设置中文字体，以便在图表中显示中文
font = FontProperties(fname='C:/Windows/Fonts/msyh.ttc', size=14)

# 加载数据集
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
data = pd.read_excel(url)  # 从URL加载Excel文件

# 提取特征和目标变量
# 注意：这里假设数据集的最后一列是目标变量，其他列是特征
X = data.drop('Concrete compressive strength(MPa, megapascals) ', axis=1)  # 丢弃目标列，剩余的作为特征
y = data['Concrete compressive strength(MPa, megapascals) ']  # 目标变量

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 划分80%为训练集，20%为测试集

# 训练决策树回归模型
dt_model = DecisionTreeRegressor(random_state=42)  # 初始化决策树模型
dt_model.fit(X_train, y_train)  # 使用训练集数据训练模型
dt_predictions = dt_model.predict(X_test)  # 使用测试集数据进行预测

# 训练随机森林回归模型
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)  # 初始化随机森林模型，设置树的数量为100
rf_model.fit(X_train, y_train)  # 使用训练集数据训练模型
rf_predictions = rf_model.predict(X_test)  # 使用测试集数据进行预测

# 计算并打印MSE（均方误差），以评估模型性能
print("决策树模型的均方误差：", mean_squared_error(y_test, dt_predictions))
print("随机森林模型的均方误差：", mean_squared_error(y_test, rf_predictions))

# 绘制预测结果与真实值的散点图
fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)  # 创建一个包含两个子图的图形

# 第一个子图：决策树回归预测结果
axes[0].scatter(y_test, dt_predictions, c='blue', alpha=0.5)  # 绘制测试集实际值与预测值的散点图
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # 绘制对角线，用于对比
axes[0].set_xlabel('实际值', fontproperties=font)  # 设置x轴标签
axes[0].set_ylabel('预测值', fontproperties=font)  # 设置y轴标签
axes[0].set_title('决策树回归预测结果', fontproperties=font)  # 设置子图标题

# 第二个子图：随机森林回归预测结果
axes[1].scatter(y_test, rf_predictions, c='green', alpha=0.5)  # 绘制测试集实际值与预测值的散点图
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # 绘制对角线，用于对比
axes[1].set_xlabel('实际值', fontproperties=font)  # 设置x轴标签
axes[1].set_title('随机森林回归预测结果', fontproperties=font)  # 设置子图标题

plt.tight_layout()
plt.show()