# 导入必要的库  
import numpy as np  # 用于数学和科学计算  
import pandas as pd  # 用于数据处理和分析  
import matplotlib.pyplot as plt  # 用于绘图  
import seaborn as sns  # 基于matplotlib的高级绘图库  
from sklearn.model_selection import train_test_split  # 用于划分训练集和测试集  
from sklearn.neural_network import MLPRegressor  # 导入多层感知机回归模型  
from sklearn.metrics import mean_squared_error, r2_score  # 用于评估模型性能的指标  
from matplotlib.font_manager import FontProperties  # 用于设置中文字体  

# 设置中文字体，以便在图表中显示中文  
font = FontProperties(fname='C:/Windows/Fonts/msyh.ttc', size=14)

# 加载Abalone数据集  
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
column_names = ["Sex", "Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight",
                "Shell weight", "Rings"]
abalone_data = pd.read_csv(url, names=column_names)  # 读取CSV文件并指定列名  

# 数据预处理  
# 将'Sex'列的文本标签映射为数字标签  
abalone_data['Sex'] = abalone_data['Sex'].map({'M': 0, 'F': 1, 'I': 2})
# 分离特征（X）和目标变量（y）  
X = abalone_data.drop("Rings", axis=1)  # 特征集，不包括'Rings'列  
y = abalone_data["Rings"]  # 目标变量，即'Rings'列  

# 划分训练集和测试集  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 划分20%的数据作为测试集  

# 创建神经网络回归模型  
# 设置隐藏层大小为(50, 50)，激活函数为ReLU，优化器为Adam，最大迭代次数为500，随机种子为42  
regressor = MLPRegressor(hidden_layer_sizes=(50, 50), activation='relu', solver='adam', max_iter=500, random_state=42)
# 使用训练集数据训练模型  
regressor.fit(X_train, y_train)

# 预测测试集的结果  
y_pred = regressor.predict(X_test)

# 评估模型性能  
mse = mean_squared_error(y_test, y_pred)  # 计算均方误差  
r2 = r2_score(y_test, y_pred)  # 计算R^2分数  
print("MSE:", mse)  # 打印均方误差  
print("R2:", r2)  # 打印R^2分数  

# 绘制预测结果与实际结果的对比图  
fig, ax = plt.subplots(figsize=(12, 8))  # 创建一个图形和坐标轴  
# 使用seaborn的scatterplot绘制散点图，但这里的hue参数按y_test着色可能不是最佳实践，因为它不区分实际与预测  
# 这里主要是为了展示如何结合seaborn和matplotlib  
sns.scatterplot(x=y_test, y=y_pred, hue=y_test, palette='viridis', legend=None, ax=ax)  # 改为直接使用颜色和透明度
ax.set_xlabel('实际年龄', fontproperties=font)  # 设置x轴标签  
ax.set_ylabel('预测年龄', fontproperties=font)  # 设置y轴标签  
ax.set_title('实际年龄 vs 预测年龄', fontproperties=font)  # 设置图表标题  
# 绘制完美预测线（即y=x线）  
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.show()  # 显示图表  

# 注意：原代码中的hue=y_test可能不是预期的效果，因为它会在每个测试点旁边显示y_test的值，这里简化为直接设置颜色