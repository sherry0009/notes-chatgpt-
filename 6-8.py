import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from matplotlib.font_manager import FontProperties

# 下载Wine Quality数据集
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
wine_data = pd.read_csv(url, sep=';')  # 读取CSV文件，分隔符为分号

# 提取特征和目标变量
X = wine_data[['alcohol']]  # 特征：酒精度
y = wine_data["quality"]  # 目标变量：葡萄酒质量

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# 70%的数据用于训练，30%的数据用于测试，随机状态设为42以确保结果可复现

# 设置多项式阶数范围
degrees = [1, 2, 3, 4]  # 尝试不同的多项式阶数来拟合数据

# 设置中文字体以支持中文显示
font = FontProperties(fname='C:/Windows/Fonts/msyh.ttc', size=14)

# 创建图形对象
plt.figure(figsize=(12, 8))

# 遍历不同的多项式阶数
for i, degree in enumerate(degrees):
    # 创建多项式回归模型管道
    # PolynomialFeatures(degree)将特征转换为多项式特征，LinearRegression()进行线性回归
    poly_reg = make_pipeline(PolynomialFeatures(degree), LinearRegression())

    # 训练模型
    poly_reg.fit(X_train, y_train)

    # 使用训练好的模型进行预测
    y_pred = poly_reg.predict(X_test)

    # 计算并打印均方误差(MSE)
    mse = mean_squared_error(y_test, y_pred)
    # 注意：这里并没有直接打印MSE，但在实际应用中，您可能会想要这样做

    # 绘制结果
    # 创建一个子图
    plt.subplot(2, 2, i + 1)
    # 绘制真实值和预测值的散点图
    plt.scatter(X_test, y_test, label='真实值', color='blue')
    plt.scatter(X_test, y_pred, label='预测值', color='red')
    # 设置x轴和y轴的标签
    plt.xlabel("酒精度", fontproperties=font)
    plt.ylabel("质量", fontproperties=font)
    # 设置子图的标题
    plt.title(f'多项式阶数 {degree}', fontproperties=font)

    # 绘制拟合曲线
    # 生成一个平滑的曲线来展示模型的拟合效果
    X_curve = np.linspace(X_test.min(), X_test.max(), 100).reshape(-1, 1)
    y_curve = poly_reg.predict(X_curve)
    plt.plot(X_curve, y_curve, 'g--', lw=3, label=f'拟合曲线 (MSE: {mse:.2f})')
    # 添加图例
    plt.legend(prop=font)

# 调整子图布局以避免重叠
plt.tight_layout()
# 显示图形
plt.show()