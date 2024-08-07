# 导入必要的库  
import pandas as pd  # 用于数据处理和分析  
import matplotlib.pyplot as plt  # 用于绘图  
import numpy as np  # 用于数学和科学计算  
from scipy.stats import norm  # 用于统计分布和拟合  

# 设置全局绘图参数，以支持中文显示和负号  
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为支持中文的'SimHei'，可根据需要更改  
plt.rcParams['axes.unicode_minus'] = False  # 设置负号显示，避免方块或乱码  

# 生成假设的数据集  
# 创建一个包含1000个随机正态分布数据的DataFrame  
np.random.seed(0)  # 设置随机种子，确保结果可重复  
data = pd.DataFrame({
    'Value': np.random.normal(loc=0, scale=1, size=1000)  # 生成均值为0，标准差为1的正态分布数据  
})

# 对数据进行描述性统计分析  
print("描述性统计:")
print(data.describe())  # 打印数据的描述性统计信息，如均值、标准差、最小值、四分位数等  

# 绘制数据分布的直方图  
plt.figure(figsize=(10, 6))  # 创建一个新的图形窗口，设置大小为10x6英寸  
plt.hist(data['Value'], bins=30, density=True, alpha=0.6, color='b')  # 绘制直方图，设置组数为30，归一化显示频率，设置透明度和颜色  
plt.title('数据分布直方图')  # 设置图形标题  
plt.xlabel('值')  # 设置x轴标签  
plt.ylabel('频率')  # 设置y轴标签  
plt.grid(True)  # 显示网格线  
plt.show()  # 显示图形  

# 绘制数据分布直方图与正态拟合曲线  
# 首先，使用scipy.stats.norm.fit计算数据的均值和标准差  
mu, std = norm.fit(data['Value'])  # 拟合数据到正态分布，并返回均值和标准差  

# 创建一个新的图形窗口  
plt.figure(figsize=(10, 6))
# 绘制数据的直方图  
plt.hist(data['Value'], bins=30, density=True, alpha=0.6, color='b', label='实际数据')  # 设置标签以便在图例中显示  

# 绘制拟合的正态分布曲线  
xmin, xmax = plt.xlim()  # 获取当前x轴的显示范围  
x = np.linspace(xmin, xmax, 100)  # 在x轴范围内生成100个等间距的点  
p = norm.pdf(x, mu, std)  # 计算这些点对应的正态分布概率密度值  
plt.plot(x, p, 'k', linewidth=2, label='拟合的正态分布')  # 绘制正态分布曲线  

# 设置图形标题、轴标签、图例和网格线  
plt.title('数据分布直方图与正态拟合')
plt.xlabel('值')
plt.ylabel('频率')
plt.legend()  # 显示图例  
plt.grid(True)
plt.show()  # 显示图形