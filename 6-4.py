import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 设置中文字体显示，以便在图表中正确显示中文和负号
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载波士顿房价数据集
# 假设数据集'housing.csv'是以空格为分隔符的，并且没有表头
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
boston_df = pd.read_csv('housing.csv', header=None, delimiter=r"\s+", names=column_names)
print(boston_df.head(5))  # 打印数据集的前5行以查看

# 提取特征变量和目标变量
# 特征变量：房间数(RM), 低收入人口比例(LSTAT), 学生与教师比(PTRATIO), 工业用地比例(INDUS)
X = boston_df[['RM', 'LSTAT', 'PTRATIO', 'INDUS']]
# 目标变量：房屋中位数价值(MEDV)
y = boston_df['MEDV']

# 使用线性回归模型拟合数据
model = LinearRegression()
model.fit(X, y)

# 打印模型系数和截距
# 模型系数表示每个特征对目标变量的影响程度
print('模型系数: \n', model.coef_)
# 截距表示当所有特征为0时，目标变量的预测值
print('截距: \n', model.intercept_)

# 绘制预测值与实际值的散点图和一条表示理想情况的对角线（非真实线性回归线）
plt.figure(figsize=(8, 6))
plt.scatter(y, model.predict(X), color='blue')  # 绘制预测值与实际值的散点图
# 绘制一条对角线，仅表示预测值与实际值完全相等的情况（非模型预测结果）
plt.plot([0, 50], [0, 50], color='red', linestyle='--', linewidth=2)

# 添加灰色区域作为预测不确定性的示例（注意：这并非基于模型的实际预测不确定性）
lower_bound = [x - 5 for x in [0, 50]]  # 假设的下边界
upper_bound = [x + 5 for x in [0, 50]]  # 假设的上边界
plt.fill_between([0, 50], lower_bound, upper_bound, color='gray', alpha=0.2)

# 添加更宽的灰色区域作为另一个示例
lower_bound = [x - 10 for x in [0, 50]]  # 更宽的下边界
upper_bound = [x + 10 for x in [0, 50]]  # 更宽的上边界
plt.fill_between([0, 50], lower_bound, upper_bound, color='gray', alpha=0.2)

# 设置坐标轴范围
plt.xlim([0, 50])
plt.ylim([0, 50])
# 设置坐标轴标签和图表标题
plt.xlabel('实际房价（千元）', fontsize=12)
plt.ylabel('预测房价（千元）', fontsize=12)
plt.title('线性回归分析示例', fontsize=14)  # 修改标题以更准确地反映内容

# 展示图像
plt.show()

# 注意：
# 1. 这里的散点图实际上是在比较预测值与实际值，而不是特征与目标变量之间的关系。
# 2. 对角线（红色虚线）并不表示线性回归模型的预测结果，而只是表示预测值与实际值相等的情况。
# 3. 灰色区域仅作为预测不确定性的示例，并不反映模型的真实预测不确定性。
