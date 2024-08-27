import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 设置字体
# 查找系统中安装的 SimHei 字体路径
font_paths = [f.fname for f in fm.fontManager.ttflist if 'SimHei' in f.name]
simhei_path = font_paths[0] if font_paths else None

# 如果没有找到 SimHei 字体，则抛出错误
if not simhei_path:
    raise ValueError("SimHei 字体未找到，请确保已安装 SimHei 字体。")

# 加载 SimHei 字体
my_font = fm.FontProperties(fname=simhei_path)

# 读取数据
# 从 CSV 文件中加载 Google 的日内交易数据
data = pd.read_csv("googl_intraday_data.csv")
# 将日期列转换为 datetime 类型，并设置为 DataFrame 的索引
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

# 选择收盘价
# 提取收盘价列
close_price = data['close'].values.reshape(-1, 1)

# 数据归一化
# 使用 MinMaxScaler 对收盘价进行归一化处理
scaler = MinMaxScaler(feature_range=(0, 1))
close_price_scaled = scaler.fit_transform(close_price)

# 数据分割
# 分割数据为训练集和测试集
train_size = int(len(close_price_scaled) * 0.8)
train_data = close_price_scaled[:train_size]
test_data = close_price_scaled[train_size:]

# 序列化数据
# 定义函数创建序列化的数据集
def create_dataset(data, window_size):
    X, Y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size), 0])
        Y.append(data[i + window_size, 0])
    return np.array(X), np.array(Y)

# 设定窗口大小
window_size = 5
# 创建训练集和测试集
X_train, Y_train = create_dataset(train_data, window_size)
X_test, Y_test = create_dataset(test_data, window_size)

# 数据整形
# 将输入数据调整为 LSTM 所需的形状
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# 创建 LSTM 模型
# 初始化一个序贯模型
model = Sequential()

# 添加第一个 LSTM 层
# 参数解释：
# 50: 这个 LSTM 层有 50 个隐藏单元（记忆单元）。
# input_shape=(window_size, 1): 输入数据的形状。这里的 `window_size` 是序列长度，`1` 表示每个时间步的特征数量。
# return_sequences=True: 设置此参数表示该 LSTM 层返回所有时间步的输出，而不是仅返回最后一个时间步的输出。这对于堆叠多个 LSTM 层时非常重要，因为下一个 LSTM 层需要接收序列数据作为输入。
model.add(LSTM(50, input_shape=(window_size, 1), return_sequences=True))

# 添加第二个 LSTM 层
# 参数解释：
# 50: 这个 LSTM 层也有 50 个隐藏单元。
# 因为这是模型中的第二个 LSTM 层，所以不需要指定 `input_shape`，Keras 会自动推断输入形状。
# 默认情况下: 这个 LSTM 层返回的是最后一个时间步的输出，因为我们没有设置 `return_sequences=True`。
model.add(LSTM(50))

# 添加 Dense 层作为输出层
# 参数解释：
# 1: 输出层只有一个节点，这是因为我们需要预测单个数值，即下一个时间点的收盘价。
model.add(Dense(1))

# 编译模型
# 参数解释：
# loss='mean_squared_error': 使用均方误差作为损失函数，这是回归问题常用的损失函数。
# optimizer='adam': 使用 Adam 优化器，这是一种高效的梯度下降算法。
model.compile(loss='mean_squared_error', optimizer='adam')

# 训练模型
# 参数解释：
# X_train: 训练数据的输入。
# Y_train: 训练数据的目标值。
# epochs=100: 训练的周期数，即整个训练数据集被遍历的次数。
# batch_size=64: 每次更新权重时使用的样本数量。
# verbose=1: 显示训练过程中的进度条和指标。
model.fit(X_train, Y_train, epochs=100, batch_size=64, verbose=1)

# 预测
# 利用训练好的模型进行预测
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# 反归一化
# 将预测结果反归一化到原始规模
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# 计算准确率
# 计算训练集和测试集的均方根误差 (RMSE)
train_rmse = np.sqrt(mean_squared_error(Y_train, train_predict))
test_rmse = np.sqrt(mean_squared_error(Y_test, test_predict))
print("训练集 RMSE：", train_rmse)
print("测试集 RMSE：", test_rmse)

# 绘图
# 创建图形
plt.figure(figsize=(16, 8))

# 绘制原始数据
# 绘制测试集范围内的原始收盘价
plt.plot(close_price[-len(test_predict):], label="原始数据", color="blue")

# 绘制测试集预测
# 准备绘制预测结果的数据结构
test_predict_plot = np.empty_like(close_price[-len(test_predict):])
test_predict_plot[:, :] = np.nan
test_predict_plot[window_size:, :] = test_predict[:-window_size]
# 绘制测试集预测结果
plt.plot(test_predict_plot, label="测试集预测", color="red")

# 设置图表标题和标签
plt.xlabel("时间", fontproperties=my_font, fontsize=14)
plt.ylabel("股票收盘价", fontproperties=my_font, fontsize=14)
plt.title("Google 股票价格预测（聚焦测试集）", fontproperties=my_font, fontsize=16)
plt.legend(prop=my_font, fontsize=12)

# 显示图表
plt.show()