# 导入绘图所需的模块
import matplotlib.pyplot as plt
# 导入字体管理模块，用于设置中文显示
import matplotlib.font_manager as fm

# 导入Keras中的CIFAR-10数据集加载模块
from keras.datasets import cifar10
# 导入Keras中的顺序模型构建模块
from keras.models import Sequential
# 导入Keras中的全连接层、卷积层、最大池化层和Dropout层模块
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
# 导入Keras中的Adam优化器模块
from keras.optimizers import Adam
# 导入Keras中的实用工具模块，用于将标签转换为one-hot编码
from keras.utils import to_categorical

# 获取 SimHei 字体的路径
font_paths = [f.fname for f in fm.fontManager.ttflist if 'SimHei' in f.name]
simhei_path = font_paths[0] if font_paths else None

if not simhei_path:
    raise ValueError("SimHei 字体未找到，请确保已安装 SimHei 字体。")

# 创建一个FontProperties对象，以便在matplotlib中使用SimHei字体
my_font = fm.FontProperties(fname=simhei_path)

# 加载CIFAR-10数据集
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# 数据预处理 - 将像素值归一化到0-1之间
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# 将标签转换为one-hot编码形式
# One-hot编码是一种表示类别数据的方法，它将每个类别值映射为一个二进制向量，在这个向量中只有一个位置是1，其余位置都是0。该位置对应的索引就是原始类别的数值索引。
# CNN的输出层通常使用softmax激活函数来输出概率分布，而训练目标则是one-hot编码的标签。因此，为了匹配CNN的输出和训练目标，我们需要将原始的整数标签转换为one-hot编码形式。
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 构建一个卷积神经网络模型
model = Sequential()  # 初始化一个顺序模型

# 第一层卷积层
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
# Conv2D: 定义一个二维卷积层
# 32: 这个数字指定了输出空间的维度，即输出特征图的数量
# (3, 3): 指定卷积核/滤波器的尺寸
# activation='relu': 激活函数类型，这里使用的是ReLU激活函数
# padding='same': 填充模式，使得输入和输出有相同的宽度和高度
# input_shape=(32, 32, 3): 输入图像的形状，对于CIFAR-10数据集，图像大小是32x32，有3个颜色通道（RGB）

# 第二层卷积层
model.add(Conv2D(32, (3, 3), activation='relu'))
# 这一层继续使用32个输出通道，使用3x3的卷积核，激活函数同样是ReLU

# 最大池化层
model.add(MaxPooling2D(pool_size=(2, 2)))
# MaxPooling2D: 定义一个二维最大池化层
# pool_size=(2, 2): 池化窗口的尺寸，这里使用2x2的窗口来下采样

# Dropout层
model.add(Dropout(0.25))
# Dropout: 一种正则化方法，随机丢弃一定比例的节点输出，防止过拟合
# 0.25: 指定丢弃的比例，即25%的节点会被暂时从网络中丢弃

# 第三层卷积层
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
# 这一层增加到64个输出通道，其余参数与第一层相同

# 第四层卷积层
model.add(Conv2D(64, (3, 3), activation='relu'))
# 同第二层，但使用64个输出通道

# 最大池化层
model.add(MaxPooling2D(pool_size=(2, 2)))

# Dropout层
model.add(Dropout(0.25))

# 展平层
model.add(Flatten())
# Flatten: 将多维输入展平为一维，通常用于连接卷积层和全连接层

# 全连接层
model.add(Dense(512, activation='relu'))
# Dense: 定义一个全连接层
# 512: 输出节点数量
# activation='relu': ReLU激活函数

# Dropout层
model.add(Dropout(0.5))

# 输出层
model.add(Dense(10, activation='softmax'))
# 这是网络的最后一层，用于分类
# 10: CIFAR-10数据集包含10类
# activation='softmax': 用于多分类任务的激活函数，将输出转化为概率分布

# 编译模型
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
# optimizer=Adam(lr=0.0001): 使用Adam优化器，学习率为0.0001
# loss='categorical_crossentropy': 分类任务常用的交叉熵损失函数
# metrics=['accuracy']: 在训练过程中监控模型的准确度

# 训练模型 - 使用训练数据拟合模型，并在每个epoch结束时验证模型性能
history = model.fit(X_train, y_train, batch_size=128, epochs=20, validation_data=(X_test, y_test), verbose=1)

# 评估模型 - 使用测试数据评估模型的最终性能
score = model.evaluate(X_test, y_test, verbose=0)
print('测试损失:', score[0])
print('测试准确率:', score[1])

# 绘制训练损失、验证损失和准确率
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 绘制损失函数子图
ax1.plot(history.history['loss'], label='训练损失', color='blue')  # 训练损失曲线
ax1.plot(history.history['val_loss'], label='验证损失', color='orange')  # 验证损失曲线
ax1.set_xlabel('迭代次数', fontproperties=my_font)  # 设置x轴标签
ax1.set_ylabel('损失', fontproperties=my_font)  # 设置y轴标签
ax1.legend(prop=my_font)  # 添加图例
ax1.set_title('损失函数曲线', fontproperties=my_font)  # 设置标题

# 绘制准确率子图
ax2.plot(history.history['accuracy'], label='训练准确率', color='blue')  # 训练准确率曲线
ax2.plot(history.history['val_accuracy'], label='验证准确率', color='orange')  # 验证准确率曲线
ax2.set_xlabel('迭代次数', fontproperties=my_font)  # 设置x轴标签
ax2.set_ylabel('准确率', fontproperties=my_font)  # 设置y轴标签
ax2.legend(prop=my_font)  # 添加图例
ax2.set_title('准确率曲线', fontproperties=my_font)  # 设置标题

# 显示图表
plt.show()