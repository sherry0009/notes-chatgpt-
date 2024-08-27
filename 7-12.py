import matplotlib.pyplot as plt
import tensorflow as tf

# 加载MNIST数据集
# 使用TensorFlow的Keras API中的mnist模块来加载数据
mnist = tf.keras.datasets.mnist

# 加载数据并检查是否成功
# mnist.load_data() 返回两个元组：(训练数据, 训练标签), (测试数据, 测试标签)
# 这里我们只关心训练数据和训练标签
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 数据预处理：归一化到 [0, 1]
# 由于MNIST图像的像素值范围是[0, 255]，为了模型训练的稳定性，我们通常会将数据归一化到[0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0

# 定义数字标签的名称
# 这是一个列表，包含了0到9的数字对应的字符串表示，用于后续给图像设置标题
labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

# 绘制图像
# 使用Matplotlib的subplots函数创建一个5x5的子图网格
fig, axs = plt.subplots(5, 5, figsize=(8, 8))

# 遍历子图网格的每一个位置
for i in range(5):
    for j in range(5):
        # 计算当前子图对应的训练数据中的索引
        # 由于我们想要绘制前25张图像（5x5），所以使用i*5+j来计算索引
        index = i * 5 + j

        # 使用imshow函数在子图上绘制图像
        # cmap='gray'表示使用灰度颜色图
        axs[i, j].imshow(x_train[index], cmap='gray')

        # 关闭坐标轴，使图像更加美观
        axs[i, j].axis('off')

        # 设置子图的标题为对应的手写数字标签
        # 使用y_train[index]来获取当前图像对应的标签索引，然后从labels列表中获取标签字符串
        axs[i, j].set_title(labels[y_train[index]])

    # 调整子图参数，使之填充整个图像区域，避免子图之间的重叠
plt.tight_layout()

# 显示图像
plt.show()