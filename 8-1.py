# 导入matplotlib.pyplot模块用于绘图  
import matplotlib.pyplot as plt

# 从keras.datasets模块导入cifar10数据集  
from keras.datasets import cifar10

# 加载CIFAR-10数据集，返回两个元组：训练集和测试集  
# 训练集包括图像数据(x_train)和对应的标签(y_train)  
# 测试集同样包括图像数据(x_test)和对应的标签(y_test)  
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 定义CIFAR-10数据集中各个类别的名称  
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# 创建一个图形窗口，设置大小为10x10英寸  
plt.figure(figsize=(10, 10))

# 遍历前25张训练图像进行显示  
for i in range(25):
    # 使用subplot在图形中创建一个子图区域，这里创建一个5x5的网格，并定位到当前索引的子图  
    plt.subplot(5, 5, i + 1)

    # 隐藏x轴和y轴的刻度  
    plt.xticks([])
    plt.yticks([])

    # 关闭网格线  
    plt.grid(False)

    # 显示图像，由于CIFAR-10的图像是RGB三通道的，所以不能直接使用binary颜色映射，这里使用默认或自定义颜色映射  
    # 注意：这里直接使用imshow可能不是最佳方式，因为CIFAR-10图像是RGB的，应该直接显示而不需要cmap参数  
    # 但为了保持示例的完整性，我们保留cmap参数，但实际应用中可能需要移除  
    plt.imshow(x_train[i])  # 如果图像颜色显示不正确，请尝试移除cmap参数  

    # 设置图像的x轴标签为对应类别的名称  
    # 注意：y_train[i]是一个数组，因为CIFAR-10是单标签多分类问题，但在这里我们只有一个标签，所以使用y_train[i][0]  
    plt.xlabel(class_names[y_train[i][0]])

# 显示整个图形窗口  
plt.show()

# 注意：在实际使用中，如果直接使用imshow显示CIFAR-10图像，可能会发现颜色有些奇怪或暗淡  
# 这是因为imshow默认将输入数据缩放到[0, 1]区间，并假设数据是灰度图（单通道）或具有特定的颜色映射  
# 对于RGB图像，直接调用imshow即可，不需要cmap参数，且图像数据应处于[0, 1]或[0, 255]（取决于imshow的输入解释）  
# 如果图像数据是uint8类型（即[0, 255]），则imshow通常能正确显示，但如果是float32等其他类型，可能需要先转换或归一化