import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

# 加载MNIST数据集
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 定义数字标签的名称
labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

# 绘制图像
fig, axs = plt.subplots(5, 5, figsize=(8, 8))
for i in range(5):
    for j in range(5):
        axs[i, j].imshow(X_train[i*5+j], cmap='gray')
        axs[i, j].axis('off')
        axs[i, j].set_title(labels[y_train[i*5+j]])

plt.show()
