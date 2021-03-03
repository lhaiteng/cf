"""
验证使用np把通道拉成平面(carafe上采样得到卷积核)
"""
import numpy as np
import cv2
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

## 设置属性防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False


# import tensorflow as tf


def upsample_flatten(x, sigma=2):
    """
    :param x: [H, W, σ^2]
    :return:
    """
    h, w, sigma2 = np.shape(x)
    x = np.reshape(x, [h, w, sigma, sigma])
    x = np.transpose(x, axes=(0, 2, 1, 3))
    x = np.reshape(x, [h * sigma, w * sigma])
    return x


# 初始图像尺寸
h, w = 4, 6
sigma = 4  # 放大倍数
nums = h * w * sigma * sigma  # 数据总数

# 展平后的目标图像
y = np.arange(nums).reshape(sigma * h, sigma * w) / nums
plt.imshow(y)
plt.title('展平后的目标图像')
plt.show()

# 待展平的数据
x = np.zeros([h, w, sigma**2])
for i in range(h):
    for j in range(w):
        start_h = i * sigma
        start_w = j * sigma
        start = start_h * w * sigma + start_w
        _x = []
        for n in range(sigma):
            _x += list(range(start + n * w * sigma, start + n * w * sigma + sigma))
        x[i, j] = np.array(_x)

_y = upsample_flatten(x, sigma)
plt.imshow(_y)
plt.title('函数得到的展平后的目标图像')
plt.show()
