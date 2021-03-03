"""
验证使用tf把通道拉成平面(carafe上采样得到卷积核)
"""
import numpy as np
import cv2
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

## 设置属性防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

import tensorflow as tf


def upsample_flatten(x, sigma=2):
    """
    :param x: [H, W, σ^2]
    :return:
    """
    x = tf.reshape(x, [-1, h, w, sigma, sigma])
    x = tf.transpose(x, perm=(0, 1, 3, 2, 4))
    x = tf.reshape(x, [-1, h * sigma, w * sigma])
    return x


def get_data(h, w, sigma):
    nums = h * w * sigma * sigma  # 数据总数

    # 展平后的目标图像
    y = np.arange(nums).reshape(sigma * h, sigma * w) / nums
    plt.imshow(y)
    plt.title('展平后的目标图像')
    plt.show()

    # 待展平的数据
    x = np.zeros([h, w, sigma ** 2])
    for i in range(h):
        for j in range(w):
            start_h = i * sigma
            start_w = j * sigma
            start = start_h * w * sigma + start_w
            _x = []
            for n in range(sigma):
                _x += list(range(start + n * w * sigma, start + n * w * sigma + sigma))
            x[i, j] = np.array(_x)

    return x, y


# 初始图像尺寸
h, w = 4, 6
sigma = 4  # 放大倍数

"""检验展平函数得到图像是否与预测一致"""

datax, datay = get_data(h, w, sigma)
datax = datax.reshape([1, *datax.shape])
datay = datay.reshape([1, -1])
inputs = tf.placeholder(tf.float32, [None, h, w, sigma ** 2])  # 用于输入原datax
output_y = upsample_flatten(inputs, sigma)  # [None, h, w]  # 得到展平的图像
with tf.Session() as sess:
    _y = sess.run(output_y, {inputs: datax})
_y = np.squeeze(_y)
plt.imshow(_y)
plt.title('函数得到的展平后的目标图像')
plt.show()

"""检查展平函数能否反向传播"""

inputs2 = tf.placeholder(tf.float32, [None, h, w, 3])
labels = tf.placeholder(tf.float32, [None, 10])

x = tf.layers.conv2d(inputs2, 10, (3, 3), (1, 1), padding='same')
x = tf.layers.conv2d(x, 10, (3, 3), (1, 1), padding='same')
x = tf.layers.conv2d(x, sigma ** 2, (3, 3), (1, 1), padding='same')
x = upsample_flatten(x, sigma)  # [None, h, w]  # 得到展平的图像
x = tf.layers.flatten(x)
x = tf.layers.dense(x, 50)
x = tf.layers.dense(x, 20)
x = tf.layers.dense(x, 10)

loss = tf.reduce_mean(tf.losses.mean_squared_error(labels, x))
train_op = tf.train.AdamOptimizer(0.001).minimize(loss)

bs = 4
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(500):
        _lo, _ = sess.run([loss, train_op], {inputs2: np.random.normal(size=[bs, h, w, 3]),
                                             labels: np.random.normal(size=[bs, 10])})
        print(f'epoch={epoch:.6f}\tloss={_lo:.6f}')
