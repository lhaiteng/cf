"""
carafe上采样
"""
import time
import numpy as np
import cv2
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf


def print_var_num():
    num = 0
    for var in tf.trainable_variables():
        s = 1
        for i in var.shape:
            s *= int(i)
        num += s
    print(f'total var parameters: {num}')


def get_carafe(x, sigma, Kup, Kencoder=None):
    H, W, C = tf.shape(x)[1], tf.shape(x)[2], x.shape[3]

    # 得到carefe核
    if Kencoder is None: Kencoder = Kup - 2
    Cm = C // 4
    kernel = tf.layers.conv2d(x, Cm, (1, 1), (1, 1), padding='same')
    Cup = sigma ** 2 * Kup ** 2
    kernel = tf.layers.conv2d(kernel, Cup, (Kencoder, Kencoder), (1, 1), padding='same')
    # # [None, H, W, Cup]
    kernel = tf.reshape(kernel, [-1, H, W, sigma, sigma, 1, Kup ** 2])
    # # [None, H, W, σ, σ, 1, Kup**2]
    kernel = tf.nn.softmax(kernel, axis=-1)  # [None, H, W, σ, σ, 1, Kup**2]

    # 进行上采样
    sizes = [1, Kup, Kup, 1]
    strides = [1, 1, 1, 1]
    rates = [1, 1, 1, 1]
    x = tf.image.extract_image_patches(x, sizes=sizes, strides=strides, rates=rates, padding='SAME')
    # # [None, H, W, C*Kup**2]
    # 按照tf.image.extract_image_patches的逻辑，通道把不同位置点的C堆叠起来，即[c1, c1, c1, ..., c2, c2, ...]
    x = tf.reshape(x, [-1, H, W, Kup ** 2, C])  # [None, H, W, Kup**2, C]
    x = tf.transpose(x, perm=[0, 1, 2, 4, 3])  # [None, H, W, C, Kup**2]
    x = tf.expand_dims(x, axis=3)
    x = tf.expand_dims(x, axis=3)
    # # [None, H, W, 1, 1, C, Kup**2]
    x = tf.multiply(x, kernel)  # [None, H, W, σ, σ, C, Kup**2]
    x = tf.reduce_sum(x, axis=-1)  # [None, H, W, σ, σ, C]
    # 调整成新图大小
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])  # [None, H, σ, W, σ, C]
    x = tf.reshape(x, [-1, sigma * H, sigma * W, C])

    return x


# 原始图像
H = W = 16
C = 1024
inputs = tf.placeholder(tf.float32, [None, H, W, 3], name='inputs')
x = tf.layers.conv2d(inputs, 64, (3, 3), (1, 1), padding='same')
x = tf.layers.conv2d(x, 128, (3, 3), (1, 1), padding='same')
x = tf.layers.conv2d(x, 256, (3, 3), (1, 1), padding='same')
x = tf.layers.conv2d(x, 512, (3, 3), (1, 1), padding='same')
x = tf.layers.conv2d(x, C, (3, 3), (1, 1), padding='same')
print_var_num()

# 上采样参数
sigma = 3  # 放大倍数
Kup = 5  # 感受野
outputs = get_carafe(x, sigma, Kup)  # [None, σH, σW, C]
# outputs = tf.layers.conv2d_transpose(x, C, (Kup, Kup), (sigma, sigma), padding='same')
print_var_num()

x = tf.layers.conv2d(outputs, 3, (3, 3), (1, 1), padding='same')
print_var_num()

labels = tf.placeholder(tf.float32, [None, sigma * H, sigma * W, 3], name='labels')
loss = tf.reduce_mean((labels - x) ** 2)
train_op = tf.train.AdamOptimizer(0.001).minimize(loss)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    bs = 2
    start = time.time()
    for epoch in range(100):
        datax = np.random.normal(size=[bs, H, W, 3])
        datay = np.random.normal(size=[bs, sigma * H, sigma * W, 3])
        lo, _ = sess.run([loss, train_op], {inputs: datax, labels: datay})
        print(f'epoch={epoch} loss={lo:.6f}')
    end = time.time()
    print(f'cost time: {end - start:.3f}')

    data = np.random.normal(size=[1, H, W, 3])
    y = sess.run(outputs, {inputs: data})
    print(y.shape)
