"""
试验tf中提取滑动窗口的函数tf.image.extract_image_patches()
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

def get_img(x, Kup=2):
    sizes = [1, Kup, Kup, 1]
    x = tf.image.extract_image_patches(x, sizes=sizes, strides=[1, 1, 1, 1], padding='SAME',
                                       rates=[1, 1, 1, 1])
    return x

H, W, C = 4, 6, 10
Kup = 3
inputs = tf.placeholder(tf.float32, [None, H, W, C])
outputs = get_img(inputs, Kup)

x = [[[[i for i in range(C)] for _ in range(W)] for _ in range(H)]]
x = np.array(x)

with tf.Session() as sess:
    x = sess.run(outputs, {inputs: x})
print(x.shape)
print(x[0, 0, 0])















