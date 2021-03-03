# -*- coding: utf-8 -*-
"""
测试tf进行全局池化，以及现图片分辨率经过resnet不加全局池化前特征图尺寸
"""
import cv2, os, math, time, random
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

shapes = []
for cnn_shape in (16, 20, 32, 50, 64, 80, 100, 120, 128, 160):
    inputs = tf.placeholder(tf.float32, shape=[None, cnn_shape, cnn_shape, 3], name='inputs')

    x = tf.layers.conv2d(inputs, 5, (7, 7), (2, 2), padding='same')
    x = tf.layers.max_pooling2d(x, (3, 3), (2, 2), padding='same')

    x = tf.layers.conv2d(x, 5, (1, 1), (2, 2), padding='same')
    x = tf.layers.conv2d(x, 5, (1, 1), (2, 2), padding='same')
    x = tf.layers.conv2d(x, 5, (1, 1), (2, 2), padding='same')
    x1 = x

    x = tf.layers.average_pooling2d(x, x.shape[1:3], x.shape[1:3], padding='valid')
    x2 = x
    x = tf.squeeze(x, [1, 2])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        r1, r2, r = sess.run([x1, x2, x],
                             feed_dict={inputs: np.random.normal(size=[10, cnn_shape, cnn_shape, 3])})
        shapes.append([r1.shape, r2.shape, r.shape])
for i, cnn_shape in enumerate((16, 20, 32, 50, 64, 80, 100, 120, 128, 160)):
    print(f'cnn_shape={cnn_shape}')
    print(shapes[i])
