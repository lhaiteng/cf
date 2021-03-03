"""
保存全部变量，只读取部分变量
"""
from io import BytesIO
import os, sys, math, random
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

batch_size = 32
x = tf.data.Dataset.from_tensors(tf.random_normal([10])).repeat().batch(batch_size)
x = x.make_initializable_iterator()
init_x = x.initializer
x = x.get_next()
y = tf.data.Dataset.range(10).repeat().shuffle(1000).batch(batch_size)
y = y.make_one_shot_iterator().get_next()
y = tf.cast(y, tf.float32)
w1 = tf.get_variable('w1', [10, 5], tf.float32, tf.ones_initializer())
w2 = tf.get_variable('w2', [5, 4], tf.float32, tf.ones_initializer())
w3 = tf.get_variable('w3', [4, 2], tf.float32, tf.ones_initializer())
w4 = tf.get_variable('w4', [2, 1], tf.float32, tf.ones_initializer(), trainable=False)

_x = tf.matmul(x, w1)
_x = tf.matmul(_x, w2)
_x = tf.matmul(_x, w3)
_x = tf.matmul(_x, w4)

_y = tf.squeeze(_x)
loss = tf.reduce_mean((_y - y) ** 2)
train_op = tf.train.AdamOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    _w1, _w2, _w3, _w4 = sess.run([w1, w2, w3, w4])
    print(f'w1={_w1}\nw2={_w2}\nw3={_w3}\nw4={_w4}')
    sess.run(init_x)
    for i in range(10000):
        _, lo = sess.run([train_op, loss])
        print(f'\ri={i} loss={lo:.3f}...', end='')
    _w1, _w2, _w3, _w4 = sess.run([w1, w2, w3, w4])
    print(f'w1={_w1}\nw2={_w2}\nw3={_w3}\nw4={_w4}')

    saver = tf.train.Saver()
    saver.save(sess, './ttt')

# 部分不同，部分相同，进行读取

tf.reset_default_graph()
w1 = tf.get_variable('w1', [5, 2], tf.float32, tf.ones_initializer())
w2 = tf.get_variable('w2', [2, 5], tf.float32, tf.ones_initializer())
w3 = tf.get_variable('w3', [4, 2], tf.float32, tf.ones_initializer())
w4 = tf.get_variable('w4', [1, 6], tf.float32, tf.ones_initializer())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    _w1, _w2, _w3, _w4 = sess.run([w1, w2, w3, w4])
    print(f'w1={_w1}\nw2={_w2}\nw3={_w3}\nw4={_w4}')

    loader = tf.train.Saver([w3])
    saver.restore(sess, './ttt')

    _w1, _w2, _w3, _w4 = sess.run([w1, w2, w3, w4])
    print(f'w1={_w1}\nw2={_w2}\nw3={_w3}\nw4={_w4}')

