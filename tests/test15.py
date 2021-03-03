# -*- coding: utf-8 -*-
"""
测试写prelu
=> perlu3与手算结果一致
"""
import itertools, cv2
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
def prelu(data, name, initializer=None):
    alpha = tf.get_variable(name, shape=data.shape[-1], dtype=tf.float32, initializer=initializer)
    _data = tf.nn.relu(data)
    return _data + alpha * (data - tf.abs(data))


def prelu2(data, name, initializer=None):
    alpha = tf.get_variable(name, shape=data.shape[-1], dtype=tf.float32, initializer=initializer)
    _data = tf.nn.relu(data)
    return _data + alpha * (data - tf.stop_gradient(tf.abs(data)))

def prelu3(data, name, initializer=None):
    alpha = tf.get_variable(name, shape=data.shape[-1], dtype=tf.float32, initializer=initializer)
    return tf.nn.relu(data) - alpha * tf.nn.relu(-data)

tf.reset_default_graph()

inputs = tf.placeholder(tf.float32, [None, 3])
labels = tf.placeholder(tf.float32, [None, 1])
w = tf.get_variable('w', shape=[3, 1], dtype=tf.float32, initializer=tf.ones_initializer())
x = tf.matmul(inputs, w)
y1 = tf.nn.relu(x)
y2 = tf.nn.leaky_relu(x)
with tf.variable_scope('test', reuse=False, initializer=tf.ones_initializer()):
    y31 = prelu(x, 'prelu1')
    y32 = prelu2(x, 'prelu2')
    y33 = prelu3(x, 'prelu3')

with tf.variable_scope('test', reuse=True):
    alpha1 = tf.get_variable('prelu1')
    alpha2 = tf.get_variable('prelu2')
    alpha3 = tf.get_variable('prelu3')

loss1 = tf.reduce_sum(labels - y1)
loss2 = tf.reduce_sum(labels - y2)
loss31 = tf.reduce_sum(labels - y31)
loss32 = tf.reduce_sum(labels - y32)
loss33 = tf.reduce_sum(labels - y33)

opt = tf.train.GradientDescentOptimizer(1)
init_op = tf.global_variables_initializer()

data = np.array([[-1, 2, 3], [-10, 2, 1], [-2, -1, 0]])
data_label = np.array([[1], [-5], [-2]])


# relu

with tf.Session() as sess:
    sess.run(init_op)
    print(f'relu w_init=\n{sess.run(w)}')
    sess.run(opt.minimize(loss1), {inputs: data, labels: data_label})
    _w = sess.run(w)
    print(f'relu w=\n{_w}')



# leaky_slop = 0.2

with tf.Session() as sess:
    sess.run(init_op)
    print(f'leaky_relu w_init=\n{sess.run(w)}')
    sess.run(opt.minimize(loss2), {inputs: data, labels: data_label})
    _w = sess.run(w)
    print(f'leaky_relu w=\n{_w}')



# prelu1 without stop_gradient

print('prelu without stop_gradient')
with tf.Session() as sess:
    sess.run(init_op)
    print(f'w_init=\n{sess.run(w)}')
    print(f'alpha1_init =\n{sess.run(alpha1)}')
    sess.run(opt.minimize(loss31), {inputs: data, labels: data_label})
    print(f'w = \n{sess.run(w)}')
    print(f'alpha1 =\n{sess.run(alpha1)}')
print('-'*100)



# prelu2 without stop_gradient

print('prelu with stop_gradient')
with tf.Session() as sess:
    sess.run(init_op)
    print(f'w_init=\n{sess.run(w)}')
    print(f'alpha2_init =\n{sess.run(alpha2)}')
    sess.run(opt.minimize(loss32), {inputs: data, labels: data_label})
    print(f'w = \n{sess.run(w)}')
    print(f'alpha2 =\n{sess.run(alpha2)}')
print('-'*100)


# prelu3

print('prelu with stop_gradient')
with tf.Session() as sess:
    sess.run(init_op)
    print(f'w_init=\n{sess.run(w)}')
    print(f'alpha3_init =\n{sess.run(alpha3)}')
    sess.run(opt.minimize(loss33), {inputs: data, labels: data_label})
    print(f'w = \n{sess.run(w)}')
    print(f'alpha3 =\n{sess.run(alpha3)}')
print('-'*100)


