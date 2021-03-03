# -*- coding: utf-8 -*-
"""
测试求如何使用traning滑动平均
"""
import itertools, cv2
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

"""构建网络"""
try:
    sess.close()
except:
    pass
tf.reset_default_graph()

inputs = tf.placeholder(tf.float32, [None, 5])
training = tf.placeholder(tf.bool, [])
w = tf.get_variable('w', [], dtype=tf.float32, initializer=tf.initializers.ones())


def update():
    assign_op = tf.assign_add(w, tf.reduce_mean(inputs))  # 一定要定义在函数体内，否则实现不了！！！
    with tf.control_dependencies([assign_op]):
        return inputs - w


def no_update():
    return inputs - w


result = tf.cond(training, update, no_update)

init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)
print('w = ', sess.run(w))

"""开始"""

x = np.random.randint(0, 10, [10, 5])
print(x)
print(f'x.mean={x.mean()}')

run_list = [w, result]
feed_dict = {inputs: x, training: True}
_w, _res = sess.run(run_list, feed_dict)
print(f'_w = {_w}')
print(f'_res =\n{_res}')
print(f'w = {sess.run(w)}')

run_list = [w, result]
feed_dict = {inputs: x, training: False}
_w, _res = sess.run(run_list, feed_dict)
print(f'_w = {_w}')
print(f'_res =\n{_res}')
print(f'w = {sess.run(w)}')
