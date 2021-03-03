# -*- coding: utf-8 -*-
"""
测试使用tf.variable_scope统一定义初始化器
1.如果scope的init与定义变量的不同
2.reuse的scope若改变init
3.scope中还包括scope，但内层未设置
4.scope中还包括scope，且两者不同
5.fc层的初始化器
6.conv层的初始化器

=> 1.如果scope中的变量，自己定义了初始化，则会覆盖scope的init
=> 2.reuse=True不会影响原定义的初始化
=> 3.嵌套的内层scope若未设置init，会沿用外层的设置
=> 4.会使用内层scope的规则，如果内层scop变量也定义了自己的初始化，则还是会使用自己的初始化
=> 5、6.会同时覆盖kernel和bias的初始化。原默认kernel随机取，而bias=0的。要小心使用。
=> 即会使用内层新定义的初始化，覆盖旧定义的初始化。重复使用变量时，不会更新初始化器。

"""
import itertools, cv2
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


"""0.默认的初始化"""
print('0.默认的初始化')
tf.reset_default_graph()
inputs = tf.placeholder(tf.float32, [2, 5])
with tf.variable_scope('test1', reuse=False):
    a1 = tf.get_variable('a1', shape=[5, 3], dtype=tf.float32)
    out_a1 = tf.matmul(inputs, a1)
b1 = tf.get_variable('b1', shape=[5, 3], dtype=tf.float32)
out_b1 = tf.matmul(inputs, b1)
print('设置scope test1的不设置默认初始化，a1不指定初始化，scope外的b1不指定。')
init_op = tf.global_variables_initializer()
x = np.arange(10).reshape([2, 5])
print(f'x=\n{x}')
with tf.Session() as sess:
    sess.run(init_op)
    print(f'a1=\n{sess.run(a1)}')
    print(f'out_a1=\n{sess.run(out_a1, {inputs:x})}')
    print(f'b1=\n{sess.run(b1)}')
    print(f'out_b1=\n{sess.run(out_b1, {inputs:x})}')
print('-'*100)


"""1.如果scope的init与定义变量的不同"""
print('1.如果scope的init与定义变量的不同')
tf.reset_default_graph()
inputs = tf.placeholder(tf.float32, [2, 5])
with tf.variable_scope('test1', reuse=False, initializer=tf.initializers.zeros()):
    a1 = tf.get_variable('a1', shape=[5, 3], dtype=tf.float32, initializer=tf.initializers.ones())
    out_a1 = tf.matmul(inputs, a1)
    b1 = tf.get_variable('b1', shape=[5, 3], dtype=tf.float32)
    out_b1 = tf.matmul(inputs, b1)
init_op = tf.global_variables_initializer()
print('设置scope test1的默认初始化0，a1指定初始化1，b1不指定。')
x = np.arange(10).reshape([2, 5])
print(f'x=\n{x}')
with tf.Session() as sess:
    sess.run(init_op)
    print(f'a1=\n{sess.run(a1)}')
    print(f'out_a1=\n{sess.run(out_a1, {inputs:x})}')
    print(f'b1=\n{sess.run(b1)}')
    print(f'out_b1=\n{sess.run(out_b1, {inputs:x})}')
print('-'*100)

"""2.reuse的scope若改变init"""
print('2.reuse的scope若改变init')
tf.reset_default_graph()
inputs = tf.placeholder(tf.float32, [2, 5])
with tf.variable_scope('test1', reuse=False, initializer=tf.initializers.zeros()):
    a1 = tf.get_variable('a1', shape=[5, 3], dtype=tf.float32, initializer=tf.initializers.ones())
    out_a1 = tf.matmul(inputs, a1)
    b1 = tf.get_variable('b1', shape=[5, 3], dtype=tf.float32)
    out_b1 = tf.matmul(inputs, b1)
with tf.variable_scope('test1', reuse=True, initializer=tf.initializers.ones()):
    a1 = tf.get_variable('a1', shape=[5, 3], dtype=tf.float32, initializer=tf.initializers.zeros())
    out_a1 = tf.matmul(inputs, a1)
    b1 = tf.get_variable('b1', shape=[5, 3], dtype=tf.float32)
    out_b1 = tf.matmul(inputs, b1)
print('设置scope test1的默认初始化0，a1指定初始化1，b1不指定。')
print('设置reuse test1的默认初始化1，a1指定初始化0，b1不指定。')
init_op = tf.global_variables_initializer()
x = np.arange(10).reshape([2, 5])
print(f'x=\n{x}')
with tf.Session() as sess:
    sess.run(init_op)
    print(f'a1=\n{sess.run(a1)}')
    print(f'out_a1=\n{sess.run(out_a1, {inputs:x})}')
    print(f'b1=\n{sess.run(b1)}')
    print(f'out_b1=\n{sess.run(out_b1, {inputs:x})}')
print('-'*100)


"""3.scope中还包括scope，但内层未设置"""
print('3.scope中还包括scope，但内层未设置')
tf.reset_default_graph()
inputs = tf.placeholder(tf.float32, [2, 5])
with tf.variable_scope('test1', reuse=False, initializer=tf.initializers.zeros()):
    a1 = tf.get_variable('a1', shape=[5, 3], dtype=tf.float32, initializer=tf.initializers.ones())
    out_a1 = tf.matmul(inputs, a1)
    b1 = tf.get_variable('b1', shape=[5, 3], dtype=tf.float32)
    out_b1 = tf.matmul(inputs, b1)
    with tf.variable_scope('test2', reuse=False):
        a2 = tf.get_variable('a2', shape=[5, 3], dtype=tf.float32)
        out_a2 = tf.matmul(inputs, a2)
        b2 = tf.get_variable('b2', shape=[5, 3], dtype=tf.float32)
        out_b2 = tf.matmul(inputs, b2)
print('设置scope test1的默认初始化0，a1指定初始化1，b1不指定。')
print('嵌套scope test2的不设置默认初始化，a2不指定，b2不指定。')
init_op = tf.global_variables_initializer()
x = np.arange(10).reshape([2, 5])
print(f'x=\n{x}')
with tf.Session() as sess:
    sess.run(init_op)
    print('外层')
    print(f'a1=\n{sess.run(a1)}')
    print(f'out_a1=\n{sess.run(out_a1, {inputs:x})}')
    print(f'b1=\n{sess.run(b1)}')
    print(f'out_b1=\n{sess.run(out_b1, {inputs:x})}')
    print('内层')
    print(f'a2=\n{sess.run(a2)}')
    print(f'out_a2=\n{sess.run(out_a2, {inputs:x})}')
    print(f'b2=\n{sess.run(b2)}')
    print(f'out_b2=\n{sess.run(out_b2, {inputs:x})}')
print('-'*100)


"""4.scope中还包括scope，且两者不同"""
print('4.scope中还包括scope，且两者不同')
tf.reset_default_graph()
inputs = tf.placeholder(tf.float32, [2, 5])
with tf.variable_scope('test1', reuse=False, initializer=tf.initializers.zeros()):
    a1 = tf.get_variable('a1', shape=[5, 3], dtype=tf.float32, initializer=tf.initializers.ones())
    out_a1 = tf.matmul(inputs, a1)
    b1 = tf.get_variable('b1', shape=[5, 3], dtype=tf.float32)
    out_b1 = tf.matmul(inputs, b1)
    with tf.variable_scope('test2', reuse=False, initializer=tf.initializers.ones()):
        a2 = tf.get_variable('a2', shape=[5, 3], dtype=tf.float32, initializer=tf.initializers.zeros())
        out_a2 = tf.matmul(inputs, a2)
        b2 = tf.get_variable('b2', shape=[5, 3], dtype=tf.float32)
        out_b2 = tf.matmul(inputs, b2)
print('设置scope test1的默认初始化0，a1指定初始化1，b1不指定。')
print('嵌套scope test2的默认初始化1，a2指定初始化0，b2不指定。')
init_op = tf.global_variables_initializer()
x = np.arange(10).reshape([2, 5])
print(f'x=\n{x}')
with tf.Session() as sess:
    sess.run(init_op)
    print(f'a1=\n{sess.run(a1)}')
    print(f'out_a1=\n{sess.run(out_a1, {inputs:x})}')
    print(f'b1=\n{sess.run(b1)}')
    print(f'out_b1=\n{sess.run(out_b1, {inputs:x})}')
    print(f'a2=\n{sess.run(a2)}')
    print(f'out_a2=\n{sess.run(out_a2, {inputs:x})}')
    print(f'b2=\n{sess.run(b2)}')
    print(f'out_b2=\n{sess.run(out_b2, {inputs:x})}')
print('-'*100)




"""5.fc层的初始化器"""
print('5.fc层的初始化器')
tf.reset_default_graph()
inputs = tf.placeholder(tf.float32, [2, 5])
out_default = tf.layers.dense(inputs, 3, name='default')
with tf.variable_scope('test1', reuse=False, initializer=tf.initializers.zeros()):
    out_a1 = tf.layers.dense(inputs, 3, name='a1')
    out_b1 = tf.layers.dense(inputs, 3, kernel_initializer=tf.initializers.ones(), name='b1')
    out_c1 = tf.layers.dense(inputs, 3, bias_initializer=tf.initializers.ones(), name='c1')
print('设置scope test1的默认初始化0，a1不指定，b1指定kernel1，c1指定bias1。')
init_op = tf.global_variables_initializer()
x = np.arange(10).reshape([2, 5])
print(f'x=\n{x}')
with tf.Session() as sess:
    sess.run(init_op)
    for v in tf.trainable_variables():
        print(f'{v.name}=\n{sess.run(v)}')
    print(f'out_a1=\n{sess.run(out_a1, {inputs:x})}')
    print(f'out_b1=\n{sess.run(out_b1, {inputs:x})}')
    print(f'out_c1=\n{sess.run(out_c1, {inputs:x})}')
print('-'*100)

"""6.conv层的初始化器"""
print('6.conv层的初始化器')
tf.reset_default_graph()
inputs = tf.placeholder(tf.float32, [2, 2, 2, 5])
out_default = tf.layers.conv2d(inputs, 3, 1, 1, name='default')
with tf.variable_scope('test1', reuse=False, initializer=tf.initializers.zeros()):
    out_a1 = tf.layers.conv2d(inputs, 3, 1, 1, name='a1')
    out_b1 = tf.layers.conv2d(inputs, 3, 1, 1, kernel_initializer=tf.initializers.ones(), name='b1')
    out_c1 = tf.layers.conv2d(inputs, 3, 1, 1, bias_initializer=tf.initializers.ones(), name='c1')
print('设置scope test1的默认初始化0，a1不指定，b1指定kernel1，c1指定bias1。')
init_op = tf.global_variables_initializer()
x = np.arange(40).reshape([2, 2, 2, 5])
print(f'x=\n{x}')
with tf.Session() as sess:
    sess.run(init_op)
    for v in tf.trainable_variables():
        print(f'{v.name}=\n{sess.run(v)}')
    print(f'out_a1=\n{sess.run(out_a1, {inputs:x})}')
    print(f'out_b1=\n{sess.run(out_b1, {inputs:x})}')
    print(f'out_c1=\n{sess.run(out_c1, {inputs:x})}')
print('-'*100)

