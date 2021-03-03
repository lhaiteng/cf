# -*- coding: utf-8 -*-
"""
对比不同方式求多分类focal loss和二分类focal loss
多分类focal loss:
方式1：使用下标提取对应类别的p
方式2：reduce_sum求对应类别的p
方式3：使用labels_onehot掩码
方式4：使用tf内置交叉熵-样本的类别p使用gather得到
方式5：使用tf内置交叉熵-样本的类别p使用reduce_sum得到
=> 初始前向传播损失都相同
=> 训练后，方式1、2、3相同，方式4、5相同。
=> 归根结底，还是tf内置的softmax_cross_entropy_with_logits_v2与自己计算有不同

二分类focal loss:
先求出全部的概率p, log(p), 1-p, log(1-p)
方式1：把labels看作掩码，使用where获取正负样本下标，使用gather提取
方式2：使用where分别获取正负样本的掩码，掩码分别与正负样本各自的p、log(p)点乘
=> 前向loss有一点不同1e-3
=> 训练后得到的权重、预测结果完全相同，但loss仍有一点不同1e-3
"""
import numpy as np
import matplotlib.pyplot as plt
import os, math, random
import tensorflow as tf

"""
*******************************************************************************
对比不同方式计算的多分类focal loss
*******************************************************************************
"""

tf.reset_default_graph()
# 参数
num_cls = 20  # 种类数
input_c = 100  # 输入数据维度
nums = 1000  # 总数据
num_epoch = 10000  # 训练批次
lr = 0.001
eps = 1e-15
alpha, gamma = 1, 2
# 占位符
inputs = tf.placeholder(tf.float32, [None, input_c])
labels = tf.placeholder(tf.int32, [None])
labels_onehot = tf.one_hot(labels, num_cls)
# 网络结构
w1 = tf.get_variable('w1', shape=[input_c, 100], dtype=tf.float32, initializer=tf.initializers.ones)
logits = tf.matmul(inputs, w1)  # [None, 100]
w2 = tf.get_variable('w2', shape=[100, num_cls], dtype=tf.float32, initializer=tf.initializers.ones)
logits = tf.matmul(logits, w2)  # [None, num_cls]

predict = tf.argmax(logits, axis=-1)
softmax = tf.nn.softmax(logits)
p = tf.maximum(softmax, eps)  # [None, num_cls]

opt = tf.train.GradientDescentOptimizer(lr)
init_op = tf.initializers.global_variables()

"""定义损失函数的不同方式"""

# 方式1：使用下标提取对应类别的p
inds = tf.range(0, tf.shape(logits)[0]) * logits.shape[1] + labels
_p = tf.gather(tf.reshape(p, [-1]), inds)
_one_p = tf.ones_like(_p, dtype=_p.dtype) - _p
_loss = alpha * (_one_p ** gamma) * tf.log(_p)
loss1 = -tf.reduce_mean(_loss)
train1 = opt.minimize(loss1)

# 方式2：reduce_sum求对应类别的p
_p = tf.reduce_sum(labels_onehot * p, axis=1)
_one_p = tf.ones_like(_p, dtype=_p.dtype) - _p
_loss = alpha * _one_p ** gamma * tf.log(_p)
loss2 = -tf.reduce_mean(_loss)
train2 = opt.minimize(loss2)

# 方式3：使用掩码
zeros = tf.zeros_like(p, dtype=p.dtype)
weight = tf.where(labels_onehot > zeros, labels_onehot - p, zeros)
_loss = alpha * weight ** gamma * labels_onehot * tf.log(p)
loss3 = -tf.reduce_mean(_loss) * num_cls
train3 = opt.minimize(loss3)

# 方式4：使用tf内置交叉熵-样本的类别p使用gather得到
_ce = tf.nn.softmax_cross_entropy_with_logits_v2(labels_onehot, logits)
inds = tf.range(0, tf.shape(logits)[0]) * logits.shape[1] + labels
_p = tf.gather(tf.reshape(p, [-1]), inds)
_one_p = tf.ones_like(_p, dtype=_p.dtype) - _p
_loss = alpha * (_one_p ** gamma) * _ce
loss4 = tf.reduce_mean(_loss)
train4 = opt.minimize(loss4)

# 方式5：使用tf内置交叉熵-样本的类别p使用reduce_sum得到
_ce = tf.nn.softmax_cross_entropy_with_logits_v2(labels_onehot, logits)
_p = tf.reduce_sum(labels_onehot * p, axis=1)
_one_p = tf.ones_like(_p, dtype=_p.dtype) - _p
_loss = alpha * _one_p ** gamma * _ce
loss5 = tf.reduce_mean(_loss)
train5 = opt.minimize(loss5)

# 汇总
losses = [loss1, loss2, loss3, loss4, loss5]
trains = [train1, train2, train3, train4, train5]
num_method = len(trains)

"""生成数据，并对比初始损失是否相同"""
ys = np.random.randint(0, num_cls, size=[nums])
xs = np.array([np.random.normal(loc=n, scale=1, size=[input_c]) for n in ys])
print(f'xs.shape = {xs.shape}, ys.shape = {ys.shape}')

# 因为使用的相同的网络，pred在初始时候应该一样
print("各方法的初始损失对比")
with tf.Session() as sess:
    sess.run(init_op)
    for i1 in range(num_method - 1):
        for i2 in range(i1 + 1, num_method):
            lo1, lo2 = sess.run([losses[i1], losses[i2]], feed_dict={inputs: xs, labels: ys})
            print(f'lo{i1 + 1}-lo{i2 + 1}={np.abs(lo1 - lo2)}')

"""循环几种方式进行训练"""
w1s = []
w2s = []
preds = []
for i in range(num_method):
    with tf.Session() as sess:
        sess.run(init_op)
        for ind_epoch in range(num_epoch):
            sess.run(trains[i], feed_dict={inputs: xs, labels: ys})
        _w1, _w2, _pred = sess.run([w1, w2, predict], feed_dict={inputs: xs})
        w1s.append(_w1)
        w2s.append(_w2)
        preds.append(_pred)

"""对比计算结果"""
print('*' * 100)
print('对比不同方式的多分类focal loss.')
# 对比w1
for i1 in range(num_method - 1):
    for i2 in range(i1 + 1, num_method):
        res = np.sum(np.abs(w1s[i1] - w1s[i2]))
        print(f'w1 method{i1 + 1}-method{i2 + 1} = {res}')
print('-' * 100)
# 对比w2
for i1 in range(num_method - 1):
    for i2 in range(i1 + 1, num_method):
        res = np.sum(np.abs(w2s[i1] - w2s[i2]))
        print(f'w2 method{i1 + 1}-method{i2 + 1} = {res}')
print('-' * 100)
#  对比pred
for i1 in range(num_method - 1):
    for i2 in range(i1 + 1, num_method):
        res = np.sum(np.abs(preds[i1] - preds[i2]))
        print(f'pred method{i1 + 1}-method{i2 + 1} = {res}')
print('-' * 100)

# 预测分布
num_pics = num_method * (num_method - 1) / 2
cols = 3
rows = math.ceil(num_pics / cols)
plt.figure(figsize=[6 * cols, 6 * rows])

no = 0
for i1 in range(num_method - 1):
    for i2 in range(i1 + 1, num_method):
        no += 1
        plt.subplot(rows, cols, no)
        plt.plot(range(nums), ys, color='r', label='label')
        plt.plot(range(nums), preds[i1], label=f'method{i1 + 1}')
        plt.plot(range(nums), preds[i2], label=f'method{i2 + 1}')
        plt.ylim([-1, num_cls + 1])
        plt.legend()
        plt.title(f'method{i1 + 1} - method{i2 + 1}')

plt.suptitle('pred')
plt.show()

"""
*******************************************************************************
对比不同方式计算的二分类focal loss
*******************************************************************************
"""

"""参数"""
nums = 1000  # 总数据
num_epoch = 20  # 训练批次
input_c = 21  # 输入数据维度
output_c = 1  # 输出数据维度
lr = 0.001  # 学习率
c_w1 = 22  # w1通道数
c_w2 = 13  # w2通道数

alpha, gamma = 0.25, 2
eps = 1e-15

"""网络结构"""
tf.reset_default_graph()
# 占位符
inputs = tf.placeholder(tf.float32, [None, input_c])
labels = tf.placeholder(tf.int32, [None, ])

w1 = tf.get_variable('w1', shape=[input_c, c_w1], initializer=tf.initializers.ones)
w2 = tf.get_variable('w2', shape=[c_w1, c_w2], initializer=tf.initializers.ones)
w_output = tf.get_variable('w_output', shape=[c_w2, output_c], initializer=tf.initializers.ones)

_x = tf.matmul(inputs, w1)  # [None, c_w1]
_x = tf.matmul(_x, w2)  # [None, c_w2]
# 输出得到概率
logits = tf.matmul(_x, w_output)  # [None, 1]
logtis = tf.squeeze(logits)  # [None]
# 转化成概率
p = tf.nn.sigmoid(logits)  # [None]
one_p = tf.ones_like(p, dtype=p.dtype) - p
p = tf.maximum(p, eps)
one_p = tf.maximum(one_p, eps)
# 统一取对数
log_p = tf.log(p)
log_one_p = tf.log(one_p)

predict = tf.where(p > 0.5, tf.ones_like(p), tf.zeros_like(p))

opt = tf.train.GradientDescentOptimizer(lr)
init_op = tf.global_variables_initializer()

"""不同方式的损失"""

# 方式1：使用掩码gather
pos_inds = tf.reshape(tf.where(labels > 0), [-1])  # 正样本下标
neg_inds = tf.reshape(tf.where(labels < 1), [-1])  # 负样本下标
# 正样本概率
_pos_log_p = tf.gather(log_p, pos_inds)
_one_pos_p = tf.gather(one_p, pos_inds)
# 负样本概率
_neg_log_p = tf.gather(log_one_p, neg_inds)
_one_neg_p = tf.gather(p, neg_inds)
# 求损失
_pos_loss1 = tf.reduce_sum(alpha * _one_pos_p ** gamma * _pos_log_p)
_pos_loss2 = tf.reduce_sum(alpha * _one_neg_p ** gamma * _neg_log_p)
loss1 = -(_pos_loss1 + _pos_loss2)
train1 = opt.minimize(loss1)

# 方式2：使用掩码点乘
# 正负样本掩码
_ones = tf.ones_like(labels, dtype=p.dtype)
_zeros = tf.zeros_like(labels, dtype=p.dtype)
_pos_labels = tf.where(labels > 0, _ones, _zeros, )
_neg_labels = tf.where(labels < 1, _ones, _zeros, )
# 损失
_loss2 = alpha * _pos_labels * (one_p ** gamma * log_p) + \
         alpha * _neg_labels * (p ** gamma * log_one_p)
loss2 = -tf.reduce_sum(_loss2)
train2 = opt.minimize(loss2)

losses = [loss1, loss2]
trains = [train1, train2]
num_method = len(losses)

"""生成数据和初始损失对比"""
xs = np.random.normal(size=[nums, input_c])  # [nums, input_c]
ys = np.random.randint(0, 2, size=[nums, ])  # [nums, ]

# 初始损失对比
feed_dict = {inputs: xs, labels: ys}
with tf.Session() as sess:
    sess.run(init_op)
    for i1 in range(num_method - 1):
        for i2 in range(i1 + 1, num_method):
            _lo1, _lo2 = sess.run([losses[i1], losses[i2]], feed_dict)
            print(f'lo{i1 + 1}-lo{i2 + 1}={np.abs(_lo1 - _lo2)}.')

"""循环训练"""

w1s, w2s, w_outputs = [0] * num_method, [0] * num_method, [0] * num_method
preds, los = [0] * num_method, [0] * num_method

run_list = [w1, w2, w_output, predict]
feed_dict = {inputs: xs, labels: ys}
with tf.Session() as sess:
    for ind in range(num_method):
        train_op, loss_op = trains[ind], losses[ind]
        sess.run(init_op)
        for ind_epoch in range(num_epoch):
            sess.run(train_op, feed_dict)
        _res = sess.run(run_list + [loss_op], feed_dict)
        w1s[ind], w2s[ind], w_outputs[ind], preds[ind], los[ind] = _res

"""对比不同方式训练的结果"""
# 权重
for i1 in range(num_method-1):
    for i2 in range(i1+1, num_method):
        _w1, _w2 = w1s[i1], w1s[i2]
        print(f'w1 m{i1+1}-m{i2+1}={np.sum(np.abs(_w1-_w2))}')
for i1 in range(num_method-1):
    for i2 in range(i1+1, num_method):
        _w1, _w2 = w2s[i1], w2s[i2]
        print(f'w2 m{i1+1}-m{i2+1}={np.sum(np.abs(_w1-_w2))}')
for i1 in range(num_method-1):
    for i2 in range(i1+1, num_method):
        _w1, _w2 = w_outputs[i1], w_outputs[i2]
        print(f'w_output m{i1+1}-m{i2+1}={np.sum(np.abs(_w1-_w2))}')
# 损失
for i1 in range(num_method-1):
    for i2 in range(i1+1, num_method):
        _lo1, _lo2 = los[i1], los[i2]
        print(f'loss m{i1+1}-m{i2+1}={np.sum(np.abs(_lo1-_lo2))}')


# 预测
for i1 in range(num_method-1):
    for i2 in range(i1+1, num_method):
        _pred1, _pred2 = preds[i1], preds[i2]
        print(f'predict m{i1+1}-m{i2+1}={np.sum(_pred1!=_pred2)}')


# 画出预测分布图
