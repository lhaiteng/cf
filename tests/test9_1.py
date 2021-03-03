# -*- coding: utf-8 -*-
"""
对比二分类focal loss，包含正负样本+背景

方式1：分别获取正负样本下标+gather
方式2：分别获取正负样本掩码+点乘+reduce_sum

=> 前向传播的loss相差1e-3，但训练和预测结果一致。

"""
import numpy as np
import matplotlib.pyplot as plt
import os, math, random
import tensorflow as tf


"""参数"""
nums = 1000  # 锚框总数
num_cls = 18  # 样本分类
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
# names = tf.placeholder(tf.int32, [None, ])
# labels_onehot = tf.one_hot(names, num_cls, dtype=inputs.dtype)
input_mask = tf.placeholder(tf.int32, [None, ])
# 权重
w1 = tf.get_variable('w1', shape=[input_c, c_w1], initializer=tf.initializers.ones)
w2 = tf.get_variable('w2', shape=[c_w1, c_w2], initializer=tf.initializers.ones)
w_output = tf.get_variable('w_output', shape=[c_w2, output_c], initializer=tf.initializers.ones)
# 结构
_x = tf.matmul(inputs, w1)  # [None, c_w1]
_x = tf.matmul(_x, w2)  # [None, c_w2]
_x = tf.matmul(_x, w_output)  # [None, 1]
logits = tf.squeeze(_x)  # [None, ]
# 概率
p = tf.nn.sigmoid(logits)
one_p = 1. - p
p = tf.maximum(p, eps)
one_p = tf.maximum(one_p, eps)

# log(p)
log_p = tf.log(p)
log_one_p = tf.log(one_p)

# 预测结果
ones = tf.ones_like(logits)
zeros = tf.zeros_like(logits)
predict = tf.where(logits>0.5, ones, zeros)

opt = tf.train.GradientDescentOptimizer(lr)
init_op = tf.global_variables_initializer()


"""不同方式的损失"""

# 方式1：使用下标+gather
_zeros_mask = tf.zeros_like(input_mask, dtype=input_mask.dtype)
pos_inds = tf.reshape(tf.where(tf.greater(input_mask, _zeros_mask)), [-1])  # 正样本下标
neg_inds = tf.reshape(tf.where(tf.equal(input_mask, _zeros_mask)), [-1])  # 负样本下标
# 正样本损失
_pos_log_p = tf.gather(log_p, pos_inds)
_one_pos_p = tf.gather(one_p, pos_inds)
# 负样本概率
_neg_log_p = tf.gather(log_one_p, neg_inds)
_one_neg_p = tf.gather(p, neg_inds)
# 求损失
_pos_loss1 = tf.reduce_sum(alpha * _one_pos_p ** gamma * _pos_log_p)
_neg_loss1 = tf.reduce_sum(alpha * _one_neg_p ** gamma * _neg_log_p)
loss1 = -(_pos_loss1 + _neg_loss1)
train1 = opt.minimize(loss1)

# 方式2：使用掩码点乘
# 正负样本掩码
_ones = tf.ones_like(input_mask, dtype=p.dtype)
_zeros = tf.zeros_like(input_mask, dtype=p.dtype)
_pos_mask = tf.where(tf.greater(input_mask, _zeros_mask), _ones, _zeros)
_neg_mask = tf.where(tf.equal(input_mask, _zeros_mask), _ones, _zeros)
# 损失
_loss2 = alpha * _pos_mask * (one_p ** gamma * log_p) + \
         alpha * _neg_mask * (p ** gamma * log_one_p)
loss2 = -tf.reduce_sum(_loss2)
train2 = opt.minimize(loss2)

losses = [loss1, loss2]
trains = [train1, train2]
num_method = len(losses)


"""生成数据和初始损失对比"""
xs = np.random.normal(size=[nums, input_c])  # [nums, input_c]
ys = np.random.randint(-1, 2, size=[nums, ])  # [nums, ]

# 初始损失对比
feed_dict = {inputs: xs, input_mask: ys}
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
feed_dict = {inputs: xs, input_mask: ys}
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
for i1 in range(num_method - 1):
    for i2 in range(i1 + 1, num_method):
        _w1, _w2 = w1s[i1], w1s[i2]
        print(f'w1 m{i1 + 1}-m{i2 + 1}={np.sum(np.abs(_w1 - _w2))}')
for i1 in range(num_method - 1):
    for i2 in range(i1 + 1, num_method):
        _w1, _w2 = w2s[i1], w2s[i2]
        print(f'w2 m{i1 + 1}-m{i2 + 1}={np.sum(np.abs(_w1 - _w2))}')
for i1 in range(num_method - 1):
    for i2 in range(i1 + 1, num_method):
        _w1, _w2 = w_outputs[i1], w_outputs[i2]
        print(f'w_output m{i1 + 1}-m{i2 + 1}={np.sum(np.abs(_w1 - _w2))}')
# 损失
for i1 in range(num_method - 1):
    for i2 in range(i1 + 1, num_method):
        _lo1, _lo2 = los[i1], los[i2]
        print(f'loss m{i1 + 1}-m{i2 + 1}={np.sum(np.abs(_lo1 - _lo2))}')

# 预测
for i1 in range(num_method - 1):
    for i2 in range(i1 + 1, num_method):
        _pred1, _pred2 = preds[i1], preds[i2]
        print(f'predict m{i1 + 1}-m{i2 + 1}={np.sum(np.abs(_pred1 - _pred2))}')



