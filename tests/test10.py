# -*- coding: utf-8 -*-
"""
时间消耗对比
"""
import time
import numpy as np
import tensorflow as tf


def time_calc(func):
    def wrapper(*args, **kwargs):
        name = kwargs.get('text', func.__name__)
        print(f'start {name}.')
        t = time.time()
        f = func(*args, **kwargs)
        print(f'cost time: {time.time() - t:.3f}')
        return f

    return wrapper


"""
单标签二分类中，计算focal loss。多标签二分类无法使用使用gather，因为需要从两个维度提取索引，如果reshape肯定很耗时间
方式1：使用gather
方式2：使用掩码点乘 
"""


@time_calc
def get_focal_loss(num_epoch, init_op, train_op, loss_op, feed_dict, text='focal_loss'):
    with tf.Session() as sess:
        sess.run(init_op)
        for ind in range(num_epoch):
            sess.run(train_op, feed_dict)
            print(f'\rEPOCH {ind}/{num_epoch}...', end='')
        loss = sess.run(loss_op, feed_dict)
    return loss

"""
单标签二分类中，计算focal loss。多标签二分类无法使用使用gather，因为需要从两个维度提取索引，如果reshape肯定很耗时间
方式1：使用gather
方式2：使用掩码点乘 
"""
# 参数
nums = 500
num_cls = 100
num_epoch = 1000
input_c = 23
output_c = num_cls
c_w1 = 21
c_w2 = 1313
c_w3 = 222
lr = 0.001


tf.reset_default_graph()
# 占位符
inputs = tf.placeholder(tf.float32, [None, input_c])
labels = tf.placeholder(tf.float32, [None, ])  # {0, 1}

w_init = tf.initializers.ones()
b_init = tf.initializers.zeros()
x = tf.layers.dense(inputs, c_w1, kernel_initializer=w_init, bias_initializer=b_init)
x = tf.nn.relu(x)
x = tf.layers.dense(x, c_w2, kernel_initializer=w_init, bias_initializer=b_init)
x = tf.nn.relu(x)
x = tf.layers.dense(x, c_w2, kernel_initializer=w_init, bias_initializer=b_init)
x = tf.nn.relu(x)

logits = tf.layers.dense(x, 1, kernel_initializer=w_init, bias_initializer=b_init)
logits = tf.squeeze(logits)

p = tf.nn.sigmoid(logits)
one_p = 1. - p
p = tf.maximum(p, 1e-15)
one_p = tf.maximum(one_p, 1e-15)

ones = tf.ones_like(labels, dtype=labels.dtype)
zeros = tf.zeros_like(labels, dtype=labels.dtype)

loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
print(loss)

opt = tf.train.GradientDescentOptimizer(lr)
init_op = tf.global_variables_initializer()

# 方式1：gather
pos_inds = tf.reshape(tf.where(tf.greater(labels, ones)), [-1])
neg_inds = tf.reshape(tf.where(tf.equal(labels, ones)), [-1])
pos_corr = tf.gather(one_p, pos_inds)
neg_corr = tf.gather(p, neg_inds)
pos_loss = tf.gather(loss, pos_inds)
neg_loss = tf.gather(loss, neg_inds)
loss1 = tf.reduce_sum(pos_corr ** 2 * pos_loss) + tf.reduce_sum(neg_corr ** 2 * neg_loss)
train1 = opt.minimize(loss1)

# 方式2：掩码点乘
pos_corr = tf.where(tf.greater(labels, ones), one_p, zeros)
neg_corr = tf.where(tf.equal(labels, ones), p, zeros)
loss2 = tf.reduce_sum(pos_corr ** 2 * loss + neg_corr ** 2 * loss)
train2 = opt.minimize(loss2)


# 生成数据运行
xs = np.random.normal(size=[nums, input_c])
ys = np.random.randint(0, 2, size=[nums])
feed_dict = {inputs: xs, labels: ys}
lo1 = get_focal_loss(num_epoch, init_op, train1, loss1, feed_dict, text='method1 focal loss')
lo2 = get_focal_loss(num_epoch, init_op, train1, loss1, feed_dict, text='method2 focal loss')

# start method1 focal loss.
# EPOCH 999/1000...cost time: 38.176
# start method2 focal loss.
# EPOCH 999/1000...cost time: 39.974
# lo1-lo2 = 0.0
