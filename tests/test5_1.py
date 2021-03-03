# -*- coding: utf-8 -*-
"""
验证自动计算的交叉熵，和为了计算focal loss手动计算的交叉熵。
=> 交叉熵结果相同
=> 在test5_2.py中，验证反向传播得到的结果是否相同
"""
import numpy as np
import cv2, os, math, random
import tensorflow as tf

alpha = 1
gamma = 2

x = tf.placeholder(tf.float32, [5, 10])
y = tf.placeholder(tf.int32, [5])
labels_onehot = tf.one_hot(y, 10)

# tf的softmax交叉熵
ce = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_onehot, logits=x)

# 自己计算softmax交叉熵
p1 = tf.nn.softmax(x)
softmax = tf.reshape(p1, [-1])
labels = tf.range(0, tf.shape(x)[0]) * x.shape[1] + y
p2 = tf.gather(softmax, labels)
CrossEntropyLoss = -tf.log(p2)
fl = alpha * (1 - p2) ** gamma * CrossEntropyLoss
loss = tf.reduce_mean(fl)

inputs, labels = np.random.normal(size=[5, 10]), np.random.randint(0, 10, size=[5])
print(inputs)
print(labels)
with tf.Session() as sess:
    run_list = [ce, p1, p2, CrossEntropyLoss, fl]
    feed_dict = {x: inputs, y:labels}
    out_ce, out_p1, out_p2, out_CrossEntropyLoss, out_fl = sess.run(run_list, feed_dict=feed_dict)
print(out_ce)
print(out_CrossEntropyLoss)  # 与out_ce相同
print(out_p1)
print(out_p2)
print(out_fl)
