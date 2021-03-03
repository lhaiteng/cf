# -*- coding: utf-8 -*-
"""
验证自动计算的交叉熵，和为了计算focal loss手动计算的交叉熵。

计算交叉熵的方式：
方式1：tf内置的softmax交叉熵
方式2：自己计算softamx-使用gather。注意寻找拉伸后的标签位置。
方式3：自己计算softamx-不用gather，乘labels_onehot再reduce_mean
方式4：按照公式一步一步求，乘labels_onehot先reduce_sum，再reduce_mean
方式5：使用把非标签项的概率变为1-p，再reduce_mean所有
方式6：把标签项概率仍为p，加上非标签项概率1-p。即概率是-log(p)-log(1-p)，把多分类转换成了各分类的二分类

对比内容：
权重w, 损失loss, 预测结果pred

=> 前向计算结果：1、2、3、4相同，5、6互不相同
=> 训练结果：2、3、4相同，其余各不相同
=> 意思是提取索引求得的结果，与直接相加求得的结果一致，可能是因为有0的参与。
=> 有人解释：自己计算会产生数值不稳定的问题，需要自己在softmax函数里面加些trick。
=> 所以官方推荐如果使用的loss function是最小化交叉熵，并且，最后一层是要经过softmax函数处理，
=> 则最好使用内置函数，因为它会帮你处理数值不稳定的问题。

"""
import numpy as np
import matplotlib.pyplot as plt
import cv2, os, math, random
import tensorflow as tf

tf.reset_default_graph()
# 参数
num_cls = 100  # 种类数
input_c = 10  # 输入数据维度
nums = 500000  # 总数据
num_epoch = 100  # 训练批次
lr = 0.001
eps = 1e-15
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

# 方式1：tf内置的softmax交叉熵
_loss1 = tf.nn.softmax_cross_entropy_with_logits_v2(labels_onehot, logits)  # [nums, ]
loss1 = tf.reduce_mean(_loss1)
train1 = opt.minimize(loss1)

# 方式2：自己计算softamx-使用gather。注意寻找拉伸后的标签位置。
_p = tf.reshape(p, [-1])
_label_ind = tf.range(0, tf.shape(logits)[0]) * logits.shape[-1] + labels  # 拉伸后的标签位置
_p = tf.gather(_p, _label_ind)
loss2 = -tf.reduce_mean(tf.log(_p))
train2 = opt.minimize(loss2)

# 方式3：自己计算softamx-不用gather
_loss3 = labels_onehot * tf.log(p)
loss3 = -tf.reduce_mean(_loss3) * num_cls
train3 = opt.minimize(loss3)

# 方式4：按照公式一步一步求
_loss4 = labels_onehot * tf.log(p)
_loss4 = tf.reduce_sum(_loss4, axis=1)
loss4 = -tf.reduce_mean(_loss4)
train4 = opt.minimize(loss4)

# 方式5：使用把非标签项的概率变为1-p，再reduce_mean所有
_p = tf.where(labels_onehot > 0.0, p, 1. - p)
_p = tf.maximum(_p, eps)
_loss5 = tf.log(_p)
loss5 = -tf.reduce_mean(_loss5) * num_cls
train5 = opt.minimize(loss5)

# 方式6：把标签项概率仍为p，加上非标签项概率1-p。即概率是-log(p)-log(1-p)，把多分类转换成了各分类的二分类
_p = tf.reshape(p, [-1])
_label_ind = tf.range(0, tf.shape(logits)[0]) * logits.shape[-1] + labels  # 拉伸后的标签位置
_p = tf.gather(_p, _label_ind)
loss6 = -tf.reduce_mean(tf.log(_p) + tf.log(tf.maximum((1. - _p), eps))) * 2
train6 = opt.minimize(loss6)

# 生成数据
ys = np.random.randint(0, num_cls, size=[nums])
xs = np.array([np.random.normal(loc=n, scale=1, size=[input_c]) for n in ys])
print(f'xs.shape = {xs.shape}, ys.shape = {ys.shape}')

"""对比初始损失是否相同"""
# 因为使用的相同的网络，pred在初始时候应该一样
print("各方法的初始损失对比")
with tf.Session() as sess:
    sess.run(init_op)
    losses = [loss1, loss2, loss3, loss4, loss5, loss6]
    num_method = len(losses)
    for i1 in range(num_method - 1):
        for i2 in range(i1 + 1, num_method):
            lo1, lo2 = sess.run([losses[i1], losses[i2]], feed_dict={inputs: xs, labels: ys})
            print(f'lo{i1 + 1}-lo{i2 + 1}={np.abs(lo1 - lo2)}')

"""循环几种方式进行训练"""
trains = [train1, train2, train3, train4, train5, train6]
w1s = []
w2s = []
preds = []

num_method = len(trains)  # 一共有几种方式

for i in range(num_method):
    with tf.Session() as sess:
        sess.run(init_op)
        for ind_epoch in range(num_epoch):
            print(f'\rtrain method{i+1}: {ind_epoch}/{num_epoch}', end='')
            sess.run(trains[i], feed_dict={inputs: xs, labels: ys})
        print('\tfinish.')
        _w1, _w2, _pred = sess.run([w1, w2, predict], feed_dict={inputs: xs})
        w1s.append(_w1)
        w2s.append(_w2)
        preds.append(_pred)

"""对比计算结果"""
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

# 预测的分布
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
