# -*- coding: utf-8 -*-
"""
测试，在tensorflow中，对掩码[0, 1, 1, 0, 0, ..]，使用gather提取1计算损失，与直接点乘计算损失，结果是否相同。
例如：目标检测中，筛选正样本进行回归损失计算。

方式1：利用掩码下标索引进行计算
方式2：利用掩码下标索引，提取事先计算完成的loss进行计算
方式3：利用掩码点乘计算。

方式4：使用掩码得到正负样本两部分掩码，分别与loss点乘后求和，直接求全部的loss进行对比。
方式5：直接求全部的loss

=> 1、2、3前向后向传播都相同
=> 4、5前向后向传播都相同
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

"""网络结构"""

tf.reset_default_graph()
# 参数
input_c = 20  # 输入数据维度
num_cls = 18
output_c = 4  # 输出数据维度
nums = 1000  # 总锚框数
num_epoch = 100  # 训练批次
lr = 0.0001
eps = 1e-15
n_w1, n_w2 = 50, 50
# 占位符
inputs = tf.placeholder(tf.float32, [None, input_c])
labels = tf.placeholder(tf.int32, [None, ])
labels_onehot = tf.one_hot(labels, num_cls)
input_mask = tf.placeholder(tf.float32, [None])  # 掩码
# 网络结构
init_w1 = np.random.normal(0, 0.2, size=[input_c, n_w1])
w1 = tf.get_variable('w1', shape=[input_c, n_w1], dtype=tf.float32,
                     initializer=tf.constant_initializer(init_w1))
logits = tf.matmul(inputs, w1)  # [None, 100]
init_w2 = np.random.normal(0, 0.2, size=[n_w2, num_cls])
w2 = tf.get_variable('w2', shape=[n_w2, num_cls], dtype=tf.float32,
                     initializer=tf.constant_initializer(init_w2))
logits = tf.matmul(logits, w2)  # [None, num_cls]

loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels_onehot, logits)  # [None, ]
predict = tf.argmax(logits, axis=1)

opt = tf.train.GradientDescentOptimizer(lr)
init_op = tf.initializers.global_variables()

"""不同的损失计算方式"""

# 方式1：利用掩码下标索引进行计算
_inds = tf.reshape(tf.where(tf.greater(input_mask, 0)), [-1])
_logits = tf.gather(logits, _inds)
_labels = tf.gather(labels_onehot, _inds)
_loss1 = tf.nn.softmax_cross_entropy_with_logits_v2(_labels, _logits)
loss1  =tf.reduce_sum(_loss1)
train1 = opt.minimize(loss1)

# 方式2：利用掩码下标索引，提取事先计算完成的loss进行计算
_inds = tf.reshape(tf.where(tf.greater(input_mask, 0)), [-1])
_loss2 = tf.gather(loss, _inds)
loss2  =tf.reduce_sum(_loss2)
train2 = opt.minimize(loss2)


# 方式3：利用掩码点乘计算。输入的掩码仅作为标识，类型是int。额外生成ones和zeros获取掩码。
_loss3 = input_mask * loss
loss3 = tf.reduce_sum(_loss3)
train3 = opt.minimize(loss3)

# 方式4：使用掩码得到正负样本两部分掩码，分别与loss点乘后求和，直接求全部的loss进行对比。
_loss4 = input_mask * loss + (1.-input_mask) * loss
loss4 = tf.reduce_sum(_loss4)
train4 = opt.minimize(loss4)

# 方式5：直接求全部的loss
loss5 = tf.reduce_sum(loss)
train5 = opt.minimize(loss5)


losses = [loss1, loss2, loss3, loss4, loss5]
trains = [train1, train2, train3,train4, train5]
num_method = len(trains)

"""生成数据"""
# 生成掩码
mask = np.random.randint(0, 2, size=[nums])
# 生成标签相关
ys = np.random.randint(0, num_cls, size=[nums])
# 根据gt生成输入
xs = np.array([np.random.normal(loc=n - num_cls / 2, scale=0.02, size=[input_c]) for n in ys])

print(f'xs.shape = {xs.shape}, ys.shape = {ys.shape}')
print('mask.sum =', mask.sum())

"""对比初始损失是否相同"""

# 因为使用的相同的网络，pred在初始时候应该一样

feed_dict = {inputs: xs, labels: ys, input_mask: mask}

# with tf.Session() as sess:
#     sess.run(init_op)
#     _lo1, _lo2, _lo3 = sess.run([_loss1, _loss2, _loss3], feed_dict)
# print(f'lo3.max={np.max(_lo3)} lo3.min={np.min(_lo3)}')

print("各方法的初始损失对比")
with tf.Session() as sess:
    sess.run(init_op)
    for i1 in range(num_method - 1):
        for i2 in range(i1 + 1, num_method):
            lo1, lo2 = sess.run([losses[i1], losses[i2]], feed_dict)
            print(f'lo{i1 + 1}-lo{i2 + 1}={np.abs(lo1 - lo2)}')
            print('-' * 100)

"""循环几种方式进行训练"""
w1s = []
w2s = []
preds = []
feed_dict={inputs: xs, labels: ys, input_mask: mask}
for i in range(num_method):
    with tf.Session() as sess:
        sess.run(init_op)
        for ind_epoch in range(num_epoch):
            sess.run(trains[i], feed_dict)
        _w1, _w2, _pred = sess.run([w1, w2, predict], feed_dict)
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
        res = np.sum(preds[i1] != preds[i2])
        print(f'pred method{i1 + 1}-method{i2 + 1} = {res}')
print('-' * 100)




