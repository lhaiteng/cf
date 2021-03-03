# -*- coding: utf-8 -*-
"""
*** 新的对比看test5_3.py，这个写错了 ***

对比计算focal loss时，使用tf.gather和tf.reduce_sum得到的p，在优化时是否得到相同结果。
方式1：使用gather
方式2：使用reduce_sum
方式3：仍然使用p*labels_onehot得到相应的样本下标，但使用乘法，不把不同类别相加
对比方式1与方式3的运行时间
=> 方式1与其他不同, 方式2与方式3结果相同，
=> 可见使用乘法进行索引的效果，与reduce_sum相同啊。
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2, os, math, random, time
import tensorflow as tf


def time_calc(func):
    def wrapper(*args, **kwargs):
        print(f'start calculate time of: {func.__name__}.')
        t = time.time()
        f = func(*args, **kwargs)
        print(f'cost time: {time.time() - t:.3f}s.')
        return f

    return wrapper


alpha = 1
gamma = 2

num_cls = 5
x = tf.placeholder(tf.float32, [5, 10])
y = tf.placeholder(tf.int32, [num_cls])
labels_onehot = tf.one_hot(y, num_cls)

# 方式1：使用gather
weight1 = tf.get_variable('weight1', shape=[10, num_cls], initializer=tf.initializers.ones())
logits1 = tf.matmul(x, weight1)  # [5, 5]
pred1 = tf.argmax(logits1, axis=-1)
p11 = tf.nn.softmax(logits1)
p11 = tf.maximum(p11, 1e-9)
softmax1 = tf.reshape(p11, [-1])
labels1 = tf.range(0, tf.shape(x)[0]) * x.shape[1] + y  # 这里写错了！！应该是logits.shape[1]。艹
p12 = tf.gather(softmax1, labels1)
ce1 = tf.log(p12)
fl1 = alpha * (1 - p12) ** gamma * ce1
loss1 = -tf.reduce_mean(fl1)


# 方式2：使用reduce_sum
weight2 = tf.get_variable('weight2', shape=[10, num_cls], initializer=tf.initializers.ones())
logits2 = tf.matmul(x, weight2)  # [5, 5]
pred2 = tf.argmax(logits2, axis=-1)
p21 = tf.nn.softmax(logits2, axis=-1)
p21 = tf.maximum(p21, 1e-9)
p22 = tf.reduce_sum(p21 * labels_onehot, axis=1)
ce2 = tf.log(p22)
fl2 = alpha * (1 - p22) ** gamma * ce2
loss2 = -tf.reduce_mean(fl2)
# loss2 = loss2 / num_cls  # 相比方式1，因为加入了labels_one的多个分类，所以要多除一点

# 方式3：仍然使用p*labels_onehot得到相应的样本下标，但使用提取索引的方式获取，不把不同类别相加
weight3 = tf.get_variable('weight3', shape=[10, num_cls], initializer=tf.initializers.ones())
logits3 = tf.matmul(x, weight3)  # [5, 5]
pred3 = tf.argmax(logits3, axis=-1)
p3 = tf.nn.softmax(logits3, axis=-1)  # [None, num_cls]
p3 = tf.maximum(p3, 1e-9)
# 得到权重
# 大于0的是标签p，否则用1。因为要用1-w作为权重
_p3 = tf.where(tf.greater(labels_onehot, 0), p3, tf.ones_like(p3))
ce3 = tf.log(p3) * labels_onehot
fl3 = alpha * (1 - _p3) ** gamma * ce3
loss3 = -tf.reduce_mean(fl3) * num_cls

train_op1 = tf.train.GradientDescentOptimizer(0.01).minimize(loss1)
train_op2 = tf.train.GradientDescentOptimizer(0.01).minimize(loss2)
train_op3 = tf.train.GradientDescentOptimizer(0.01).minimize(loss3)
init_op = tf.global_variables_initializer()

# 共有5个batch
datas_x = np.random.normal(size=[25, 10])
datas_y = np.random.randint(0, 5, size=[25])

# 计算损失、得到权重、最终预测结果是否相同
with tf.Session() as sess:
    sess.run(init_op)
    for i in range(50):
        no = i % 5
        feed_dict = {x: datas_x[no * 5:no * 5 + 5, :],
                     y: datas_y[no * 5:no * 5 + 5]}
        sess.run([train_op1, train_op2, train_op3], feed_dict=feed_dict)
        lo1, lo2, lo3, pr1, pr2, pr3 = sess.run([loss1, loss2, loss3, pred1, pred2, pred3], feed_dict=feed_dict)
        print(f'i={i} lo1-lo2={lo1 - lo2}\tlo1-lo3={lo1 - lo3}\tlo2-lo3={lo2 - lo3}')
        print(f'pred1-pred2={np.sum(np.abs(pr1 - pr2))}\t'
              f'pred1-pred3={np.sum(np.abs(pr1 - pr3))}\t'
              f'pred2-pred3={np.sum(np.abs(pr2 - pr3))}')
        print('-'*100)
    # 权重
    w1, w2, w3 = sess.run([weight1, weight2, weight3])

print(f'w1 - w2 = {np.sum(np.abs(w1 - w2))}')  # w1 - w2 = 1.8974905014038086
print(f'w1 - w3 = {np.sum(np.abs(w1 - w3))}')  # w1 - w3 = 1.8974905014038086
print(f'w2 - w3 = {np.sum(np.abs(w2 - w3))}')  # w2 - w3 = 0.0

bs = 5
plt.figure(figsize=[15, 5])
plt.subplot(131)
plt.plot(range(bs), pr1, label='l1')
plt.plot(range(bs), pr2, label='l2')
plt.ylim([-1, num_cls+1])
plt.legend()
plt.subplot(132)
plt.plot(range(bs), pr1, label='l1')
plt.plot(range(bs), pr3, label='l3')
plt.ylim([-1, num_cls+1])
plt.legend()
plt.subplot(133)
plt.plot(range(bs), pr2, label='l2')
plt.plot(range(bs), pr3, label='l3')
plt.ylim([-1, num_cls+1])
plt.legend()
plt.suptitle('pred')
plt.show()



