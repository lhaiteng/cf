# -*- coding: utf-8 -*-
"""
不同参数下的focal loss以及求导的结果区别
alhpa作为乘积项，不参与对p求导，视为调节因子，可取1进行对比
使用tensorflow帮助求focal loss或ce loss对p、theta的导数，即theta - p - loss
=> 跟自己写公式的结果类似，p需要+1e-6
"""
import numpy as np
import cv2, os, math, random
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf


def get_focal_loss(p, alpha=1, gamma=2):
    # 根据概率得到focal loss，及损失对p的导数
    loss = -alpha * (1 - p) ** gamma * np.log(p)
    loss_p = alpha * gamma * (1 - p) ** (gamma - 1) * np.log(p) - alpha * ((1 - p) ** gamma) / p
    return loss, loss_p


def get_ce_loss(p):
    loss = -np.log(p)
    loss_p = -1 / p
    return loss, loss_p


def get_arcface_softamx(theta, uthetas, m1, m2, m3, s):
    # 根据标签项和非标签项的夹角，得到arcface的softamx概率，及概率对theta的导数
    x = np.cos(m1 * theta + m2) - m3
    x = np.exp(s * x)
    uxs = np.exp(s * np.cos(uthetas))
    sum_uxs = np.sum(uxs)

    p = x / (x + sum_uxs)
    # p += 1e-6  # 敏感！！
    p_theta = s * m1 * np.sin(m1 * theta + m2) * (p ** 2 - p)
    # p += 1e-6  # 敏感！！
    return p, p_theta


def plot_opt(title):
    plt.legend()
    plt.xlim(0, 90)
    plt.xticks(range(0, 100, 10))
    plt.title(title)


"""不同标签项夹角下的损失、概率以及对应导数"""
n = 284  # 种类
# 非标签项
uangles = np.random.normal(90, 10, size=[n - 1])
uthetas = uangles / 180 * np.pi
ucosines = np.cos(uthetas)
# 非标签项夹角分布
sns.distplot(uangles, rug=True)
plt.title('distribution of unangles')
plt.xlim([0, 180])
plt.show()

# 标签项
m1, m2, m3 = 1, 0.3, 0.2
ss = (8, 16, 32, 64, 128)

# min_angle, max_angle = 30, 90
# num = max_angle - min_angle
# logits = np.arange(min_angle, max_angle)

num = 180
angles = np.sort(np.random.randint(0, 120, size=[num]))

tf_angles = tf.placeholder(tf.float32, shape=[num])
thetas = tf_angles / 180 * math.pi
labels = tf.placeholder(tf.int32, shape=[num])
labels_onehot = tf.one_hot(labels, n)
opt = tf.train.GradientDescentOptimizer(0.01)

ps, fl_losses, ce_losses = [], [], []
fl_thetas, ce_thetas = [], []
fl_ps, ce_ps = [], []

for s in ss:
    x = tf.reshape(s * (tf.cos(m1 * thetas + m2) - m3), [-1, 1])
    uxs = s * ucosines.reshape([1, -1])
    uxs = np.tile(uxs, [num, 1])
    cosines = tf.concat([x, uxs], axis=1)

    softmax = tf.nn.softmax(cosines, axis=1)  # [None, num_cls]
    softmax = tf.reshape(softmax, [-1])  # 拉伸以便gather
    _labels = tf.range(0, num) * n + labels  # 拉伸后的标签位置
    p = tf.gather(softmax, _labels)
    ce_loss = tf.log(p)  # 注意后面统一取负
    fl_loss = 1 * (1. - p) ** 2 * ce_loss
    fl_loss = -tf.reduce_mean(fl_loss)  # 注意取负
    ce_loss = -tf.reduce_mean(ce_loss)
    fl_p = opt.compute_gradients(fl_loss, [p])
    fl_theta = opt.compute_gradients(fl_loss, [thetas])
    ce_p = opt.compute_gradients(ce_loss, [p])
    ce_theta = opt.compute_gradients(ce_loss, [thetas])

    with tf.Session() as sess:
        run_list = [p, fl_loss, ce_loss, fl_theta, ce_theta, fl_p, ce_p]
        _p, _fl_loss, _ce_loss, _fl_theta, _ce_theta, _fl_p, _ce_p = \
            sess.run(run_list, {labels: np.zeros([num]), tf_angles: angles})
        ps.append(_p)
        fl_losses.append(_fl_loss)
        ce_losses.append(_ce_loss)
        fl_thetas.append(_fl_theta[0][0])
        ce_thetas.append(_ce_theta[0][0])
        fl_ps.append(_fl_p[0][0])
        ce_ps.append(_ce_p[0][0])

for ind, s in enumerate(ss):
    plt.plot(angles, ps[ind], label=f's={s}')
plt.legend()
plt.title('p')
plt.show()

for ind, s in enumerate(ss):
    plt.plot(angles, fl_thetas[ind], label=f's={s}')
plt.legend()
plt.title('fl_theta')
plt.show()
for ind, s in enumerate(ss):
    plt.plot(angles, ce_thetas[ind], label=f's={s}')
plt.legend()
plt.title('ce_theta')
plt.show()
for ind, s in enumerate(ss):
    plt.plot(angles, fl_ps[ind], label=f's={s}')
plt.legend()
plt.title('fl_p')
plt.show()
for ind, s in enumerate(ss):
    plt.plot(angles, ce_ps[ind], label=f's={s}')
plt.legend()
plt.title('ce_p')
plt.show()
