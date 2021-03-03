# -*- coding: utf-8 -*-
"""
不同参数下的focal loss以及求导的结果区别
alhpa作为乘积项，不参与对p求导，视为调节因子，可取1进行对比
从不同夹角到最后的focal loss或ce loss，即theta - p - loss，并最只用对theta求导
=> 对p在求theta前或后+1e-6很敏感。在test62.py中使用tensorflow帮助求导
"""
import numpy as np
import cv2, os, math, random
import matplotlib.pyplot as plt
import seaborn as sns


def focal_loss(p, alpha=1, gamma=2):
    # 根据概率得到focal loss，及损失对p的导数
    loss = -alpha * (1 - p) ** gamma * np.log(p)
    loss_p = alpha * gamma * (1 - p) ** (gamma - 1) * np.log(p) - alpha * ((1 - p) ** gamma) / p
    return loss, loss_p


def ce_loss(p):
    loss = -np.log(p)
    loss_p = -1 / p
    return loss, loss_p


def arcface_softamx(theta, uthetas, m1, m2, m3, s):
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
angles = np.arange(180)
theta = angles / 180 * np.pi
ps, p_thetas, fls, fl_ps, ces, ce_ps, fl_thetas, ce_thetas = [], [], [], [], [], [], [], []
ys = [ps, p_thetas, fls, fl_ps, ces, ce_ps, fl_thetas, ce_thetas]
for s in ss:
    p, p_theta = arcface_softamx(theta, uthetas, m1, m2, m3, s)
    fl, fl_p = focal_loss(p, alpha=1, gamma=3)
    fl_theta = fl_p * p_theta
    ce, ce_p = ce_loss(p)
    ce_theta = ce_p * p_theta
    for ind, y in enumerate([p, p_theta, fl, fl_p, fl_theta, ce, ce_p, ce_theta]):
        ys[ind].append(y)

# 作图
plt.figure(figsize=[10, 5])
plt.subplot(121)
for ind, y in enumerate(ys[0]):
    plt.plot(angles, y, label=f's={ss[ind]}')
plot_opt(title='angle - p')
plt.subplot(122)
for ind, y in enumerate(ys[1]):
    plt.plot(angles, y, label=f's={ss[ind]}')
plot_opt(title='angle - p_theta')
plt.show()

plt.figure(figsize=[15, 5])
plt.subplot(131)
for ind, y in enumerate(ys[2]):
    plt.plot(angles, y, label=f's={ss[ind]}')
plot_opt(title='angle - fl')
plt.subplot(132)
for ind, y in enumerate(ys[3]):
    plt.plot(angles, y, label=f's={ss[ind]}')
plot_opt(title='angle - fl_p')
plt.subplot(133)
for ind, y in enumerate(ys[4]):
    plt.plot(angles, y, label=f's={ss[ind]}')
plot_opt(title='angle - fl_theta')
plt.show()

plt.figure(figsize=[15, 5])
plt.subplot(131)
for ind, y in enumerate(ys[5]):
    plt.plot(angles, y, label=f's={ss[ind]}')
plot_opt(title='angle - ce')
plt.subplot(132)
for ind, y in enumerate(ys[6]):
    plt.plot(angles, y, label=f's={ss[ind]}')
plot_opt(title='angle - ce_p')
plt.subplot(133)
for ind, y in enumerate(ys[7]):
    plt.plot(angles, y, label=f's={ss[ind]}')
plot_opt(title='angle - ce_theta')
plt.show()
