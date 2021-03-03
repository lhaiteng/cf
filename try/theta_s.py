import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

## 设置属性防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

for s in (1, 3, 6, 10):
    for theta in (0, 0.1, 0.3, 0.6, 0.8, 0.9, 1):
        x = np.arange(314) / 100
        y = np.exp(s * np.cos(theta)) / (np.exp(s * np.cos(theta)) + 499 * np.exp(s * np.cos(x)))
        y = -np.log(y)
        plt.plot(x, y, label=f'thetas={theta}')
    plt.xlabel('非标签夹角')
    plt.ylabel('交叉熵损失')
    plt.ylim([0, 12])
    plt.legend()
    plt.grid()
    plt.title(f's={s}')
    plt.show()

# 不同非标签项夹角下，交叉熵损失随标签项夹角的变化
for s1, s2 in ((0, 10), (10, 20), (20, 30)):
    un_theta = 1.57
    for s in range(s1, s2):
        theta = np.arange(314) / 100
        y = np.exp(s * np.cos(theta)) / (np.exp(s * np.cos(theta)) + 499 * np.exp(s * np.cos(un_theta)))
        y = -np.log(y)
        plt.plot(theta, y, label=f's={s}')
    plt.title(f'非标签夹角{un_theta}')
    plt.xlabel('标签项夹角')
    plt.ylabel('交叉熵损失')
    # plt.ylim([0, 30])
    plt.legend()
    plt.grid()
    plt.show()

# 不同扩大系数下的损失
for t1, t2 in ((0, 10), (10, 20), (20, 30)):
    for theta in range(t1, t2):
        theta /= 10
        s = np.arange(31)
        un_theta = 1.57
        y = np.exp(s * np.cos(theta)) / (np.exp(s * np.cos(theta)) + 499 * np.exp(s * np.cos(un_theta)))
        y = -np.log(y)
        plt.plot(s, y, label=f'thetas={theta}')
    plt.title(f'非标签夹角{un_theta}')
    plt.xlabel('扩大系数')
    plt.ylabel('交叉熵损失')
    plt.legend()
    plt.grid()
    plt.show()

""" 
L对z求导 x-1, L对zk求导 ak
l'对z求导 s(x'-1), l'对zk求导 sa'k
现对比两个导数的关系
"""


def get_loss(theta, un_theta=1.57, n=10):
    fenmu = np.exp(np.cos(theta)) + (n - 1) * np.exp(np.cos(un_theta))
    a = np.exp(np.cos(theta)) / fenmu
    return -np.log(a)


def get_loss1(theta, s, un_theta=1.57, n=10):
    fenmu = np.exp(s * np.cos(theta)) + (n - 1) * np.exp(s * np.cos(un_theta))
    a = np.exp(s * np.cos(theta)) / fenmu
    return -np.log(a)


def get_L(theta, un_theta=1.57, n=10):
    fenmu = np.exp(np.cos(theta)) + (n - 1) * np.exp(np.cos(un_theta))
    a = np.exp(np.cos(theta)) / fenmu
    ak = np.exp(np.cos(un_theta)) / fenmu
    Lz = a - 1
    Lzk = ak
    return Lz, Lzk


def get_L1(theta, s, un_theta=1.57, n=500):
    fenmu = np.exp(s * np.cos(theta)) + (n - 1) * np.exp(s * np.cos(un_theta))
    a = np.exp(s * np.cos(theta)) / fenmu
    ak = np.exp(s * np.cos(un_theta)) / fenmu
    Lz = s * (a - 1)
    Lzk = s * ak
    return Lz, Lzk


# 固定标签项夹角和非标签项夹角，改变扩大系数
s = np.arange(40)
theta = 1.7
un_theta = 1.57
Lz, Lzk = get_L(theta, un_theta)
L1z, L1zk = get_L1(theta, s)
plt.plot(s, L1z, label=f'L1z')
plt.plot(s, L1zk, label=f'L1zk')
plt.title(f'thetas={theta} / untheta={un_theta} / Lz={Lz:.3f} / Lzk={Lzk:.3f}')
plt.xlabel('s')
plt.ylabel('Lz / Lzk')
plt.grid()
plt.legend()
plt.show()

# 固定扩大系数和非标签项夹角，改变标签项夹角
s = 30
theta = np.arange(30) / 10
un_theta = 1.57
Lz, Lzk = get_L(theta, un_theta)
L1z, L1zk = get_L1(theta, s)
# plt.plot(thetas, Lz, label=f'Lz-s{s}')
# plt.plot(thetas, Lzk, label=f'Lzk-s{s}')
plt.plot(theta, L1z, label=f'L1z-s{s}')
# plt.plot(thetas, L1zk, label=f'L1zk-s{s}')
plt.title(f'untheta={un_theta}')
plt.xlabel('thetas')
plt.ylabel('L1z')
plt.grid()
plt.legend()
# 把x轴的刻度间隔设置为1，并存在变量里
x_major_locator = plt.MultipleLocator(0.2)
# ax为两条坐标轴的实例
ax = plt.gca()
# 坐标轴应用间隔
ax.xaxis.set_major_locator(x_major_locator)
plt.show()

"""
用等势图表示loss-(thetas, s), L1z-(thetas, s), L1zk-(thetas, s)
"""


def plot_contour(theta, s, y, title, levels=10):
    # 颜色集，6层颜色，默认的情况不用写颜色层数,
    cset = plt.contourf(theta, s, y, levels, cmap=plt.cm.hot)  # or cmap='hot'
    # 画出8条线，并将颜色设置为黑色
    contour = plt.contour(theta, s, y, levels)
    # 等高线上标明z（即高度）的值，字体大小是10，颜色分别是黑色和红色
    plt.clabel(contour, fontsize=10, fmt='%.1f')  # colors=('k','r')
    # 去掉坐标轴刻度
    # plt.xticks(())
    # plt.yticks(())
    # 设置颜色条，（显示在图片右边）
    plt.colorbar(cset)
    # 显示
    plt.title(f'{title} - untheta=1.57')
    plt.xlabel('thetas')
    plt.ylabel('s')
    plt.grid()
    plt.show()


nlevel = 15
# loss1 - thetas, s
theta = np.arange(315) / 100
theta = np.tile(theta, (21, 1))
s = np.arange(21).reshape([-1, 1])
s = np.tile(s, (1, 315))
loss1 = get_loss1(theta, s)
plot_contour(theta, s, loss1, 'loss1', nlevel)

# L1z/L1zk - thetas, s
theta = np.arange(315) / 100
theta = np.tile(theta, (21, 1))
s = np.arange(21).reshape([-1, 1])
s = np.tile(s, (1, 315))
l1z, l1zk = get_L1(theta, s)
plot_contour(theta, s, l1z, 'L1z', nlevel)
plot_contour(theta, s, l1zk, 'L1zk', nlevel)

"""
用等势图表示
loss1-(thetas, m1, m2, m3, s), 
L1z-(thetas, m1, m2, m3, s), 
L1zk-(thetas, m1, m2, m3, s)
"""


def get_loss_m_s(theta, m2, s, n=234, un_theta=1.57):
    fenzi = np.exp(s * (np.cos(theta) - m2))
    fenmu = fenzi + (n - 1) * np.exp(s * np.cos(un_theta))
    a = fenzi / fenmu
    return -np.log(a)


def plot_contour_m_s(theta, s, y, title, levels=10):
    # 颜色集，6层颜色，默认的情况不用写颜色层数,
    cset = plt.contourf(theta, s, y, levels, cmap=plt.cm.hot)  # or cmap='hot'
    # 画出8条线，并将颜色设置为黑色
    contour = plt.contour(theta, s, y, levels)
    # 等高线上标明z（即高度）的值，字体大小是10，颜色分别是黑色和红色
    plt.clabel(contour, fontsize=10, fmt='%.1f')  # colors=('k','r')
    # 去掉坐标轴刻度
    # plt.xticks(())
    # plt.yticks(())
    # 设置颜色条，（显示在图片右边）
    plt.colorbar(cset)
    # 显示
    plt.title(f'{title} - untheta=1.57')
    plt.xlabel('thetas')
    plt.ylabel('s')
    plt.grid()
    plt.show()


nlevel = 15
# loss1 - thetas, s
theta = np.arange(315) / 100
theta = np.tile(theta, (21, 1))
s = np.arange(21).reshape([-1, 1])
s = np.tile(s, (1, 315))
loss = get_loss_m_s(theta, 0.2, s)
plot_contour_m_s(theta, s, loss, 'loss1', nlevel)
