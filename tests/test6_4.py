"""
对arcface参数调试

ce/fl对p求导，发现前期变化大，后期变化小
所以夹角对p的影响，也是前期变化大，后期变化小？

"""
import numpy as np
import matplotlib.pyplot as plt

"""ce/fl对p求导"""


def fl(alpha, gamma, p):
    p = np.maximum(p, 1e-9)
    return -alpha * (1 - p) ** gamma * np.log(p), -alpha * (
            1 / p * (1 - p) ** gamma - gamma * (1 - p) ** (gamma - 1) * np.log(p))


def ce(p):
    p = np.maximum(p, 1e-9)
    return -np.log(p), -1 / p


p = np.linspace(0.05, 1, 1000)
cey, dcey = ce(p)

fly, dfly = [], []
abs = [[0.25, 2], [0.5, 2], [0.75, 2], [1, 2], ]
for a, b in abs:
    _y, _dy = fl(a, b, p)
    fly.append(_y)
    dfly.append(_dy)

for i in range(len(abs)):
    plt.plot(p, dfly[i], label=f'ab={abs[i]}')
plt.plot(p, dcey, label='ce')
plt.title('dy')
plt.legend()
plt.grid()
plt.show()

"""
对于dy，前期变化率大，后期变化率小，
所以希望p随夹角，前期变化率小，后期变化率大？
或者p随夹角前期变化率大，后期变化率小？

根据导数曲线，希望p=0.1时，夹角在60附近
"""


def arcface_p(radians, ucos, m1, m2, m3, s):
    ulogits = np.exp(s * ucos)
    cos = np.cos(m1 * radians + m2) - m3
    logits = np.exp(s * cos)
    p = logits / (logits + np.sum(ulogits))
    up = np.max(ulogits) / (logits + np.sum(ulogits))
    return p, up


ucos = np.cos(np.random.randint(80, 100, 283) / 180 * np.pi)
radians = np.linspace(0, np.pi, 1000)
m1, m2, m3, s = 1, 0, 0, 1

_s = [16, 24, 32, 64]
for s in _s:
    _p, _up = arcface_p(radians, ucos, m1, m2, m3, s)
    plt.plot(radians * 180 / np.pi, _p, label=f'm123={[m1, m2, m3, s]}')
plt.title('arcface_p s')
plt.xlim([0, 110])
plt.xticks(np.arange(0, 110, 10))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.legend()
plt.grid()
plt.show()
s = 16

_m2 = [-0.3, -0.15, 0, 0.05, 0.1, 0.15, 0.2, 0.3]
for m2 in _m2:
    _p, _up = arcface_p(radians, ucos, m1, m2, m3, s)
    plt.plot(radians * 180 / np.pi, _p, label=f'm123={[m1, m2, m3, s]}')
plt.title('arcface_p m2')
plt.xlim([0, 110])
plt.xticks(np.arange(0, 110, 10))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.legend()
plt.grid()
plt.show()
m2 = 0.1

_m3 = [-0.3, -0.15, 0, 0.05, 0.1, 0.15, 0.2, 0.3]
for m3 in _m3:
    _p, _up = arcface_p(radians, ucos, m1, m2, m3, s)
    plt.plot(radians * 180 / np.pi, _p, label=f'm123={[m1, m2, m3, s]}')
plt.title('arcface_p m3')
plt.xlim([0, 110])
plt.xticks(np.arange(0, 110, 10))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.legend()
plt.grid()
plt.show()
m3 = 0.2

_m1 = [0.7, 0.8, 0.9, 1, 1.05, 1.1, 1.15, 1.2]
for m1 in _m1:
    _p, _up = arcface_p(radians, ucos, m1, m2, m3, s)
    plt.plot(radians * 180 / np.pi, _p, label=f'm123={[m1, m2, m3, s]}')
plt.title('arcface_p m1')
plt.xlim([0, 110])
plt.xticks(np.arange(0, 110, 10))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.legend()
plt.grid()
plt.show()
m1 = 0.75


def plot_compare(_s, _m1, _m2, _m3):
    ucos = np.cos(np.random.randint(80, 100, 283) / 180 * np.pi)
    radians = np.linspace(0, np.pi, 1000)

    s, m1, m2, m3 = 64, 1, 0.3, 0.2
    _p, _up = arcface_p(radians, ucos, m1, m2, m3, s)
    plt.plot(radians * 180 / np.pi, _p, label=f'p - m123={[s, m1, m2, m3]}', color='r')
    plt.plot(radians * 180 / np.pi, _up, label=f'up - m123={[s, m1, m2, m3]}', color='r')

    s, m1, m2, m3 = _s, _m1, _m2, _m3
    _p, _up = arcface_p(radians, ucos, m1, m2, m3, s)
    plt.plot(radians * 180 / np.pi, _p, label=f'p - m123={[s, m1, m2, m3]}', color='b')
    plt.plot(radians * 180 / np.pi, _up, label=f'up - m123={[s, m1, m2, m3]}', color='b')
    plt.title('arcface_p')
    plt.xlim([0, 110])
    plt.xticks(np.arange(0, 110, 10))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.legend()
    plt.grid()
    plt.show()


plot_compare(24, 1, 0.3, 0.2)

plot_compare(18, 0.8, 0.1, 0.2)

plot_compare(16, 1.1, 0.15, 0.15)

plot_compare(64, 1, 0.2, 0.2)

ucos = np.cos(np.random.randint(80, 100, 283) / 180 * np.pi)
radians = np.linspace(0, np.pi, 1000)
m1, m2, m3, s = 1, 0, 0, 64
_p, _up = arcface_p(radians, ucos, m1, m2, m3, s)
plt.plot(radians * 180 / np.pi, _p, label=f'p - m123={[m1, m2, m3, s]}', color='k')
plt.plot(radians * 180 / np.pi, _up, label=f'up - m123={[m1, m2, m3, s]}', color='k')
plt.title('arcface_p')
plt.xlim([0, 110])
plt.xticks(np.arange(0, 110, 10))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.legend()
plt.grid()
plt.show()

"""
从ce反推至θ的残差
"""

import numpy as np
import matplotlib.pyplot as plt


def delta_theta(radians, ucos, s, m1, m2, m3):
    # 不同情况下的cos
    cos1 = np.cos(radians)
    cos2 = np.cos(radians) - m3
    cos3 = np.cos(m1 * radians + m2) - m3
    # 不同情况下的logits
    logits1 = np.exp(s * cos1)
    logits2 = np.exp(s * cos2)
    logits3 = np.exp(s * cos3)
    ulogits = np.exp(s * ucos)
    # 不同情况下的p
    p1 = logits1 / (logits1 + np.sum(ulogits))
    up1 = np.max(ulogits) / (logits1 + np.sum(ulogits))
    p2 = logits2 / (logits2 + np.sum(ulogits))
    up2 = np.max(ulogits) / (logits2 + np.sum(ulogits))
    p3 = logits3 / (logits3 + np.sum(ulogits))
    up3 = np.max(ulogits) / (logits3 + np.sum(ulogits))

    # 不同情况下反推至θ的残差
    dt1 = np.sin(radians) * s * (1 - p1)
    dt2 = np.sin(radians) * s * (1 - p2)
    dt3 = m1 * np.sin(m1 * radians + m2) * s * (1 - p3)

    return p1, up1, dt1, p2, up2, dt2, p3, up3, dt3


def plot_pup(radians, ucos, s=64, m1=1, m2=0.3, m3=0.2, plot=True):
    p1, up1, dt1, p2, up2, dt2, p3, up3, dt3 = delta_theta(radians, ucos, s, m1, m2, m3)

    plt.plot(radians * 180 / np.pi, p1, label=f'p1 - cos(θ)', color='r')
    plt.plot(radians * 180 / np.pi, up1, label=f'up1 - cos(θ)', color='r')
    plt.plot(radians * 180 / np.pi, p2, label=f'p2 - cos(θ)-m3', color='b')
    plt.plot(radians * 180 / np.pi, up2, label=f'up2 - cos(θ)-m3', color='b')
    plt.plot(radians * 180 / np.pi, p3, label=f'p3 - cos(m1*θ+m2)-m3', color='g')
    plt.plot(radians * 180 / np.pi, up3, label=f'up3 - cos(m1*θ+m2)-m3', color='g')
    plt.xlim(0, 110)
    plt.xticks(np.arange(0, 110, 10))
    plt.legend()
    plt.grid()
    plt.title(f'm123 = {[s, m1, m2, m3]}')
    plt.show()


def plot_dt(radians, ucos, s=64, m1=1, m2=0.3, m3=0.2, plot=True):
    p1, up1, dt1, p2, up2, dt2, p3, up3, dt3 = delta_theta(radians, ucos, s, m1, m2, m3)

    x = 90 - m2 * 180 / np.pi
    x = [x, x]
    plt.plot(x, [0, 70], '--', color='y')
    plt.plot([90, 90], [0, 70], '--', color='y')

    angles = radians * 180 / np.pi
    plt.plot(angles, dt1, label=f'dt1 - cos(θ)', color='r')
    plt.plot(angles, dt2, label=f'dt2 - cos(θ)-m3', color='b')
    plt.plot(angles, dt3, label=f'dt3 - cos(m1*θ+m2)-m3', color='g')
    plt.xlim(0, 180)
    plt.xticks(np.arange(0, 190, 10))
    plt.ylim(0, 70)
    plt.yticks(np.arange(0, 80, 10))
    plt.legend()
    plt.grid()
    plt.xlabel('θ')
    plt.ylabel('δθ')
    plt.title(f'm123 = {[s, m1, m2, m3]}')
    plt.show()


radians = np.linspace(0, np.pi, 1000)
ucos = np.cos(np.random.randint(80, 100, 283) / 180 * np.pi)
plot_dt(radians, ucos)
plot_dt(radians, ucos, m2=0, m3=0)
plot_dt(radians, ucos, m2=0, m3=0.2)
plot_dt(radians, ucos, m2=0.2, m3=0.2)
plot_dt(radians, ucos, m2=0.3, m3=0.2)
plot_dt(radians, ucos, m2=0.3, m3=0.3)
plot_dt(radians, ucos, m2=0.3, m3=0)
plot_dt(radians, ucos, m1=0.8, m2=0, m3=0)
plot_dt(radians, ucos, m1=1, m2=0, m3=0)
plot_dt(radians, ucos, m1=1.2, m2=0, m3=0)

# 结论：
# m3只挪动θ较小的部分。因为m3本身不影响求导，只影响saftmax的概率，当theta较大时，其softmax概率趋向于稳定，所以不受影响。
# m2代表整体的残差曲线的平移
# m1会影响整个曲线的放缩，但基本不会影响残差随θ减小，从最大降低的斜率。


plot_dt(radians, ucos)
plot_dt(radians, ucos, m1=1, m2=0, m3=0)
plot_dt(radians, ucos, s=64, m1=1, m2=0, m3=0)
plot_dt(radians, ucos, s=32, m1=1, m2=0, m3=0)
plot_dt(radians, ucos, s=24, m1=1, m2=0, m3=0)
plot_dt(radians, ucos, s=16, m1=1, m2=0, m3=0)
plot_dt(radians, ucos, s=8, m1=1, m2=0, m3=0)

plot_dt(radians, ucos, s=16, m1=1.5, m2=-0.3, m3=0)
plot_dt(radians, ucos, s=18, m1=1.2, m2=0.1, m3=0.2)

