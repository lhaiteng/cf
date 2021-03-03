"""
验证公考-资料分析中的现期和差、混合增长率、比重之差的问题
"""
import numpy as np
import cv2
import seaborn as sns
import matplotlib.pyplot as plt

# import tensorflow as tf


# 1和2，现期的和差，与基期的和差相比
n = 100000
A1 = np.random.random(n)
r1 = np.random.random(n)
A2 = np.random.random(n)
r2 = np.random.random(n)

verify1 = np.clip(A1 / (1 + r1) + A2 / (1 + r2) - (A1 + A2), -1, 1)
np.sum(verify1 > 0)
verify2 = np.clip(A1 / (1 + r1) - A2 / (1 + r2) - (A1 - A2), -1, 1)
np.sum(verify2 >= 0)

sns.distplot(verify1)
plt.xlim([-1, 1])
plt.title('verify1')
plt.show()
sns.distplot(verify2)
plt.xlim([-1, 1])
plt.title('verify2')
plt.show()

# 1和2的混合增长率，与平均增长率相比
n = 100000
A1 = np.random.random(n)
r1 = np.random.random(n) * 2 - 1
A2 = np.random.random(n)
r2 = np.random.random(n) * 2 - 1
rmix = (A1 + A2) / (A1 / (1 + r1) + A2 / (1 + r2))
delta_r = rmix - (r1 + r2) / 2
sns.distplot(delta_r)
plt.xlim([-1, 1])
plt.title(f'delta_r-{np.sum(delta_r < 0)}')
plt.show()

# 比重之差<|a-b| => 同时a、b小于0不成立
n = 10000000
A = np.random.random(n)  # 现期部分
ratio = np.random.random(n)  # 现期比重
B = A / ratio  # 现期整体
a = np.random.random(n) * 2 - 1  # 部分增长率
b = np.random.random(n) * 2 - 1  # 整体增长率

ratio_base = A / (1 + a) / (B / (1 + b))  # 基期比重
delta = ratio - ratio_base
res = delta - np.abs(a - b)
np.sum(res < 0)

sns.distplot(res)
plt.xlim([-1, 1])
plt.title(f'res_ratio-{np.sum(res < 0)}')
plt.show()

sns.distplot(a, label='a', kde=False)
sns.distplot(a[res < 0], label='res < 0', kde=False)
plt.xlim([-1, 1])
plt.legend()
plt.show()
sns.distplot(b, label='b', kde=False)
sns.distplot(b[res < 0], label='res < 0', kde=False)
plt.xlim([-1, 1])
plt.legend()
plt.show()
