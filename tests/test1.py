"""
测试lr曲线只使用1个cycle stage
"""
import cv2, os, math, time, random
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from util.lr_generator import LR_decay


warm_up_stage = 4
cycle_stage = 1  # 1表示一条cos曲线至结束，lr变化较多阶段更平滑
epochs = 100
decay_steps = (epochs - warm_up_stage) // cycle_stage
lr_para = {'start_lr': 0.0001, 'lr': 0.01, 'end_lr': 0.003,
           'warm_up_stage': warm_up_stage,
           'cycle': True,
           'decay_steps': decay_steps,
           'decay_rate': 0.99,
           'power': 0.5}
LR = LR_decay(**lr_para, epochs=100)

x, y = list(range(100)), []
for a in x:
    y.append(LR.get_lr(a, cate='cosine_decay'))
print(y[-1])
plt.plot(x, y, label='cos1')

x, y = list(range(100)), []
for a in x:
    y.append(LR.get_lr(a, cate='cosine_decay2'))
print(y[-1])
plt.plot(x, y, label='cos2')

plt.legend()
plt.show()

