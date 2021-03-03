# -*- coding: utf-8 -*-
"""
对比不同s、theta下的arcface概率和损失
"""
import matplotlib.pyplot as plt
import numpy as np


"""s和theta的变化影响"""

# s的变化影响
s = np.arange(65)
logits = np.exp((np.cos(0.7 + 0.3) - 0.2) * s)
un_logits = np.exp(np.cos(1.570) * s)

cls = 284

ps = logits / (logits + un_logits * (cls - 1))
_ps = un_logits / (logits + un_logits * (cls - 1))

plt.plot(s, _ps)
plt.title('s-_ps')
plt.show()

plt.plot(s, ps)
plt.title('s-ps')
plt.show()

plt.plot(s, -np.log(ps))
plt.title('s-ce')
plt.show()

fl = (1 - ps) ** 2 * -np.log(ps)
plt.plot(s, fl)
plt.title('s-fl')
plt.show()

# theta变化的影响

s = 32
theta = np.arange(30, 150) / 100
logits = np.exp((np.cos(theta + 0.3) - 0.2) * s)
un_logits = np.exp(np.cos(1.570) * s)

cls = 284

ps = logits / (logits + un_logits * (cls - 1))
_ps = un_logits / (logits + un_logits * (cls - 1))

plt.plot(theta, _ps)
plt.title('thetas-_ps')
plt.show()

plt.plot(theta, ps)
plt.title('thetas-ps')
plt.show()

plt.plot(theta, -np.log(ps))
plt.title('thetas-ce')
plt.show()

plt.plot(theta, (1 - ps) ** 2 * -np.log(ps))
plt.title('thetas-fl')
plt.show()
