"""
focal不同alpha、gamma
"""
import numpy as np
import matplotlib.pyplot as plt


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

# for i in range(len(abs)):
#     plt.plot(p, fly[i], label=f'ab={abs[i]}')
# plt.plot(p, cey, label='ce')
# plt.title('y')
# plt.legend()
# plt.show()

for i in range(len(abs)):
    plt.plot(p, dfly[i], label=f'ab={abs[i]}')
plt.plot(p, dcey, label='ce')
plt.title('dy')
plt.legend()
plt.xticks(np.arange(0, 1, 0.1))
plt.grid()
plt.show()

cos = np.cos(60 / 180 * np.pi+0.3)-0.2
logits = np.exp(64 * cos)
cos = np.cos(np.random.randint(70, 110, 283) / 180 * np.pi)
ulogits = np.exp(64 * cos)
arcface_p = logits / (logits + np.sum(ulogits))
arcface_p
arcface_up = np.max(ulogits) / (logits + np.sum(ulogits))
arcface_up


