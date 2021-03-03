# -*- coding: utf-8 -*-
"""
如何使用plot保存图片。保存的与plt画出的一致，类似截图。
=> 一定要加上plt.close()
"""
import os, time, sys, cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def time_calc(text):
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(f'开始运行 <{text}> 并计算时间。')
            t = time.time()
            f = func(*args, **kwargs)
            print(f'运行时间 <{text}> : {time.time() - t:.3f} s.')
            return f

        return wrapper

    return decorator


file = 'E:/TEST/AI/datasets/test_face1/{name}.png'
img1 = cv2.imread(file.format(name='bqb1'))
img2 = cv2.imread(file.format(name='bqb2'))
img3 = cv2.imread(file.format(name='bqb3'))
imgs = [img1, img2, img3]

plot = False
save_file = './tests/gen_pics/no_{name}.png'

for i in range(10):
    plt.figure(figsize=[30, 10])
    for ind, img in enumerate(imgs):
        plt.subplot(1, 3, ind + 1)
        plt.imshow(img[:, :, ::-1])
        plt.title(f'label {ind}', y=-0.1)
        plt.axis('off')
    plt.suptitle(f'test_{i}')
    if save_file: plt.savefig(save_file.format(name=i))
    if plot: plt.show()
    plt.close()
plt.show()
