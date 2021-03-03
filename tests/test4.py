# -*- coding: utf-8 -*-
"""
不同尺寸的原始图片和不同batch_size下，arcface占用显存计算
---------- image_shape=128 base_filters=64  batch_size=8 ----------
初始图片: 393216  图片+参数: 38763712
过程图片: 46268416  图片+参数: 84638912
---------- image_shape=128 base_filters=32 batch_size=32 ----------
初始图片: 1572864  图片+参数: 11793504
过程图片: 93323264  图片+参数: 103543904
---------- image_shape=128 base_filters=64  batch_size=16 ----------
初始图片: 786432  图片+参数: 39156928
过程图片: 92536832  图片+参数: 130907328
---------- image_shape=64  batch_size=8 ----------
初始图片: 98304  图片+参数: 38468800
过程图片: 11567104  图片+参数: 49937600
---------- image_shape=64  batch_size=16 ----------
初始图片: 196608  图片+参数: 38567104
过程图片: 23134208  图片+参数: 61504704
---------- image_shape=64  batch_size=32 ----------
初始图片: 393216  图片+参数: 38763712
过程图片: 46268416  图片+参数: 84638912
---------- image_shape=64  batch_size=64 ----------
初始图片: 786432  图片+参数: 39156928
过程图片: 92536832  图片+参数: 130907328
---------- image_shape=64  batch_size=128 ----------
初始图片: 1572864  图片+参数: 39943360
过程图片: 185073664  图片+参数: 223444160
"""
import numpy as np


def print_nums(image_shape, base_filters, batch_size, w):
    print('-' * 10, f'image_shape={image_shape} base_filters={base_filters} batch_size={batch_size}', '-' * 10)

    s = image_shape
    nums = s * s * 3
    print(f'初始图片: {nums * batch_size}  图片+参数: {nums * batch_size + w}')

    s //= 2
    nums += s * s * base_filters
    s //= 2
    nums += s * s * base_filters
    for i, n in enumerate((3, 4, 6, 3)):
        s //= 2 if i > 0 else 1
        base_filters *= 2
        nums += s * s * base_filters * 6 * n
    print(f'过程图片: {nums * batch_size}  图片+参数: {nums * batch_size + w}')


w = 10220640  # 38370496  # 参数

image_shape = 128
base_filters = 32
for bs in (8, 16, 32, 64, 128):
    print_nums(image_shape, base_filters, bs, w)
# print_nums(image_shape=128, batch_size=8, w=w)
# print_nums(image_shape=128, batch_size=16, w=w)
