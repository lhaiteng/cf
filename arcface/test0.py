# -*- coding: utf-8 -*-
"""


"""
import numpy as np
import matplotlib.pyplot as plt

# import tensorflow as tf


def quick_sort(l: list, start, end):
    if start >= end: return
    i, j, x = start, end, l[start]
    while i < j:
        while i < j and l[j] > x:
            j -= 1
        if i < j:
            l[i] = l[j]
            i += 1
        while i < j and l[i] < x:
            i += 1
        if i < j:
            l[j] = l[i]
            j -= 1
    l[i] = x
    quick_sort(l, start, i - 1)
    quick_sort(l, i + 1, end)


l = np.random.randint(0, 100, 20).tolist()
print(l)
quick_sort(l, 0, 19)
print(l)






