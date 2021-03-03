"""
求iou和giou
"""
import numpy as np
import cv2
import seaborn as sns
import matplotlib.pyplot as plt


# import tensorflow as tf

def relu(x):
    return np.maximum(x, 0)


# 使用relu求overlap
def overlap1(l1, l2):
    a1, a2 = np.min(l1, axis=1), np.max(l1, axis=1)
    b1, b2 = np.min(l2, axis=1), np.max(l2, axis=1)
    return relu(b2 - a1 - relu(b2 - a2) - relu(b1 - a1))


# 比较大小关系求overlap
def overlap2(l1, l2):
    a1, a2 = np.min(l1, axis=1), np.max(l1, axis=1)
    b1, b2 = np.min(l2, axis=1), np.max(l2, axis=1)

    res = np.zeros_like(a1)


    return res
