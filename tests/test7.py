# -*- coding: utf-8 -*-
"""
在ssd中，几种计算P、TP的时间消耗对比
方式1：单独计算每张图片的is_tps，同时累积P、TP
    1. 使用提取下标累积
    2. 使用顺序切片累积
方式2：计算每张图片的is_tps，累积is_tps，最后再计算P、TP
=> 累积is_tps，最后再计算P、TP时间消耗最短
=> 要注意方式2累积单张is_tps[n_iou_thres, n_boxes]时，因为n_iou_thrs不变，而n_boxes随累积进行而增加
    所以要对单张图片的is_tps进行转置后用列表的extend功能，累积成[n_boxes, n_iou_thres], 累积完成后再转置回来
    因为np.column_stack消耗时间成平方增长
=> 注意对形成的is_tps沿score从大到小排序
"""
import os, time, sys
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


@time_calc('提取下标')
def f1(score_thres_list, scores, is_tps, n=5):
    n_score_thres = score_thres_list.size
    n_iou_thres = 3
    total_P = np.zeros([n_score_thres])
    TP = np.zeros([n_iou_thres, n_score_thres])
    for _ in range(n):
        for ind_score, score_thres in enumerate(score_thres_list):
            P_inds = np.where(scores >= score_thres)[0]
            total_P[ind_score] += P_inds.size
            TP[:, ind_score] += np.sum(is_tps[:, P_inds], axis=1)
    total_P += 1e-6
    return total_P, TP


@time_calc('顺序切片')
def f2(score_thres_list, scores, is_tps, n=5):
    n_score_thres = score_thres_list.size
    n_iou_thres = 3
    total_P = np.zeros([n_score_thres])
    TP = np.zeros([n_iou_thres, n_score_thres])
    for _ in range(n):
        if scores is None: scores = np.sort(np.random.random(size=[n_boxes]))[::-1]  # 每张图片的得分
        if is_tps is None: is_tps = np.random.randint(0, 2, size=[3, n_boxes])  # 每张图片的is_tps
        for ind_score, score_thres in enumerate(score_thres_list):
            P_inds = np.where(scores >= score_thres)[0]
            n = P_inds.size
            if n > 0:
                total_P[ind_score] += n
                TP[:, ind_score] += np.sum(is_tps[:, :n], axis=1)
    total_P += 1e-6
    return total_P, TP


@time_calc('score阈值与scores一一对应')
def f3(scores, is_tps, n=5):
    n_score_thres = scores.size
    n_iou_thres = 3
    total_P = np.zeros([n_score_thres])
    TP = np.zeros([n_iou_thres, n_score_thres])
    for _ in range(n):
        for i, score in enumerate(scores):
            total_P[i] += 1
            TP[:, i] += is_tps[:, i]
            if i > 0:
                total_P[i] += total_P[i - 1]
                TP[:, i] += TP[:, i - 1]
    total_P += 1e-6
    return total_P, TP


n_iou_thres = 3
n_boxes = int(100e4)
scores = np.sort(np.random.random(size=[n_boxes]))[::-1]
is_tps = np.random.randint(0, 2, size=[n_iou_thres, n_boxes])
print(f'size of scores: {sys.getsizeof(scores)}')
print(f'size of is_tps: {sys.getsizeof(is_tps)}')

n_score_thres = 1000
score_thres_list = np.linspace(0, 1, n_score_thres)[::-1]
# score_thres_list = scores

n = 1
P1, TP1 = f1(score_thres_list, scores, is_tps, n)
P2, TP2 = f2(score_thres_list, scores, is_tps, n)
wrong_inds = np.where(TP1[0] > P1)[0]
print(f'num of wrong inds: {wrong_inds.size} .')
print((P1 - P2).sum())
print((TP1 - TP2).sum())
P3, TP3 = f3(scores, is_tps, n)
# print((P2 - P3).sum())
# print((TP2 - TP3).sum())

# 计算一下两种方式的PR曲线
n_gt = int(70e4)
rs1, ps1 = TP1 / n_gt, TP1 / P1
rs2, ps2 = TP3 / n_gt, TP3 / P3

from ssd.dl16_CF_ssd_utils import get_APs, plot_PR

APs1 = get_APs(rs1, ps1)
plot_PR(rs1, ps1, [0, 1, 2], APs1, title='1000 score_thres PR-curve')

APs2 = get_APs(rs2, ps2)
plot_PR(rs2, ps2, [0, 1, 2], APs2, title='all score_thres PR-curve')

"""对比两种方式：1.单独计算图片累积成TP；2.所有图片is_tps汇总后，再计算TP"""
n_iou_thres = 3

n_boxes = int(100)  # 每张图片有n_boxes个box
print(f'num of boxes per pics: {n_boxes}.')
for n in (10, 100, 1000, 5000):  # n表示图片总数
    print('-' * 100)
    print(f'n = {n}.')
    # 方式1：每张图片的is_tps单独计算，再汇总成P，TP
    print(f'size of scores: {sys.getsizeof(scores)}')
    print(f'size of is_tps: {sys.getsizeof(is_tps)}')
    # 提前设定好score阈值
    n_score_thres = 1000
    score_thres_list = np.linspace(0, 1, n_score_thres)[::-1]
    # 计算TP
    P2, TP2 = f2(score_thres_list, scores=None, is_tps=None, n=n)

    # 方式2：先得到所有的is_tps再计算TP
    # 把方式1中的scores累积起来
    t = time.time()
    all_scores, all_is_tps = [], []
    for i in range(n):
        scores = np.sort(np.random.random(size=[n_boxes]))[::-1]  # 每张图片的得分
        is_tps = np.random.randint(0, 2, size=[n_iou_thres, n_boxes])  # 每张图片的is_tps
        all_scores.extend(scores)  # 所有图片的得分
        all_is_tps.extend(is_tps.T)  # 所有图片的is_tps
    all_is_tps = np.transpose(all_is_tps)
    print(f'方式2生成 {n} 张图片all_is_tps用时: {time.time() - t:.3f} s.')
    # 排序
    t = time.time()
    all_scores = np.array(all_scores)
    new_inds = np.argsort(all_scores)[::-1]
    all_scores = all_scores[new_inds]
    all_is_tps = all_is_tps[:, new_inds]
    print(f'方式2生成 {n} 张图片排序用时: {time.time() - t:.3f} s.')
    print(f'size of all_scores: {sys.getsizeof(all_scores)}')
    print(f'size of all_is_tps: {sys.getsizeof(all_is_tps)}')
    # 不设定score阈值
    # n_score_thres = 1000
    # score_thres_list = np.linspace(0, 1, n_score_thres)[::-1]
    # 计算TP
    P3, TP3 = f3(scores, is_tps, n)

"""对比np.column_stack的时间消耗，与优化方法"""


@time_calc('计算循环')
def f(n):
    for i in range(n):
        scores = np.sort(np.random.random(size=[n_boxes]))[::-1]  # 每张图片的得分
        is_tps = np.random.randint(0, 2, size=[n_iou_thres, n_boxes])  # 每张图片的is_tps


f(1000)
f(2000)


@time_calc('加入stack')
def f(n):
    all_tps = np.array([]).reshape([3, 0])
    for i in range(n):
        scores = np.sort(np.random.random(size=[n_boxes]))[::-1]  # 每张图片的得分
        is_tps = np.random.randint(0, 2, size=[n_iou_thres, n_boxes])  # 每张图片的is_tps
        all_tps = np.column_stack((all_tps, is_tps))
    print(all_tps.shape)


f(1000)
f(2000)


@time_calc('转置成列表，使用extend')
def f(n):
    all_tps = []
    for i in range(n):
        scores = np.sort(np.random.random(size=[n_boxes]))[::-1]  # 每张图片的得分
        is_tps = np.random.randint(0, 2, size=[n_iou_thres, n_boxes])  # 每张图片的is_tps
        all_tps.extend(is_tps.T)
    all_tps = np.transpose(all_tps)
    print(all_tps.shape)


f(1000)
f(2000)
