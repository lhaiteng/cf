# -*- coding: utf-8 -*-
import numpy as np
import math, cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import tensorflow as tf

## 设置属性防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False


# 获取矩阵的上三角阵，把结果拉成一维的
def get_triu(matrix, k=1):
    temp = np.ones_like(matrix, dtype=matrix.dtype)
    temp = np.triu(temp, k=k)
    return matrix[temp > 0]


def get_arcface_result(thetas, labels, labels_onehot, num_cls):
    # 预测结果
    pred = tf.argmin(thetas, axis=1, output_type=tf.int32)
    # 预测准确率
    acc = tf.reduce_mean(tf.cast(tf.equal(pred, labels), tf.float32))

    num_batch = tf.cast(tf.shape(thetas)[0], tf.float32)
    # 标签项的平均夹角
    thetas_label = tf.reduce_sum(thetas * labels_onehot)
    theta_label = thetas_label / num_batch
    # 非标签的平均夹角
    thetas_unlabel = tf.reduce_sum(thetas) - thetas_label
    theta_unlabel = thetas_unlabel / (num_batch * (num_cls - 1.))

    return acc, theta_label, theta_unlabel


# 画出人脸语义向量、人脸权重、centers的热力图
def heatmap_face_weight(ids, face_weight, prelogit_centers=None):
    plt.figure(figsize=[60, 20])
    total_fig = 2 if prelogit_centers is None else 3
    no_fig = 0

    # 人脸向量分布
    no_fig += 1
    plt.subplot(1, total_fig, no_fig)
    sns.heatmap(ids)
    plt.title(f'人脸向量分布')

    # 人脸权重矩阵
    no_fig += 1
    plt.subplot(1, total_fig, no_fig)
    num_zero = np.sum((face_weight < 1e-6) & (face_weight > -1e-6))
    sns.heatmap(face_weight)
    plt.title(f'人脸权重矩阵_num_zero: {num_zero}')

    # 人脸向量中心
    if prelogit_centers is not None:
        no_fig += 1
        plt.subplot(1, total_fig, no_fig)
        sns.heatmap(prelogit_centers)
        plt.title(f'人脸向量中心')

    plt.show()


# 与第一张图片，对比其他图片的相似度
def plot_similarity(imgs, ids, labels, suptitle=None, root_ing=0, **kwargs):
    """
    :param imgs:
    :param ids:
    :param labels:
    :param suptitle:
    :param root_ing:
    :param kwargs:
        plot: 默认True，False表示不画图
        save_file: 默认None不保存，否则按照save_file保存图片
    :return:
    """
    plot = kwargs.get('plot', True)
    save_file = kwargs.get('save_file', None)

    # 分别与第1张图片及其id求夹角余弦
    similarities = np.matmul(ids, ids.T)
    num = imgs.shape[0]
    cols = min(num, 3)
    rows = math.ceil(num / cols)

    plt.figure(figsize=[cols * 3, rows * 3])
    for ind, img in enumerate(imgs):
        plt.subplot(rows, cols, ind + 1)
        plt.imshow(img[:, :, ::-1])
        plt.title(f'{labels[ind]}:{similarities[root_ing, ind]:.3f}', y=-0.1)
        plt.axis('off')
    plt.suptitle(suptitle)
    if save_file: plt.savefig(save_file)
    if plot: plt.show()
    plt.close()


# 任意单个矩阵画出热力图
def heatmap_matrix(matrix, xlim=None, ylim=None, xticks=None, yticks=None,
                   title=f'余弦相似度矩阵-人为分级', *args, **kwargs):
    plt.figure(figsize=[20, 20])
    sns.heatmap(matrix, annot=True, cbar=False)
    if xlim is not None: plt.xlim(xlim)
    if ylim is not None: plt.ylim(ylim)
    if xticks is not None: plt.xticks(*xticks, fontsize=12)
    if yticks is not None: plt.yticks(*yticks, fontsize=12)
    plt.axis('equal')
    plt.title(title)
    plt.show()


# 将单个相似度矩阵绘制成热力图，并分级[0~1]，保留1位小数。
def heatmap_sim_matrix(sim_matrix, labels=[],
                       title=f'余弦相似度矩阵-人为分级', *args, **kwargs):
    """
    将相似度矩阵绘制成热力图，并分级。
    注意能绘制子图
    :param sim_matrix:
    :param labels: 标签，用来生成xlim, ylim, xticks, yticks
    :param title:
    :return:
    """
    sim_matrix[sim_matrix < 0] = 0
    sim_matrix = np.around(sim_matrix, 1)
    if labels:
        xylim = [-0.5, len(labels) + 0.5]
        xyticks = [np.arange(0.5, len(labels), 1), labels]
        heatmap_matrix(sim_matrix, xylim, xylim, xyticks, xyticks, title=title)
    else:
        heatmap_matrix(sim_matrix, title=title)


# 画出多个matrix
def heatmap_matrixs(sim_matrix_list, axis_info_list, num,
                    suptitles=None, **kwargs):
    """
    :param sim_matrix_list:
    :param axis_info_list: [[xlim, ylim, xticks, yticks], [xlim, ylim, xticks, yticks], ...]
    :param num:
    :param suptitles:
    :param kwargs:
        可选title_list
    :return:
    """
    cols = 3
    rows = math.ceil(num / cols)
    plt.figure(figsize=[10 * cols, 10 * rows])
    title_list = kwargs.get('title_list', [])

    for ind in range(num):
        plt.subplot(rows, cols, ind + 1)
        sim_matrix = sim_matrix_list[ind]

        sns.heatmap(sim_matrix, annot=True, cbar=False)
        if axis_info_list:
            xlim, ylim, xticks, yticks = axis_info_list[ind]
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.xticks(*xticks, fontsize=12)
            plt.yticks(*yticks, fontsize=12)
        if title_list: plt.title(title_list[ind])
        plt.axis('equal')
    if suptitles is None:
        plt.title(f'余弦相似度矩阵-人为分级')
    plt.show()


# 将多个相似度矩阵绘制成热力图，并分级[0~1]，保留1位小数。
def heatmap_sim_matrixs(sim_matrix_list, labels_list=[], num=1,
                        suptitles=None, **kwargs):
    """
    将相似度矩阵绘制成热力图，并分级。
    注意能绘制子图
    :param sim_matrix_list:
    :param num: 表示有几个图
    :param labels_list: 标签，用来生成xlim, ylim, xticks, yticks
    :param suptitles:
    :param kwargs:
        可选title_list
    :return:
    """
    if num == 2:
        heatmap_sim_matrix(sim_matrix_list, labels_list)
        return None

    cols = 3
    rows = math.ceil(num / cols)
    plt.figure(figsize=[10 * cols, 10 * rows])
    plt.show()

    # 此步的目的仅仅对列表中的sim_matrix固定分级
    axis_info_list = []
    for ind in range(num):
        plt.subplot(rows, cols, ind + 1)
        sim_matrix = sim_matrix_list[ind]
        sim_matrix[sim_matrix < 0] = 0
        sim_matrix_list[ind] = np.around(sim_matrix, 1)
        if labels_list:
            xylim = [-0.5, len(labels_list) + 0.5]
            xyticks = [np.arange(0.5, len(labels_list), 1), labels_list]
            axis_info_list.append([xylim, xylim, xyticks, xyticks])

    heatmap_matrixs(sim_matrix_list, axis_info_list, num, suptitles=suptitles, **kwargs)


# 画出logit的热力图分布，并标出标签和预测编号
def heatmap_logit(logits, labels, kind='angle', title='angle', **kwargs):
    """
    绘制角度/夹角余弦的热力图分布，并标明标签项o和预测项x
    注意sns.heatmap的x轴是列，y轴是行。y轴本来是从上往下的，但设置ylim后就变为从下往上
    单元格左上角为实际行列数，标注的坐标轴是单元格中心。
    行为样本，列为类别。
    :param logits: 与cls_weight的夹角或余弦
    :param labels:
    :param kind:
    :param title:
    :param kwargs:
        plot: 默认True
        save_file: 保存文件的路径
    :return:
    """
    plot = kwargs.get('plot', True)
    save_file = kwargs.get('save_file', None)

    if kind == 'angle':
        vmax, vmin = 90, 30
        pre_labels = np.argmin(logits, axis=1)
    else:
        vmax, vmin = 0.9, 0
        pre_labels = np.argmax(logits, axis=1)

    plt.figure(figsize=[10, 10])
    sns.heatmap(logits, cbar=False, vmax=vmax, vmin=vmin)  # 设定图例最大值最小值

    if len(labels.shape) > 1:
        labels = np.argmax(labels, axis=1)
    plt.scatter(labels + 0.5, np.arange(labels.shape[0]) + 0.5,
                color='k', marker='o', alpha=1)

    plt.scatter(pre_labels + 0.5, np.arange(labels.shape[0]) + 0.5,
                color='r', marker='x', alpha=1)

    plt.ylim([-0.5, labels.shape[0] + 0.5])
    plt.yticks(np.arange(0.5, len(labels), 1), labels)
    plt.title(title)

    if save_file: plt.savefig(save_file)
    if plot: plt.show()
    plt.close()


# 打印出夹角的信息，包括标签项和非均值项的最大最小值、均值、方差等
def print_thetas_infos(thetas, labels, cate='angle'):
    print_values_infos(thetas, labels, delta_value='min', cate=cate)


# 打印出values的信息，包括标签项和非均值项的最大最小值、均值、方差等
def print_values_infos(values, labels, delta_value='min', cate='angle'):
    # 打印出values的标签项、最大项
    argmax = np.argmax(values, axis=1)
    argmin = np.argmin(values, axis=1)
    value_label = values[range(values.shape[0]), labels]
    # 各样本的非标签项值
    values_unlabel = values.tolist()
    values_unlabel = np.array([values_unlabel[i][:labels[i]] + values_unlabel[i][labels[i] + 1:]
                               for i in range(len(values_unlabel))])
    value_mean = np.mean(values_unlabel, axis=1)
    value_max = np.max(values_unlabel, axis=1)
    value_min = np.min(values_unlabel, axis=1)
    delta = value_label - value_min if delta_value == 'min' else value_max - value_label

    show_data = [labels, argmin,
                 value_label, value_mean, value_max, value_min,
                 delta]
    index_name = ['arg_label', 'arg_min',
                  f'label', f'mean', f'max', f'min',
                  f'delta_{delta_value}']
    if delta_value != 'min': show_data[1], index_name[1] = argmax, 'arg_max'

    data = pd.DataFrame(show_data, index=index_name)
    data = data.T
    list1 = ['arg_label', 'arg_min'] if delta_value == 'min' else ['arg_label', 'arg_max']
    data[list1] = data[list1].astype(int)
    list2 = [f'label', f'mean', f'max', f'min', f'delta_{delta_value}']
    data[list2] = data[list2].round(3)
    data.set_index('arg_label', inplace=True)
    print(f'{"-" * 100}\n{cate}:\n{data}\n{"-" * 100}')


# 画出夹角余弦相似度分布
def distribution_sims(sims, labels=None, bins=None, title='cosine similarity', **kwargs):
    """
    画出夹角余弦相似度分布。
    :param sims: 余弦相似度或相似度矩阵
    :param labels: 若为None，说明是余弦相似度，直接画出分布。否则是与各类别的的余弦值，需要分成标签项和非标签项。
    :param bins: 默认np.linspace(-1, 1, 41)
    :param title:
        name_list: legend名称
        xlim, xticks: 默认[-1, 1], np.arange(-1, 1, 21)
        save_file: 保存路径，默认不保存
        plot: 是否显示，默认显示
    :return:
    """
    if bins is None: bins = np.linspace(-1, 1, 41)
    new_kw = {}
    if 'xlim' not in kwargs.keys(): new_kw['xlim'] = [-1, 1]
    if 'xticks' not in kwargs.keys(): new_kw['xticks'] = np.linspace(-1, 1, 21)
    if labels is None:  # 说明是列表
        distribution_values(sims, bins=bins, title=title, **kwargs, **new_kw)
    else:  # 说明是logit
        distribution_logits(sims, labels, bins=bins, title=title, **kwargs, **new_kw)


# 分别画出angles的标签项分布、非标签项分布。
def distribution_angles(angles, labels=None, bins=range(0, 180, 2), title='angle', **kw):
    new_kw = {}
    if 'xlim' not in kw.keys(): new_kw['xlim'] = [0, 180]
    if 'xticks' not in kw.keys(): new_kw['xticks'] = range(0, 190, 10)
    if labels is not None:
        distribution_logits(angles, labels, bins=bins, title=title, **new_kw, **kw)
    else:
        distribution_values(angles, bins=bins, title=title, **new_kw, **kw)


# 在同一张图中画出logits的标签项分布、非标签项分布。
def distribution_logits(logits, labels, bins=None, title='logits', **kwargs):
    """
    在同一张图中画出logits的标签项分布、非标签项分布。
    :param logits: [None, num_cls]
    :param labels: [None]
    :param kwargs:
        name_list: legend名称
        xlim, xticks:
        save_file: 保存路径
        plot: 是否显示
    :return:
    """

    values_label, values_unlabel = get_logits_label_unlabel(logits, labels)

    # 最大的方差及其样本编号
    max_var = np.max(np.var(values_unlabel, axis=1))
    arg_max_var = np.argmax(np.var(values_unlabel, axis=1))
    no1 = labels[arg_max_var]
    # 最小值与标签项最接近的样本，且最小值与标签项不同。那么就用标签项-最小值，值越大，标明误差越大
    min_values_unlabel = np.min(values_unlabel, axis=1)
    deltas = values_label - min_values_unlabel
    max_delta = np.max(deltas)
    arg_max_delta = np.argmax(deltas)
    no2 = labels[arg_max_delta]

    # 把标签项、非标签项的分布都画在一张图上
    values_list = [values_label, values_unlabel.reshape([-1])]
    name_list = [f'{title}_label', f'{title}_unlabel']
    title = f'{title}  mean_label: {values_label.mean():.3f}\n' \
            f'label: {no1}  max_var: {max_var:.3f}\n' \
            f'label: {no2}  max_delta: {max_delta:.3f}'
    new_kew = {}
    if 'name_list' not in kwargs.keys(): new_kew['name_list'] = name_list
    distribution_values(values_list, bins=bins, title=title, **kwargs, **new_kew)


# 画出多个一维数组的分布情况
def distribution_values(values_list: list, bins=None, title='distribution', **kwargs):
    """
    画出多个一维数组的分布情况
    :param values_list:
    :param bins:
    :param title:
    :param kwargs:
        name_list: legend名称
        xlim, xticks:
        save_file: 保存路径
        plot: 是否显示
    :return:
    """
    name_list = kwargs.get('name_list', [])
    xlim = kwargs.get('xlim', None)
    xticks = kwargs.get('xticks', None)
    plt.figure(figsize=[10, 10])
    for ind, values in enumerate(values_list):
        name = name_list[ind] if name_list else ind
        sns.distplot(values, bins=bins, hist=True, kde=True, rug=False,
                     vertical=False, hist_kws={'alpha': 0.3, 'edgecolor': 'k'}, label=name)
    plt.title(title)
    if xlim is not None: plt.xlim(xlim)
    if xticks is not None: plt.xticks(xticks)
    if name_list: plt.legend()
    plt.grid(linestyle='--')
    save_file = kwargs.get('save_file', None)
    plot = kwargs.get('plot', True)
    if save_file: plt.savefig(save_file)
    if plot: plt.show()
    plt.close()


# 得到lopgit的标签项和非标签项，返回一维数组
def get_logits_label_unlabel(logits, labels, unlabel_ndim=2):
    logits = np.asarray(logits)
    num = logits.shape[0]
    values_label = logits[range(num), labels]  # 各样本的标签项值
    # 各样本的非标签项值
    unlable_inds = np.zeros_like(logits)
    unlable_inds[range(num), labels] = 1  # 标签项
    if unlabel_ndim == 2:  # logits[inds]索引出来会自动拉成一维，若保持2维形状需要reshape一下
        values_unlabel = logits[unlable_inds == 0].reshape([num, -1])
    else:
        values_unlabel = logits[unlable_inds == 0]
    return values_label, values_unlabel


if __name__ == '__main__':
    # cosines = np.random.uniform(-1, 1, size=[64, 284])
    # thetas_angle = np.arccos(cosines) * 180 / math.pi
    # labels_list = np.random.randint(0, 284, size=[64])
    #
    # print(f'thetas_angle:\n{thetas_angle}')
    # print(f'labels_list:\n{labels_list}')
    # print(f'thetas_label:\n{thetas_angle[range(64), labels_list]}')
    # plot_distribution_angle(thetas_angle, labels_list)
    pass
