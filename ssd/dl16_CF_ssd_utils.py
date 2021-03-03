# -*- coding: utf-8 -*-
import cv2, time
import numpy as np
import matplotlib.pyplot as plt
from ssd.dl16_CF_ssd_config import ssd_cfg


def time_calc(text):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            f = func(*args, **kwargs)
            exec_time = time.time() - start_time
            print(f'exec_time for {text}: {exec_time:.3f} s.')
            return f

        return wrapper

    return decorator


def print_result(epoch, epochs, batch, num_batch,
                 loss_cla=0, loss_reg=0):
    print(f'\rEpoch {epoch + 1}/{epochs}  Batch {batch + 1}/{num_batch}'
          f' - Loss cla = {loss_cla:.6f} reg = {loss_reg:.6f}',
          end='')


def plot_boxes_img(img, multi_boxes, color_list, line_width=2, show_img=True):
    """
    在一张图中画多组标注框。所有boxes都是r1, c1, r2, c2
    :param img:
    :param multi_boxes: [gt_boxes, rpn_boxes, fast_boxes]
    :param color_list:
    :param line_width:
    :param show_img:
    :return:
    """

    color_dict = {'k': (0, 0, 0), 'y': (255, 0, 0), 'g': (0, 255, 0), 'r': (0, 0, 255)}

    for i, boxes in enumerate(multi_boxes):
        color = color_list[i]
        if color not in color_dict:
            color = 'k'
        for box in boxes:
            img = cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]),
                                color_dict[color], line_width)

    if show_img:
        plt.imshow(img[:, :, ::-1])
        plt.show()


# 根据分类结果、回归结果，进行boxes调整
def get_result(img_shape, cla_prob, regs, score_thres=ssd_cfg.test_result_score_thres):
    """
    :param img_shape:
    :param cla_prob: [num_boxes]
    :param regs: [num_boxes, 4]
    :param score_thres: 保留作为预测结果的得分阈值
    :return:
        cla_id_scores [num_res, 2]
        boxes [num_res, 4] 按照得分从大到小排序好了
    """
    # 生成所有锚框
    img_h, img_w = img_shape
    total_anchor_boxes = get_all_anchor_boxes(img_shape)

    # 是前景且概率大于阈值的索引
    fg_inds = np.where(cla_prob > score_thres)[0]

    # 满足概率阈值的前景个数
    num_res = fg_inds.size
    # 若无结果，则直接返回
    if num_res == 0:
        return np.array([]).reshape([0, 2]), np.array([]).reshape([0, 4])

    # 前景概率大于阈值的分类id、概率、boxes、regs
    scores = cla_prob[fg_inds]  # score
    boxes = total_anchor_boxes[fg_inds]
    regs = regs[fg_inds]

    shifts = regs
    # 根据回归结果调整boxes
    boxes = bbox_transform_inv(boxes, shifts)
    # 边界处截断
    boxes[boxes[:, 0] < 0, 0] = 0
    boxes[boxes[:, 1] < 0, 1] = 0
    boxes[boxes[:, 2] > img_h - 1, 2] = img_h - 1
    boxes[boxes[:, 3] > img_w - 1, 3] = img_w - 1
    # 保留尺寸大于阈值的预测框
    _keep_inds = np.where(((boxes[:, 2] - boxes[:, 0]) > ssd_cfg.test_min_size) &
                          ((boxes[:, 3] - boxes[:, 1]) > ssd_cfg.test_min_size))[0]

    # 若无结果，则直接返回
    if _keep_inds.size == 0:
        return np.array([]).reshape([0, 2]), np.array([]).reshape([0, 4])

    scores = scores[_keep_inds]  # score
    boxes = boxes[_keep_inds]

    # 若有结果，则对结果进行nms
    # 将boxes按照得分从高到低排布
    sorted_inds = np.argsort(scores)[::-1]
    scores = scores[sorted_inds]
    boxes = boxes[sorted_inds]

    keep_inds = self_nms(boxes, ssd_cfg.test_result_nms_iou)

    nmsed_scores = scores[keep_inds]
    nmsed_boxes = boxes[keep_inds]

    return nmsed_scores.astype(float), nmsed_boxes.astype(float)


# 一组boxes从前往后进行自nms筛选，得到保留boxes的下标
def self_nms(boxes, thres):
    num_boxes = boxes.shape[0]
    keep_inds = np.ones(num_boxes)  # 0是剔除，1是保留

    for i, box in enumerate(boxes):
        # 若当前box已剔除，则计算下一个
        if keep_inds[i] == 0:
            continue
        if i + 1 > boxes.shape[0] - 1:
            break
        last_boxes = boxes[i + 1:]
        ious = get_iou(last_boxes, box)
        # 剔除与当前box的iou大于阈值的boxes
        kill_inds = np.where(ious > thres)[0] + (i + 1)
        keep_inds[kill_inds] = 0

    return np.where(keep_inds == 1)[0]


def get_conv_infos(img_shape, conv_layers, first_h, first_w):
    """
    :param img_shape:
    :param conv_layers: [(p, k, s), ...]
    :param first_h: (n_h, j_h, r_h, START_h)
    :param first_w: (n_w, j_w, r_w, START_w)
    :return: conv_infos [((n_h, j_h, r_h, START_h), (n_w, j_w, r_w, START_w)), ...]
    """
    # 开始信息
    x_h, x_w = img_shape
    if first_h is None:
        first_h = (x_h, 1, 1, 0)  # (n_h, j_h, r_h, START_h)
    if first_w is None:
        first_w = (x_w, 1, 1, 0)  # (n_w, j_w, r_w, START_w)

    h_info, w_info = first_h, first_w
    conv_infos = []

    for layer in conv_layers:
        h_info = _get_conv_output_size(*h_info, *layer)
        w_info = _get_conv_output_size(*w_info, *layer)
        conv_infos.append((h_info, w_info))

    return conv_infos


def _get_conv_output_size(n, j, r, start, p, k, s):
    """
    输入信息， 经过pks卷积，返回输出信息
    :param n: 尺寸
    :param j: 单点间距
    :param r: 感受野
    :param start: 左上第一点对应原图坐标
    :param p: padding圈数
    :param k: 卷积核尺寸
    :param s: 步长
    :return:
    """
    _n = int(np.floor((n + 2 * p - k) / s) + 1)
    _j = j * s
    _r = r + (k - 1) * j
    _start = start + ((k - 1) / 2 - p) * j
    return _n, _j, _r, _start


def get_all_anchor_boxes(img_shape, first_h=None, first_w=None,
                         conv_layers=ssd_cfg.conv_layers, anchor_scales=ssd_cfg.anchor_scales,
                         anchor_ratio=ssd_cfg.anchor_ratio, fm_inds=ssd_cfg.fm_inds):
    conv_infos = get_conv_infos(img_shape, conv_layers, first_h, first_w)
    # # [((n_h, j_h, r_h, START_h), (n_w, j_w, r_w, START_w)), ...]
    total_anchor_boxes = None
    for i in range(len(fm_inds)):
        # [num_anchor_box, 4] -> r1, c1, r2, c2
        point_anchor_box = gen_point_anchor_box(anchor_scales[i], anchor_scales[i + 1],
                                                anchor_ratio)
        # [num_shifts * num_anchor_box, 4] -> r1, c1, r2, c2
        # 按行排列 □→□→□→□→□→□→□→□→□→□↓
        #         ↓←              ←
        #         □→□→□→□→□→...
        fm_ind = fm_inds[i]
        total_anchor_box = gen_all_anchor_boxes(img_shape, point_anchor_box, conv_infos[fm_ind])
        if total_anchor_boxes is None:
            total_anchor_boxes = total_anchor_box
        else:
            total_anchor_boxes = np.r_[total_anchor_boxes, total_anchor_box]
    return total_anchor_boxes


def gen_point_anchor_box(scale1, scale2, ratio):
    """
    :param scale1: int
    :param scale2: int
    :param ratio: list
    :return: [num_anchor_box， 4]
    """
    ratio = np.array(ratio)  # [num_ratio]
    w = scale1 / (ratio ** 0.5)  # [num_ratio]
    h = w * ratio  # [num_ratio]
    w = np.append(w, (scale1 * scale2) ** 0.5)
    h = np.append(h, (scale1 * scale2) ** 0.5)
    zeros = np.zeros_like(w)

    r1 = zeros - 0.5 * (h - 1)
    c1 = zeros - 0.5 * (w - 1)
    r2 = zeros + 0.5 * (h - 1)
    c2 = zeros + 0.5 * (w - 1)

    return np.vstack((r1, c1, r2, c2)).transpose()


# 原图中的所有锚框。按照行排列 ->->->->->->
def gen_all_anchor_boxes(img_shape, point_anchor_box, conv_info):
    """
    使用一组单点锚框，根据锚框偏移量，广播得到所有锚框。
    注意所有锚框[num_boxes, 4]需要按照行排列，因为需要对[fm_h, fm_w, 4].reshape([-1, 4])得到对应位置
    :param img_shape:
    :param point_anchor_box: [num_point_anchor_box, 4]
    :param conv_info: ((n_h, j_h, r_h, START_h), (n_w, j_w, r_w, START_w))
    :return: [num_boxes, 4]
    """
    img_h, img_w = img_shape
    fm_h, hj, hr, hstart = conv_info[0]
    fm_w, wj, wr, wstart = conv_info[1]
    # 使用卷积公式计算锚框中心点
    shift_r = [hstart + i * hj for i in range(fm_h)]
    shift_c = [wstart + i * wj for i in range(fm_w)]
    # # 使用特征图均分原图，各网格中点作为锚框中心点
    # _hj = img_h / fm_h
    # _hstart = _hj / 2
    # _wj = img_w / fm_w
    # _wstart = _wj / 2
    # shift_r = [_hstart + i * _hj for i in range(fm_h)]
    # shift_c = [_wstart + i * _wj for i in range(fm_w)]

    # c作为x，r作为y，如此偏移得到的boxes是沿行偏移的
    shift_c, shift_r = np.meshgrid(shift_c, shift_r)
    shifts = np.vstack((shift_r.ravel(), shift_c.ravel(), shift_r.ravel(), shift_c.ravel())).transpose()

    num_point_anchor_box = point_anchor_box.shape[0]
    num_shifts = shifts.shape[0]

    point_anchor_box = np.reshape(point_anchor_box, [1, num_point_anchor_box, 4])
    shifts = np.reshape(shifts, [num_shifts, 1, 4])

    anchor_boxes = point_anchor_box + shifts  # [num_shifts, num_anchor_box, 4]
    anchor_boxes = np.reshape(anchor_boxes, [-1, 4])  # [num_shifts * num_anchor_box, 4]

    return anchor_boxes


# 一组boxes和多个gt_box的ious矩阵
def get_ious(boxes, gt_boxes):
    ious = np.zeros((boxes.shape[0], gt_boxes.shape[0]))

    for i, gt_box in enumerate(gt_boxes):
        iou = get_iou(boxes, gt_box)
        ious[:, i] = iou

    return ious  # [num_boxes, num_boxes]


# 一组boxes和一个box的ious
def get_iou(boxes, gt_box):
    r1, c1, r2, c2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    r1_gt, c1_gt, r2_gt, c2_gt = gt_box[0], gt_box[1], gt_box[2], gt_box[3]
    over_r = get_overlap_length(r1, r2, r1_gt, r2_gt)  # [num_boxes]
    over_c = get_overlap_length(c1, c2, c1_gt, c2_gt)  # [num_boxes]
    inter = over_r * over_c  # [num_boxes]
    box_area = np.abs((r1 - r2) * (c1 - c2))  # [num_boxes]
    gt_area = np.abs((r1_gt - r2_gt) * (c1_gt - c2_gt))  # []
    union = box_area + gt_area - inter
    iou = inter / union
    return iou


def relu(x):
    return np.maximum(x, 0)


def get_overlap_length(a1, a2, b1, b2):
    a1, a2 = np.minimum(a1, a2), np.maximum(a1, a2)
    b1, b2 = np.minimum(b1, b2), np.maximum(b1, b2)
    return relu(a2 - a1 - relu(b1 - a1) - relu(a2 - b2))


# 根据原box和目标box，得到回归标签
def bbox_transform(ex_rois, gt_rois):
    # ex_rois -> r1, c1, r2, c2
    # gt_rois -> r1, c1, r2, c2
    # return -> r, z, h ,w

    ex_heights = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_widths = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_r = ex_rois[:, 0] + 0.5 * (ex_heights - 1)
    ex_ctr_c = ex_rois[:, 1] + 0.5 * (ex_widths - 1)

    gt_heights = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_widths = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_r = gt_rois[:, 0] + 0.5 * (gt_heights - 1)
    gt_ctr_c = gt_rois[:, 1] + 0.5 * (gt_widths - 1)

    targets_dr = (gt_ctr_r - ex_ctr_r) / ex_heights
    targets_dc = (gt_ctr_c - ex_ctr_c) / ex_widths
    targets_dh = np.log(gt_heights / ex_heights)
    targets_dw = np.log(gt_widths / ex_widths)

    targets = np.vstack((targets_dr, targets_dc, targets_dh, targets_dw)).transpose()

    return targets


# 根据box和回归标签，得到调整后的box
def bbox_transform_inv(boxes, deltas):
    # boxes -> r1, c1, r2, c2
    # deltas -> r, z, h, w
    # return -> r1, c1, r2, c2

    # if boxes.shape[0] == 0:
    #     return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)
    #
    # boxes = boxes.astype(deltas.dtype, copy=False)

    heights = boxes[:, 2] - boxes[:, 0] + 1.0
    widths = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_r = boxes[:, 0] + 0.5 * (heights - 1)
    ctr_c = boxes[:, 1] + 0.5 * (widths - 1)

    dr = deltas[:, 0::4]  # 虽然只有1个数，但如此得到的是二维数组[num_deltas, 1]
    dc = deltas[:, 1::4]
    dh = deltas[:, 2::4]
    dw = deltas[:, 3::4]

    pred_ctr_r = dr * heights[:, np.newaxis] + ctr_r[:, np.newaxis]
    pred_ctr_c = dc * widths[:, np.newaxis] + ctr_c[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]

    pred_boxes = np.zeros(boxes.shape, dtype=boxes.dtype)
    # r1
    pred_boxes[:, 0::4] = pred_ctr_r - 0.5 * (pred_h - 1)
    # c1
    pred_boxes[:, 1::4] = pred_ctr_c - 0.5 * (pred_w - 1)
    # r2
    pred_boxes[:, 2::4] = pred_ctr_r + 0.5 * (pred_h - 1)
    # c2
    pred_boxes[:, 3::4] = pred_ctr_c + 0.5 * (pred_w - 1)

    return pred_boxes


def get_APs(recalls, precisions, method=None):
    """
    根据PR对来计算得到某一类别的AP值
    :param recalls: [n_iou_thres, n_score_thres]
    :param precisions: [n_iou_thres, n_score_thres]
    :param method: 根据什么方式计算。None默认的是最大precision求面积，其实就是all points方法
            可选方法：11-points、all-points
    :return:
    """
    if type(recalls) in (list, tuple): recalls, precisions = np.array(recalls), np.array(precisions)
    if recalls.ndim == 2:  # 表明是多iou_thres
        aps = [_get_AP(recalls[i], precisions[i], method) for i in range(recalls.shape[0])]
    else:
        aps = [_get_AP(recalls, precisions, method)]
    return aps


def _get_AP(recalls, precisions, method):
    """
    根据什么方式计算。None默认的是最大precision求面积，其实就是all points方法
    可选方法：11-points、all-points
    :param recalls: [n_score_thres]
    :param precisions: [n_iou_thres]
    :param method:
    :return:
    """
    ap = temp_r = 0
    if method is None:
        sorted_inds = np.argsort(precisions)[::-1]  # 按照precision从大到小排列
        recalls, precisions = recalls[sorted_inds], precisions[sorted_inds]
        while recalls.size > 0:
            r, p = recalls[0], precisions[0]
            ap, temp_r = ap + (r - temp_r) * p, r
            inds = np.where(recalls > r)[0]
            recalls, precisions = recalls[inds], precisions[inds]
    elif method == 'all-points':
        sorted_inds = np.argsort(recalls)  # 按照recall从小到大排列
        recalls, precisions = recalls[sorted_inds], precisions[sorted_inds]
        for i, r in enumerate(recalls):
            p = np.max(precisions[recalls >= r])
            ap, temp_r = ap + (r - temp_r) * p, r
    elif method == '11-points':
        for r in np.linspace(0, 1, 11):
            _p = precisions[recalls >= r]
            if _p.size > 0: ap += np.max(_p)
        ap /= 11
    else:
        raise ValueError(f'method:{method} must be in (None, "all_points", "11_points")')
    return ap


# 根据is_tps画出PR曲线
def plot_PR_from_is_tps(n_gt, scores, is_tps, score_thres_list=None, iou_thres_list=None):
    """
    :param n_gt:
    :param scores: [num_box]
    :param is_tps: [num_iou_thres, num_box]
    :param score_thres_list: 默认None是根据tps中的每个scores进行划分求解
    :param iou_thres_list: 
    :return:
    """
    if score_thres_list is None:
        score_thres_list = scores
    num_score_thres = len(score_thres_list)
    num_iou_thres = is_tps.shape[1] - 1
    total_p = np.zeros(shape=[num_score_thres])
    cum_tps = np.zeros(shape=[num_iou_thres, num_score_thres])

    for ind_score_thres, score_thres in enumerate(score_thres_list):
        ind_p = np.where(scores >= score_thres)[0]
        total_p[ind_score_thres] = ind_p.size
        for ind_iou_thres, iou_thres in enumerate(iou_thres_list):
            cum_tps[ind_iou_thres, ind_score_thres] = np.sum(is_tps[ind_iou_thres, ind_p])
    recalls = cum_tps / n_gt
    precisions = cum_tps / total_p

    plot_PR(recalls, precisions, iou_thres_list)


def plot_PR(recalls, precisions, iou_thres_list, APs=None, title='PR-curve'):
    plt.figure(figsize=[10, 10])
    for ind_iou_thres, iou_thres in enumerate(iou_thres_list):
        label = f'iou_thres={iou_thres:.2f}'
        if APs: label += f', AP={APs[ind_iou_thres]:.2%}'
        plt.plot(recalls[ind_iou_thres], precisions[ind_iou_thres], label=label)
    plt.legend()
    plt.xlim([0, 1.1])
    plt.ylim([0, 1.1])
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title(title)
    plt.show()


# 得到各阈值下所有锚框的is_tps [num_iou_thres, num_box]
def get_is_tps(gt_infos, pred_boxes, mAP_iou_thres_list):
    """

    :param gt_infos:
    :param pred_boxes: 已经按照得分从大到小排序好
    :param mAP_iou_thres_list:
    :return:
    """
    n_gt = gt_infos.shape[0]
    n_iou_thres = len(mAP_iou_thres_list)
    n_boxes = pred_boxes.shape[0]
    is_tps = np.zeros(shape=[n_iou_thres, n_boxes])
    if n_gt == 0: return is_tps
    ious = get_ious(pred_boxes, gt_infos)
    for ind_iou, iou_thres in enumerate(mAP_iou_thres_list):
        gt_inds = list(range(n_gt))
        for ind_box in range(n_boxes):
            iou = ious[ind_box, gt_inds]
            max_iou, argmax = np.max(iou), int(np.argmax(iou))
            if max_iou >= iou_thres:
                gt_inds.pop(argmax)
                is_tps[ind_iou, ind_box] = 1
                if not gt_inds: break

    return is_tps


if __name__ == '__main__':
    tps = [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    n_p = len(tps)
    tps = np.array(tps)

    n_gt = 15
    total_P = np.arange(1, n_p + 1)
    TP = np.array([np.sum(tps[:i + 1]) for i in range(n_p)])
    recalls = TP / n_gt
    precisions = TP / total_P

    for method in (None, '11-points', 'all-points'):
        aps = get_APs(recalls, precisions, method=method)
        print(f'{method}: {aps[0]:.2%}')
