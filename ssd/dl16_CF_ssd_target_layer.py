"""
检验fast_target_layer是否正确：
"""
# -*- coding: utf-8 -*-
import numpy as np
from ssd.dl16_CF_ssd_config import ssd_cfg
from ssd.dl16_CF_ssd_utils import get_all_anchor_boxes
from ssd.dl16_CF_ssd_utils import get_ious
from ssd.dl16_CF_ssd_utils import bbox_transform


def ssd_target_layer(img_shape, gt_infos):
    """
    根据所有锚框，生成单独对fast训练的分类标签和回归标签
    :param img_shape: img_h, img_w
    :param gt_infos: [num_boxes, 4] -> r1, c1, r2, c2
    :return: cla_labels, reg_labels
        cla_labels [sum_anchor_box] -> -1未选中，0后景
        reg_labels [sum_anchor_box, 4] -> tr, tc, th, tw
    """
    img_h, img_w = img_shape
    total_anchor_boxes = get_all_anchor_boxes(img_shape)

    # 使用全部锚框，超出图片的也使用
    anchor_boxes = total_anchor_boxes
    labels = np.empty([total_anchor_boxes.shape[0]])

    labels.fill(-1)

    # 求锚框与所有gt_boxes的iou矩阵  [num_inside, num_gt_boxes]
    ious = get_ious(anchor_boxes, gt_infos)

    # 锚框与gt_box的最大IOU对应关系
    max_iou_per_box = np.max(ious, axis=1)
    max_gt_id_per_box = np.argmax(ious, axis=1)
    max_iou_per_gt = np.max(ious, axis=0)
    max_box_id_per_gt = np.argmax(ious, axis=0)

    # 所有满足负样本条件的下标
    # - 在高低阈值之间
    # 首先，取得负样本标签
    # 在lo~high之间
    big_bg_inds = np.where((max_iou_per_box < ssd_cfg.train_ssd_negative_iou_high) &
                           (max_iou_per_box > ssd_cfg.train_ssd_negative_iou_lo))[0]

    labels[big_bg_inds] = -0.1
    # 比lo还小
    small_bg_inds = np.where((max_iou_per_box < ssd_cfg.train_ssd_negative_iou_lo))[0]
    labels[small_bg_inds] = -0.2

    # 之后，取得正样本标签，以免同时满足最大IOU和负样本条件的样本被覆盖认为是负样本
    # 所有满足正样本条件的下标
    fg_inds1 = np.where(ious == max_iou_per_gt)[0]  # 取得最大IOU的是正样本
    fg_inds2 = np.where(max_iou_per_box > ssd_cfg.train_ssd_positive_iou)[0]  # 满足iou阈值条件的是正样本
    all_fg_inds = np.unique(np.append(fg_inds1, fg_inds2))
    # 把所有正样本都标记为其真实分类
    all_fg_gt_ids = max_gt_id_per_box[all_fg_inds]  # 前景样本对应的gt
    labels[all_fg_inds] = 1  # gt_infos[all_fg_gt_ids, 0]

    # 若正样本过多，抽取多余个为不选中
    all_fg_inds = np.where(labels > 0)[0]  # 当前所有正样本下标
    max_num_fg = ssd_cfg.train_ssd_fg_fraction * ssd_cfg.train_ssd_batch_size  # 最大正样本数目
    if all_fg_inds.size > max_num_fg:
        kill_num = int(all_fg_inds.size - max_num_fg)
        # 抽取小iou为不选中
        kill_fg_inds = np.argsort(max_iou_per_box[all_fg_inds])
        kill_fg_inds = all_fg_inds[kill_fg_inds][:kill_num]
        labels[kill_fg_inds] = -1
    # 最终选取的正样本下标
    fg_inds = np.where(labels > 0)[0]

    # 大后景和小后景样本按比例构成负样本，多余样本为不选中
    # 当前所有负样本下标
    big_bg_inds = np.where((labels == -0.1))[0]
    small_bg_inds = np.where(labels == -0.2)[0]
    num_bg_now = int(big_bg_inds.size + small_bg_inds.size)  # 现有的负样本总数
    num_fg = int(fg_inds.size)  # 现有的正样本总数
    # 应有的负样本总数
    num_bg = min(num_bg_now, ssd_cfg.train_ssd_batch_size - num_fg)
    # 先令负样本中80%都是big的
    big_bg_num = int(ssd_cfg.train_negative_big_ratio * num_bg)  # 应有的大后景负样本数
    # 若big总数满足比例，则从中抽取所需的80%个
    if big_bg_inds.size > big_bg_num:
        _inds = np.random.choice(big_bg_inds, big_bg_num, False)
        labels[_inds] = 0
    # 若big总数不满足比例，则全将其标记为负样本
    else:
        labels[big_bg_inds] = 0
    # 抽取的更小的负样本数目
    small_bg_num = int((1 - ssd_cfg.train_negative_big_ratio) * num_bg)  # 应有的小后景负样本数
    # 若更小的样本满足抽取数目，则抽取
    if small_bg_inds.size > small_bg_num:
        _inds = np.random.choice(small_bg_inds, small_bg_num, False)
        labels[_inds] = 0
    # 若不满足，则全标记为负样本
    else:
        labels[small_bg_inds] = 0
    # 当前负样本下标
    bg_inds = np.where(labels == 0)[0]
    # 若已抽取的负样本不满足数目要求，则从抽剩的后景样本中抽取至满足要求
    if bg_inds.size < num_bg:
        mixed_bg_inds = np.where((labels == -0.1) |
                                 (labels == -0.2))[0]
        _inds = np.random.choice(mixed_bg_inds, int(num_bg - bg_inds.size), False)
        labels[_inds] = 0
        # 最终选取的负样本下标
        bg_inds = np.where(labels == 0)[0]
    # 最后，将剩余的后景标记为不选中
    not_sample_inds = np.where((labels == -0.1) |
                               (labels == -0.2))[0]
    labels[not_sample_inds] = -1

    # 回归标签
    bbox_regs = _compute_reg(anchor_boxes, gt_infos[max_gt_id_per_box, :])

    # # 放回全部的锚框中
    # cls_label = _unmap(labels_onehot, num_total_anchor_boxes, inds_inside, fill=-1)
    # reg_label = _unmap(bbox_regs, num_total_anchor_boxes, inds_inside, fill=0)
    # 本身使用的就是全部锚框，不用放回
    cls_label = labels
    reg_label = bbox_regs

    return cls_label, reg_label


def _compute_reg(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    return bbox_transform(ex_rois, gt_rois)


def _unmap(data, count, inds, fill=0):
    """ Unmap x subset of item (x) back to the original set of items (of size count) """
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret
