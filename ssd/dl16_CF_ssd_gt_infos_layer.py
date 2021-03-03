# -*- coding: utf-8 -*-
from lxml import etree
import numpy as np


def gt_infos_layer(anno_path, cla_id, resize_ratio=1):
    """
    训练时从样本说明文件可直接得到：
    gt_info [num_boxes, 5] -> class, r1, c1, r2, c2
    :param anno_path: 说明文件路径
    :param cla_id: 分类字典 {cla1: id1, cla1: id1, ...}
    :param resize_ratio: 长宽方向同时放缩的比例
    :return: gt_info [num_boxes, 5] -> class, r1, c1, r2, c2
    """
    with open(anno_path, 'r', encoding='utf8') as f:
        f = f.read()
        html = etree.HTML(f)
    names = html.xpath('//object/name/text()')
    x1s = html.xpath('//object/bndbox/xmin/text()')
    y1s = html.xpath('//object/bndbox/ymin/text()')
    x2s = html.xpath('//object/bndbox/xmax/text()')
    y2s = html.xpath('//object/bndbox/ymax/text()')

    gt_clas, gt_locs = [], []
    for i, name in enumerate(names):
        if name not in cla_id:
            print(f'{name} not in {cla_id}')
            continue
        gt_clas.append(cla_id[name])
        gt_locs.append([int(y1s[i]), int(x1s[i]), int(y2s[i]), int(x2s[i])])
    if resize_ratio != 1:
        for i, gt_loc in enumerate(gt_locs):
            gt_locs[i] = get_resized_gt(gt_loc, resize_ratio)

    gt_infos = np.c_[gt_clas, gt_locs]

    return gt_infos


def get_resized_gt(gt_loc, resize_ratio, img_shape):
    h, w = img_shape
    r1, c1, r2, c2 = gt_loc

    r1 = r1 * resize_ratio + resize_ratio / 2
    c1 = c1 * resize_ratio + resize_ratio / 2
    r2 = r2 * resize_ratio + resize_ratio / 2
    c2 = c2 * resize_ratio + resize_ratio / 2

    r1, r2 = min(r1, r2), max(r1, r2)
    c1, c2 = min(c1, c2), max(c1, c2)

    r1, c1 = max(r1, 0), max(c1, 0)
    r2, c2 = min(r2, h - 1), min(c2, w - 1)

    return r1, c1, r2, c2
