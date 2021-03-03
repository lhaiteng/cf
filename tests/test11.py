# -*- coding: utf-8 -*-
"""
ssd的标签、掩码，以及损失函数构建的测试代码
掩码包括：
    样本掩码：[num_anchor_boxs, ] 用于分类误差，选择正负样本, {1, 0, -1}
    匹配掩码：[num_anchor_boxs, num_boxes, ] 用于回归误差， 标识是否与gt框匹配, {1, 0}
标签包括：
    分类标签 [num_anchor_boxs, num_cls+1] 默认0
    回归标签 [num_anchor_boxs, num_boxes, 4]  默认0
损失函数：
    分类损失：正样本锚框与每个相匹配的gt框类别都计算损失，负样本锚框正常计算。
    回归损失：正样本锚框与每个相匹配的gt框都计算smooth l1 loss，负样本锚框不计算。
"""
import math, itertools, random
import numpy as np
import matplotlib.pyplot as plt
import itertools
import tensorflow as tf

"""参数"""


class Config:
    def __init__(self):
        self.num_epoch = 1000  # 训练次数
        self.batch_size = 128  # 批次内样本总数
        self.pos_batch_size = 32  # 批次内的正样本数
        self.num_cls = 33  # 锚框类别数。不包括背景
        self.layers = (16, 16, 32)  # 每层通道数
        self.point_anchor_boxes = {'sizes': [4, 8, 16], 'ratios': [1, 0.5, 2]}
        self.point_anchor_boxes['num'] = len(self.point_anchor_boxes['sizes']) * len(self.point_anchor_boxes['ratios'])

        self.min_img_shape = 128  # 图片的最小尺寸
        self.max_img_shape = 300  # 图片的最大尺寸
        self.max_n_gt = 20  # 一张图的最大锚框数
        self.min_n_size = 8  # 最小锚框尺寸

        # 生成正负样本的阈值
        self.gen_pos_iou = 0.5
        self.gen_neg_iou = 0.4

        self.lr = 0.001


cfg = Config()

"""网络结构"""
try:
    sess.close()
except:
    pass


class Net:
    def __init__(self):
        tf.reset_default_graph()
        # 占位符
        self.training = tf.placeholder(tf.bool, [])
        self.inputs = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        # 标签
        self.labels_cls = tf.placeholder(tf.float32, shape=[None, cfg.num_cls + 1])  # 分类标签 [0, 1, 1, 0, 0, 0, ...]
        self.labels_reg = tf.placeholder(tf.float32, shape=[None, None, 4])  # 回归标签 [None, n_gt, 4]
        # 掩码
        self.mask_sa = tf.placeholder(tf.float32, shape=[None, ])  # 样本掩码{-1, 0, 1} [None, ]
        self.mask_mate = tf.placeholder(tf.float32, shape=[None, None, ])  # 匹配掩码{0, 1} [None, n_gt]

        opt = tf.train.GradientDescentOptimizer(cfg.lr)

        for ind, f in enumerate(cfg.layers):
            if ind == 0:
                x = tf.layers.conv2d(self.inputs, f, (3, 3), (1, 1), padding='same', use_bias=False)
            else:
                x = tf.layers.conv2d(x, f, (3, 3), (1, 1), padding='same', use_bias=False)
            x = tf.layers.batch_normalization(x, training=self.training)
            x = tf.layers.max_pooling2d(x, (2, 2), (2, 2), padding='same')
            x = tf.nn.relu(x)

        x = tf.layers.conv2d(x, f, (3, 3), (1, 1), padding='same', use_bias=False)
        x = tf.layers.batch_normalization(x, training=self.training)
        x = tf.nn.relu(x)

        # 得到分类
        cls = tf.layers.conv2d(x, (cfg.num_cls + 1) * cfg.point_anchor_boxes['num'], (1, 1), (1, 1), use_bias=False)
        # 得到回归
        reg = tf.layers.conv2d(x, 4 * cfg.point_anchor_boxes['num'], (1, 1), (1, 1), use_bias=False)

        # 重组
        self.cls_logits = tf.reshape(cls, [-1, cfg.num_cls + 1])  # [n_boxes, num_cls+1]
        self.regs = tf.reshape(reg, [-1, 1, 4])  # [num_boxes, 1, 4]

        self.loss_cls, self.loss_reg, self.loss = self.get_loss()
        self.init_op = tf.global_variables_initializer()

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.train_op = opt.minimize(self.loss)

    def get_loss(self):
        # 分类损失
        # 根据掩码选中样本
        sa_inds = tf.reshape(tf.where(tf.greater(self.mask_sa, -1.)), [-1])
        # 选中样本的标签和概率
        labels = tf.gather(self.labels_cls, sa_inds)
        cls_logits = tf.gather(self.cls_logits, sa_inds)
        # 得到损失
        cls_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels, cls_logits)
        cls_loss = tf.reduce_mean(cls_loss)
        tf.losses.add_loss(cls_loss)

        # 回归损失
        # 根据样本掩码选中正样本的匹配掩码、标签和预测回归值
        fg_inds = tf.reshape(tf.where(tf.greater(self.mask_sa, 0.)), [-1])
        fg_mask_mate = tf.gather(self.mask_mate, fg_inds)  # [num_boxes, n_gt]
        labels = tf.gather(self.labels_reg, fg_inds)  # [num_boxes, n_gt, 4]
        regs = tf.gather(self.regs, fg_inds)  # [num_boxes, 1, 4]

        reg_loss = self.get_smooth_l1(labels - regs)  # [num_boxes, n_gt]
        reg_loss = fg_mask_mate * reg_loss
        reg_loss = tf.reduce_mean(reg_loss)
        tf.losses.add_loss(reg_loss)

        loss = tf.add_n(tf.losses.get_losses())

        return cls_loss, reg_loss, loss

    def get_smooth_l1(self, x):
        # 注意此处的x是三维的 [num_boxes, n_gt, 4]
        loss = tf.where(tf.abs(x) < 1., 0.5 * tf.square(x), tf.abs(x) - 0.5)
        return tf.reduce_sum(loss, axis=-1)  # 输出[n_boxes, ng_gt]


ts = Net()
sess = tf.Session()

"""生成数据"""


class Samples:
    def __init__(self):
        self.should_gen_data = True
        self.data_generator = self.gen_data()

    # 数据生成器，包括图片、gt框、标签和掩码
    def gen_data(self):
        # 假设特征图单点对应的锚框信息都一致，生成单点对应的锚框w, h
        point_anchor_boxes = self.get_point_anchor_boxes(**cfg.point_anchor_boxes)

        while self.should_gen_data:
            # 图片数据
            img_h, img_w = np.random.randint(cfg.min_img_shape, cfg.max_img_shape + 1, size=[2])
            imgs = np.random.random(size=[1, img_h, img_w, 3])

            # gt框数据
            n_gt = 0
            while n_gt < 1:
                n_gt = np.random.randint(1, cfg.max_n_gt + 1)
                gt_boxes = self.get_gt_boxes(img_h, img_w, n_gt)  # r1, c1, r2, c2
                n_gt = gt_boxes.shape[0]
            gt_cls = np.random.randint(0, cfg.num_cls, [n_gt])  # gt框的类别

            # 所有锚框r1, c1, r2, c2
            total_anchor_boxes = self.get_total_anchor_boxes(img_h, img_w, point_anchor_boxes)

            # 锚框的掩码和标签
            mask_sa, mask_mate, labels_cls, labels_reg = self.get_mask_labels(gt_boxes, gt_cls, total_anchor_boxes)

            yield imgs, [gt_boxes, gt_cls], [mask_sa, mask_mate, labels_cls, labels_reg]

    # 假设特征图单点对应的锚框信息都一致，生成单点对应的锚框w, h
    def get_point_anchor_boxes(self, sizes, ratios, **kwargs):
        # 生成w, h
        srs = np.array(list(itertools.product(sizes, ratios)))  # [[s1, r1], [s1, r2], ...]
        point_anchor_boxes = np.empty_like(srs)
        point_anchor_boxes[:, 0] = srs[:, 0] * np.sqrt(srs[:, 1])
        point_anchor_boxes[:, 1] = srs[:, 0] / np.sqrt(srs[:, 1])
        return point_anchor_boxes

    # 根据图形大小获取所有锚框的坐标
    def get_total_anchor_boxes(self, img_h, img_w, point_anchor_boxes):
        # 假设网络是固定的，则得到的特征图对应的感受野信息也是固定的
        # len(cfg.layers)个c3s2+c3s1 -> c3s2+c3s2+c3s2+c3s1
        layers = ((3, 1, 1), (2, 0, 2),  # (k, p, s), c3s1+max_pool
                  (3, 1, 1), (2, 0, 2),
                  (3, 1, 1), (2, 0, 2),
                  (3, 1, 1))
        fm_h, fm_w, fm_j, fm_r, fm_START = self.get_fm_infos(img_h, img_w, layers)

        # 根据特征图和单点锚框信息，生成全部的锚框
        total_anchor_boxes = self.gen_total_anchor_boxes(fm_h, fm_w, fm_j, fm_r, fm_START, point_anchor_boxes)

        return total_anchor_boxes

    # 根据layers生成最终特征图的信息fm_h, fm_w, fm_j, fm_r, fm_START
    def get_fm_infos(self, img_h, img_w, layers):
        fm_h, fm_w = img_h, img_w
        fm_j, fm_r, fm_START = 1, 1, 0.5
        for k, p, s in layers:
            fm_h = math.floor((fm_h + 2 * p - k) / s) + 1
            fm_w = math.floor((fm_w + 2 * p - k) / s) + 1
            fm_j, fm_r, fm_START = fm_j * s, fm_r + (k - 1) * fm_j, ((k - 1) / 2 - p) * fm_j
        return fm_h, fm_w, fm_j, fm_r, fm_START

    # 根据特征图和单点锚框信息，生成全部的锚框
    def gen_total_anchor_boxes(self, fm_h, fm_w, fm_j, fm_r, fm_START, point_anchor_boxes):
        # 所有锚框中心点。笛卡尔积
        # shift_r =
        # [[0, 0, 0, ...],
        #  [j, j, j, ...],
        #  [2j, 2j, 2j, ...],...]
        # shift_c =
        # [[0, j, 2j, ...],
        #  [0, j, 2j, ...],
        #  [0, j, 2j, ...],...]
        center_shifts = np.array(list(itertools.product(range(fm_h), range(fm_w))))  # r, c
        center_shifts = center_shifts[:, ::-1]  # x, y
        centers = center_shifts * fm_j + fm_START
        # 加入单锚框的w、h。笛卡尔积
        boxes = np.reshape(list(itertools.product(centers, point_anchor_boxes)), [-1, 4])
        # # x, y, w, h

        # 得到r1, c1, r2, c2。
        # w = c2 - c1 + 1, x = c1 + w/2
        new_boxes = np.empty_like(boxes)
        new_boxes[:, 1] = boxes[:, 0] - boxes[:, 2] / 2  # c1 = x - w / 2
        new_boxes[:, 3] = boxes[:, 2] + new_boxes[:, 1] - 1  # c2 = w + c1 - 1
        new_boxes[:, 0] = boxes[:, 1] - boxes[:, 3] / 2
        new_boxes[:, 2] = boxes[:, 3] + new_boxes[:, 0] - 1

        return new_boxes.astype(int)

    # 生成掩码和标签
    def get_mask_labels(self, gt_boxes, gt_cls, anchor_boxes):
        # 返回数据的格式
        mask_sa = np.empty(anchor_boxes.shape[0])
        mask_sa.fill(-1)
        mask_mate = np.zeros([anchor_boxes.shape[0], gt_boxes.shape[0]])
        labels_cls = np.zeros([anchor_boxes.shape[0], cfg.num_cls + 1])
        labels_reg = self.transform_bbox(anchor_boxes, gt_boxes)  # [n_boxes, n_gt, 4]

        # 求iou
        ious = self.get_all_ious(anchor_boxes, gt_boxes)  # [n_boxes, n_gt]

        # 必选正样本: 每个最大iou
        _pos_inds = np.argmax(ious, axis=0)
        mask_sa[_pos_inds] = 1
        mask_mate[_pos_inds, np.arange(gt_boxes.shape[0])] = 1
        labels_cls[_pos_inds, gt_cls + 1] = 1

        # 候选正样本
        _num_pos = cfg.pos_batch_size - _pos_inds.size  # 剩余数目
        _for_pos_inds = np.where(np.max(ious, axis=1) >= cfg.gen_pos_iou)[0]
        _num_pos = np.minimum(_for_pos_inds.size, _num_pos)  # 防止候选正样本不足
        if _num_pos > 0 and _for_pos_inds.size:
            _pos_inds = np.random.choice(_for_pos_inds, _num_pos, False)
            mask_sa[_pos_inds] = 1
            _r_inds, _c_inds = np.where(ious[_pos_inds] >= cfg.gen_pos_iou)
            mask_mate[_pos_inds[_r_inds], _c_inds] = 1
            labels_cls[_pos_inds[_r_inds], np.asarray(gt_cls + 1)[_c_inds]] = 1

        # 负样本
        _num_neg = cfg.batch_size - np.sum(mask_sa > 0)
        _for_neg_inds = np.where(np.max(ious, axis=1) < cfg.gen_neg_iou)[0]
        if _num_neg > 0:
            _neg_inds = np.random.choice(_for_neg_inds, _num_neg, False)
            mask_sa[_neg_inds] = 0
            labels_cls[_neg_inds, 0] = 1

        return mask_sa, mask_mate, labels_cls, labels_reg

    # 得到锚框与gt的回归标签 [n_boxes, n_gt, 4]
    def transform_bbox(self, anchor_boxes, gt_boxes):
        anchor_boxes = np.reshape(anchor_boxes, [-1, 1, 4])  # [n_boxes, 1, 4]
        gt_boxes = np.reshape(gt_boxes, [1, -1, 4])  # [1, n_gt, 4]

        ab_w = anchor_boxes[:, :, 3] - anchor_boxes[:, :, 1] + 1
        ab_h = anchor_boxes[:, :, 2] - anchor_boxes[:, :, 0] + 1
        ab_x = anchor_boxes[:, :, 1] + ab_w / 2
        ab_y = anchor_boxes[:, :, 0] + ab_h / 2
        gt_w = gt_boxes[:, :, 3] - gt_boxes[:, :, 1] + 1
        gt_h = gt_boxes[:, :, 2] - gt_boxes[:, :, 0] + 1
        gt_x = gt_boxes[:, :, 1] + gt_w / 2
        gt_y = gt_boxes[:, :, 0] + gt_h / 2

        tx = (gt_x - ab_x) / ab_w  # [n_boxes, n_gt]
        ty = (gt_y - ab_y) / ab_h  # [n_boxes, n_gt]
        tw = np.log(gt_w / ab_w)  # [n_boxes, n_gt]
        th = np.log(gt_h / ab_h)  # [n_boxes, n_gt]

        return np.concatenate((tx[:, :, np.newaxis], ty[:, :, np.newaxis], tw[:, :, np.newaxis], th[:, :, np.newaxis]),
                              axis=-1)

    # 根据标签还原boxes [n_boxes, 4]
    def transform_bbox_inv(self, boxes, labels):
        bw = boxes[:, 3] - boxes[:, 1] + 1
        bh = boxes[:, 2] - boxes[:, 0] + 1
        bx = boxes[:, 1] + bw / 2
        by = boxes[:, 0] + bh / 2

        tx, ty, tw, th = labels[:, 0], labels[:, 1], labels[:, 2], labels[:, 3]

        nbw, nbh = bw * np.exp(tw), bh * np.exp(th)
        nbx, nby = bx + tx * bw, by + ty * bh

        nbc1 = nbx - nbw / 2
        nbc2 = nbx + nbw / 2 + 1
        nbr1 = nby - nbh / 2
        nbr2 = nby + nbh / 2 + 1
        return np.column_stack((nbr1, nbc1, nbr2, nbc2))

    # 产生下一个批次的数据
    def next_batch(self):
        return next(self.data_generator)

    # 多个boxes对多个boxes的iou
    def get_all_ious(self, anchor_boxes, gt_boxes):
        ious = []

        for box in gt_boxes:
            iou = self.get_ious(anchor_boxes, box)
            ious.append(iou)
        return np.column_stack(ious)

    # 多个boxes对一个box的iou
    def get_ious(self, boxes, box):
        r1, c1, r2, c2 = box
        br1, bc1, br2, bc2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        ious = self._overlap(br1, br2, r1, r2) * self._overlap(bc1, bc2, c1, c2)
        return ious

    # 线段x1,x2与线段y1, y2重合长度
    def _overlap(self, x1, x2, y1, y2):
        return self._relu(x2 - x1 - self._relu(y1 - x1) - self._relu(x2 - y2))

    def _relu(self, x):
        return np.maximum(x, 0)

    # 生成gt框r1, c1, r2, c2
    def get_gt_boxes(self, img_h, img_w, n_gt):
        r1s = np.random.randint(0, img_h, [n_gt])
        c1s = np.random.randint(0, img_w, [n_gt])
        r2s = np.floor(r1s + np.random.random([n_gt]) * (img_h - r1s - 1))
        c2s = np.floor(c1s + np.random.random([n_gt]) * (img_w - c1s - 1))
        keep_inds = np.ones([n_gt])
        keep_inds[r2s - r1s < cfg.min_n_size] = 0
        keep_inds[c2s - c1s < cfg.min_n_size] = 0
        gt_infos = np.column_stack((r1s, c1s, r2s, c2s))
        return gt_infos[keep_inds > 0, :]

    def close(self):
        self.should_gen_data = False


sa = Samples()

"""验证是否正确"""

datas = sa.next_batch()

imgs, [gt_boxes, gt_cls], [mask_sa, mask_mate, labels_cls, labels_reg] = datas

# 如何验证？
"""训练"""

sess.run(ts.init_op)
_num_epoch = cfg.num_epoch
for ind_epoch in range(_num_epoch):
    datas = sa.next_batch()
    imgs, [gt_boxes, gt_cls], [mask_sa, mask_mate, labels_cls, labels_reg] = datas
    run_list = [ts.train_op, ts.loss_cls, ts.loss_reg, ts.loss]
    feed_dict = {ts.inputs: imgs, ts.labels_cls: labels_cls, ts.labels_reg: labels_reg,
                 ts.training: True,
                 ts.mask_sa: mask_sa, ts.mask_mate: mask_mate}
    _, loss_cls, loss_reg, loss = sess.run(run_list, feed_dict)
    print(f'\rEPOCH {ind_epoch}/{_num_epoch} LOSS cls={loss_cls:.3f} reg={loss_reg:.3f} loss={loss:.3f}...', end='')

"""训练结果"""
