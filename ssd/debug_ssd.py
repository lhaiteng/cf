# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import time, cv2, os
import tensorflow as tf
from ssd.dl16_CF_ssd_app import App
from ssd.dl16_CF_ssd_config import ssd_cfg
from ssd.dl16_CF_ssd_tensors import Tensors
from ssd.dl16_CF_ssd_samples import Samples
from ssd.dl16_CF_ssd_utils import time_calc, \
    get_all_anchor_boxes, bbox_transform_inv, self_nms, get_iou, \
    plot_PR_from_is_tps


class Debug:
    def __init__(self, app: App):
        self.app = app

    def run(self):
        """
        制作不同iou_thres下的ROC曲线
        """

        datasets = 'train'
        mAP_iou_thres_list, scores, is_tps, n_gt, total_P, TP = self.restore_TP(datasets)
        recalls, precisions = TP / n_gt, TP / np.maximum(total_P, 1e-6)

    # 恢复保存的数据
    def restore_TP(self, datasets='train'):
        dir = 'E:/TEST/ChangeFace/ssd/record/'

        save_file1 = f'./record/{datasets}_mAP_iou_thres_list.npy'
        mAP_iou_thres_list = np.load(save_file1)

        save_file1 = f'{dir}{datasets}_scores.npy'
        save_file2 = f'{dir}{datasets}_is_tps.npy'
        scores = np.load(save_file1)
        is_tps = np.load(save_file2)

        save_file1 = f'{dir}{datasets}_n_gt.npy'
        save_file2 = f'{dir}{datasets}_total_P.npy'
        save_file3 = f'{dir}{datasets}_TP.npy'
        n_gt = np.load(save_file1)
        total_P = np.load(save_file2)
        TP = np.load(save_file3)

        return mAP_iou_thres_list, scores, is_tps, n_gt, total_P, TP

    # 随机挑选一张图片画出其PR曲线
    def plot_PR_one_sa(self, iou_thres_list, score_thres_list, datasets='train'):
        # 随机选择一张图片，得到候选预测框的得分、矫正后的结果，并返回gt框位置。
        # 已按照score排序并NMS。
        imgs, gt_infos, scores, boxes = self.get_one_res(datasets)
        # 接下来需要加入返回is_tps [num_iou_thres, num_boxes]
        is_tps = self.get_is_tps(scores, boxes, gt_infos, iou_thres_list)
        # 根据tps画出PR曲线
        n_gt = gt_infos.shape[0]
        plot_PR_from_is_tps(n_gt, scores, is_tps, score_thres_list=score_thres_list, iou_thres_list=iou_thres_list)
        # 原图gt框
        img = imgs[0].copy()
        for gt in gt_infos:
            r1, c1, r2, c2 = gt.astype(int)
            img = cv2.rectangle(img, (c1, r1), (c2, r2), [0, 0, 255], thickness=2)
        plt.figure(figsize=[10, 10])
        plt.imshow(img[:, :, ::-1])
        plt.title('boxes')
        plt.show()
        # 画几个试一下
        img1 = img.copy()
        for ind, box in enumerate(boxes[:10]):
            r1, c1, r2, c2 = box.astype(int)
            img1 = cv2.rectangle(img1, (c1, r1), (c2, r2), [0, 255, 0], thickness=1)
            if is_tps[1, ind] > 0:  # 如果满足最低阈值，则画出得分
                text = f'score:{scores[ind]:.3f}'
                img1 = cv2.putText(img1, text, (c1 + 10, r1 + 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 255, 0], thickness=1)
        plt.figure(figsize=[10, 10])
        plt.imshow(img1[:, :, ::-1])
        plt.show()

    # 返回不同iou下是否为tp
    @time_calc('get_is_tps')
    def get_is_tps(self, scores, boxes, gt_infos, iou_thres_list):
        """
        其中scores和boxes已按照score从大到小排序，并已NMS过
        :param scores:
        :param boxes:
        :param gt_infos:
        :param iou_thres_list:
        :return: [num_iou_thres, num_boxes]
        """
        num_boxes = scores.size
        num_iou_thres = len(iou_thres_list)
        num_gt = gt_infos.shape[0]
        is_tps = np.zeros(shape=[num_iou_thres, num_boxes])

        for ind_iou_thres, iou_thres in enumerate(iou_thres_list):
            ind_gt = list(range(num_gt))
            for ind_box, box in enumerate(boxes):
                if not ind_gt: break
                ious = get_iou(gt_infos[ind_gt], box)
                max_iou, argmax = np.max(ious), int(np.argmax(ious))
                if max_iou >= iou_thres:
                    is_tps[ind_iou_thres, ind_box] = 1
                    ind_gt.pop(argmax)
        return is_tps

    # 获取单张随机图片的得到候选预测框的得分、矫正后的结果，并返回gt框位置。已按照score排序并NMS。
    @time_calc('get_one_res')
    def get_one_res(self, datasets):
        app = self.app
        ts = app.tensors.sub_ts[0]
        # 获取样本
        imgs, gt_infos, cla_labels, reg_labels = app.samples.next_batch(datasets=datasets)
        # 对图片进行预测
        run_list = [ts.cla_probs, ts.regs]
        feed_dict = {ts.training: False, ts.x: imgs}
        scores, regs = app.session.run(run_list, feed_dict)

        # 先只保留得分>=0.5的
        boxes = get_all_anchor_boxes(imgs.shape[1:3])
        keep_inds = np.where(scores >= 0.5)[0]
        boxes, scores, regs = boxes[keep_inds], scores[keep_inds], regs[keep_inds]

        # 对锚框进行矫正
        boxes = bbox_transform_inv(boxes, regs)
        # 按得分排序
        ind = np.argsort(scores)[::-1]
        scores, boxes = scores[ind], boxes[ind]
        # NMS
        keep_inds = self_nms(boxes, 0.7)
        scores, boxes = scores[keep_inds], boxes[keep_inds]

        return imgs, gt_infos, scores, boxes

    def close(self):
        self.app.close()


if __name__ == '__main__':
    app = App()

    debug = Debug(app)
    debug.run()

    debug.close()

    print('Finished!')
