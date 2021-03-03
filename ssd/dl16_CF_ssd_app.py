"""
ssd
"""
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import time, cv2, os, json
import tensorflow as tf
from ssd.dl16_CF_ssd_config import ssd_cfg
from ssd.dl16_CF_ssd_tensors import Tensors
from ssd.dl16_CF_ssd_samples import Samples
from ssd.dl16_CF_ssd_utils import get_all_anchor_boxes, bbox_transform_inv, self_nms, get_result, \
    get_is_tps, plot_PR, get_APs


class App:
    def __init__(self):
        self.samples = Samples()
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.tensors = Tensors()
            conf = tf.ConfigProto()
            conf.allow_soft_placement = True
            self.session = tf.Session(config=conf)
            self.saver = tf.train.Saver()
            try:
                self.saver.restore(self.session, ssd_cfg.save_path)
                print(f'Restore model from {ssd_cfg.save_path} succeed.')
            except:
                self.session.run(tf.global_variables_initializer())
                print(f'Restore model from {ssd_cfg.save_path} failed.')
        print(f'anchor_box_scales: {ssd_cfg.anchor_scales}')

    def train(self, batch_size=1):
        samples = self.samples
        print('=' * 100)
        for att in dir(ssd_cfg):
            value = getattr(ssd_cfg, att)
            if type(value) in (float, int, str, list, tuple, dict, set) and not att.startswith('_'):
                print(f'{att}: {value}')
        print('=' * 100)
        print(f'total train num: {samples.train_num}.')
        print(f'total test num: {samples.test_num}.')

        epochs = ssd_cfg.train_epochs

        # 一个epoch有多少batch
        num_batches = samples.train_num // (batch_size * ssd_cfg.train_gpu_num)
        print(f'train {num_batches} batches / epoch.')
        # 经过多少batch显示一次损失
        train_num_to_print_losses = ssd_cfg.train_num_batch_to_print_losses
        print(f'show loss / {train_num_to_print_losses} batches.')
        # 经过多少batch显示一次预测
        show_num = ssd_cfg.train_show_num_rate
        print(f'show predict / {num_batches} batches.')

        ts = self.tensors
        feed_dict = {ts.lr: ssd_cfg.train_lr, ts.training: True}
        run_list = [ts.global_step, ts.train_op,
                    ts.loss_cla, ts.loss_reg]

        print('Start training...')
        step = 0  # 保证新开始训练的第一个batch，都打印loss和显示预测
        for epoch in range(ssd_cfg.train_start_num_epoch, epochs):
            time1 = time.time()
            for batch in range(num_batches):
                # 喂入各gpu值
                for gpu_index in range(ssd_cfg.train_gpu_num):
                    sub_ts = ts.sub_ts[gpu_index]
                    imgs, gt_infos, cla_labels, reg_labels = samples.next_batch(batch_size)
                    feed_dict[sub_ts.x] = imgs
                    # feed_dict[sub_ts.gt_infos] = gt_infos
                    feed_dict[sub_ts.cla_labels] = cla_labels
                    feed_dict[sub_ts.reg_labels] = reg_labels

                # 每过几次打印一代损失
                if not train_num_to_print_losses or step % train_num_to_print_losses == 0:
                    step, _, loss_cla, loss_reg = self.session.run(run_list, feed_dict)
                    print(f'\rEpoch {epoch + 1}/{epochs}  Batch {batch + 1}/{num_batches}'
                          f' - Loss cla = {loss_cla:.3f} reg = {loss_reg:.3f}',
                          end='')
                else:
                    self.session.run(ts.train_op, feed_dict)

                # 整除时显示预测一次
                if show_num and step % show_num == 0:
                    self.predict(save_pic=False)

            time2 = time.time()
            # 每代结束后，保存模型
            self.saver.save(self.session, ssd_cfg.save_path)
            print(f'\tSave into model {ssd_cfg.save_path}')
            print(f'Training Epoch {epoch + 1} costs time: {(time2 - time1) / 60:.2f}min')
            # 每代结束后，进行预测并保存预测图片，但不显示
            self.predict(show_pic=False, save_pic=True, save_pic_name=f'E{epoch}')

            time.sleep(1)

    def predict(self, img=None, src='test', top_boxes=0, show_pic=True,
                save_pic=True, save_dir=None, save_pic_name=None,
                return_pic=False, return_face_locs=False):
        """
        :param img:
        :param src:
        :param top_boxes: 显示得分最高的几个。默认0是全部显示。
        :param show_pic:
        :param save_pic:
        :param save_dir:
        :param save_pic_name: 不带后缀
        :param return_pic:
        :param return_face_locs:
        :return: 若设置有返回参数，则返回字典：{'pic': img, 'face_locs': boxes, ...}
        """
        return_datas = {}
        # 获取数据
        if img is None:
            imgs, _, _, _ = self.samples.next_batch(datasets=src)
        else:
            imgs = np.expand_dims(img, axis=0) if len(img.shape) < 4 else img
        # 使用网络预测
        ts = self.tensors
        sub_ts = ts.sub_ts[0]
        feed_dict = {ts.training: False, sub_ts.x: imgs}
        run_list = [sub_ts.ob_scores, sub_ts.ob_boxes]
        ob_scores, ob_boxes = self.session.run(run_list, feed_dict)
        # # [-1, 4]  r1, c1, r2, c2
        img = imgs[0].copy()
        # 在图中标记
        if top_boxes:
            ob_scores = ob_scores[:top_boxes]
            ob_boxes = ob_boxes[:top_boxes]
        for score, box in zip(ob_scores, ob_boxes):
            r1, c1, r2, c2 = box.astype(int)
            img = cv2.rectangle(img, (c1, r1), (c2, r2), [0, 0, 255], ssd_cfg.box_linewidth)
            text = f'FACE: {score:.3f}'
            img = cv2.putText(img, text, (c1 + 10, r1 + 10),
                              ssd_cfg.font, ssd_cfg.fontScale, [0, 0, 255], ssd_cfg.font_width)
        # 若显示图像
        if show_pic:
            plt.imshow(img[:, :, ::-1])
            plt.show()
        # 保存图像
        if save_pic:
            if save_dir is None:
                save_dir = ssd_cfg.gen_dir
            elif save_dir[-1] != '/':
                save_dir += '/'
            if save_pic_name is None: save_pic_name = ssd_cfg.name
            cv2.imwrite(save_dir + save_pic_name + f'.png', img)
        # 返回标注后的图像，cv2格式
        if return_pic:
            return_datas['pic'] = img
        # 返回图像中的坐标
        if return_face_locs:
            return_datas['face_locs'] = ob_boxes
        # 若有返回数据
        if return_datas:
            return return_datas

    def predict_from_dir(self, img_dir, show_pic=True, save_pic=False, show_num=0, **kwargs):
        for pic_name in os.listdir(img_dir):
            pic_path = img_dir + pic_name
            if show_num: pic_path = pic_path[:show_num]
            if os.path.isfile(pic_path):
                img = cv2.imread(pic_path)
                if img is None:
                    img = cv2.imdecode(np.fromfile(pic_path, dtype=np.uint8), -1)
                min_line = min(img.shape[:2])
                resize_ratio = max(ssd_cfg.min_img_size / min_line, 1)
                if resize_ratio > 1:
                    img = cv2.resize(img, (0, 0), fx=resize_ratio, fy=resize_ratio)
                kwargs['save_name'] = pic_name[:-4]
                app.predict(img=img, show_pic=show_pic, save_pic=save_pic, **kwargs)

    def close(self):
        self.session.close()
        self.samples.close()

    def write_face_loc(self, pic_dir, save_path, **kwargs):
        """
        根据图片文件夹，输出图片脸部坐标文件{pic_path:face_loc[[r1, c1, r2, c2], ...], ...}
        :param pic_dir:
        :param save_path:
        :return:
        """
        path_locs = {}
        print(f'START record path_loc dict from {pic_dir}...')
        for pic_name in os.listdir(pic_dir):
            path = pic_dir + pic_name
            if os.path.isfile(path):
                print(f'\rprocessing {path}...', end='')
                # 读取图片，并放缩！！很重要！！！
                img = cv2.imread(path)
                if img is None:
                    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
                if img is None:
                    os.remove(path)
                    print(f'\nremove {path}.')
                    continue
                img, _ = self.samples.resize_img(img, ssd_cfg.max_img_size, ssd_cfg.min_img_size)
                # face_locs = app.predict(img=img, show_pic=False, save_pic=False, return_face_locs=True, **kwargs)
                _kwargs = {'img': img, 'show_pic': False, 'save_pic': False, 'return_face_locs': True}  # 默认的参数
                _kwargs.update(kwargs)  # 新添及更新的参数
                face_locs = app.predict(**_kwargs)
                path_locs[path] = face_locs['face_locs'].tolist()
        print()
        print(f'START dumping path_loc into string...')
        js_str = json.dumps(path_locs)
        print(f'START writing string into {save_path}...')
        with open(save_path, 'w') as f:
            f.writelines(js_str)
        print(f'FINISH writing into {save_path}.')

    def get_mAP(self, datasets='train', save_record=True):
        """
        所有锚框的分类得分、回归值。
        由于本模型只是人脸识别，且用的ssd模型，所以得分是sigmoid得分，
        没有使用多分类的softmax，也无法得到所有的候选预测框。因为当阈值是0或很小时，一张图片的所有锚框都是候选预测框。
        所以只能变通一下，假定得分>=0.5时为前景。
        本模型需要统计的数据中：
            n_gt是固定常数，total_P[n_scores]， TP[n_ious, n_scores]
        实现设定好score_thres_list，这样遍历每张图片时就能累加更新需要统计的数据，不需要保存所有图片后统一计算了。
        :param datasets:
        :param save_record: 是否保存数据
        :return:
        """

        """得到数据集的gt数量、不同score阈值下的P数量、不同iou阈值score阈值下的TP数量"""
        mAP_iou_thres_list = [0.3, 0.5, 0.75]
        if save_record:
            save_file1 = f'./record/{datasets}_mAP_iou_thres_list.npy'
            np.save(save_file1, mAP_iou_thres_list)

        # 不设定score列表，提取所有的图片候选预测框
        # 遍历数据集，得到所有图片中候选预测框的scores, is_tps
        n_gt, scores, is_tps, n_pred = self.get_all_is_tps(mAP_iou_thres_list, datasets=datasets)
        # 保存计算得到的n_gt、scores、is_tps
        print(f'datasets {datasets} - num of pred boxes: {n_pred}.')
        if save_record:
            save_file1 = f'./record/{datasets}_n_gt.npy'
            save_file2 = f'./record/{datasets}_scores.npy'
            save_file3 = f'./record/{datasets}_is_tps.npy'
            np.save(save_file1, n_gt)
            np.save(save_file2, scores)
            np.save(save_file3, is_tps)
        # 得到total_P和TP
        total_P, TP = self.get_P_TP_from_is_tps(is_tps, scores=scores)
        # 保存计算得到的n_gt、total_P、TP
        if save_record:
            save_file1 = f'./record/{datasets}_total_P.npy'
            save_file2 = f'./record/{datasets}_TP.npy'
            np.save(save_file1, total_P)
            np.save(save_file2, TP)

        # # 提前设定score阈值列表，想遍历图片过程中直接累积P、TP
        # score_thres_list = np.linspace(0.5, 1, 1000)[::-1].tolist()
        # n_gt, total_P, TP, n_pred = self.get_T_TP_with_score_thres_list(score_thres_list, mAP_iou_thres_list, datasets=datasets)
        # print(f'{datasets} - num of pred boxes: {n_pred} .')
        # # 保存计算得到的n_gt、total_P、TP
        # if save_record:
        #     save_file1 = f'./record/{datasets}_n_gt.npy'
        #     save_file2 = f'./record/{datasets}_total_P.npy'
        #     save_file3 = f'./record/{datasets}_TP.npy'
        #     np.save(save_file1, n_gt)
        #     np.save(save_file2, total_P)
        #     np.save(save_file3, TP)

        """计算各类别下的recalls, precisions。"""
        recalls, precisions = TP / n_gt, TP / np.maximum(total_P, 1e-6)

        """逐一计算各类别下的AP。不考虑类别，应是一维的[iou阈值数]"""
        APs = get_APs(recalls, precisions)

        # 画出PR曲线
        plot_PR(recalls, precisions, mAP_iou_thres_list, APs, title=f'{datasets} PR-curve')

        """计算不同iou阈值下的mAP"""
        mAP = np.mean(APs)

        return APs, mAP

    # 不设定score列表时，遍历数据集，得到所有图片中候选预测框的n_gt, scores, is_tps, n_pred
    def get_all_is_tps(self, mAP_iou_thres_list=None, datasets='train'):
        """
        如果没有提前设定score阈值列表，则遍历数据集，得到所有数据的scores、is_tps，并返回
        :param mAP_iou_thres_list:
        :param datasets:
        :return:
            n_gt, scores, is_tps
        """
        sa = self.samples
        ts = self.tensors
        sub_ts = ts.sub_ts[0]
        feed_dict = {ts.training: False}
        run_list = [sub_ts.cla_probs, sub_ts.regs]

        """选定一组iou阈值，并准备好需要存储的数据格式"""
        # 提前设定的参数
        if mAP_iou_thres_list is None: mAP_iou_thres_list = [0.3]
        # 需要累积统计的数据
        n_gt = 0
        all_scores, all_is_tps = [], []
        n_pred = 0  # 所有图片的候选预测框之和

        """逐一计算各类别下的is_tps。不考虑类别，应是二维的[n_iou_thres，n_pred]"""
        total_num = sa.train_num if datasets == 'train' else sa.test_num
        # total_num = 10
        for _no_ in range(total_num):
            print(f'\rprocessing {datasets} pic: {_no_}/{total_num} n_pred: {n_pred}...', end='')
            imgs, gt_infos, cla_labels, reg_labels = sa.next_batch(1, datasets=datasets)
            feed_dict[sub_ts.x] = imgs
            scores, regs = self.session.run(run_list, feed_dict)

            # 得分大于0.5的作为候选预测框。
            # 步骤：生成所有锚框-挑出>0.5的作为前景-调整锚框-保留尺寸大于阈值的-按得分排序-NMS
            # 注意若无>0.5或无保留尺寸大于阈值的，则scores和boxes是空
            scores, pred_boxes = get_result(imgs.shape[1:3], scores, regs, score_thres=0.5)
            n_pred += scores.shape[0]

            # 得到各阈值下本张图片锚框的is_tps [num_iou_thres, num_box]
            # 输入的pred_boxes已经是按得分排序好的
            is_tps = get_is_tps(gt_infos, pred_boxes, mAP_iou_thres_list)

            # 累积gt
            n_gt += gt_infos.shape[0]
            # 累积all_scores和all_is_tps
            all_scores.extend(scores)
            all_is_tps.extend(is_tps.T)  # 若使用np.column_stack非常消耗时间。
        all_scores = np.array(all_scores)
        all_is_tps = np.transpose(all_is_tps)  # 转置回[n_iou_thres, n_boxes]
        print(f'\nFINISH get all_is_tps from datasets {datasets} {total_num} pics .')

        # 按照得分排序
        new_inds = np.argsort(all_scores)[::-1]
        all_scores = all_scores[new_inds]
        all_is_tps = all_is_tps[:, new_inds]

        return n_gt, all_scores, all_is_tps, n_pred

    def get_P_TP_from_is_tps(self, is_tps, scores=None, score_thres_list=None):
        """
        从is_tps累积获取T, TP。
        优先使用score_thres_list进行划分。
        :param is_tps: [n_iou_thres, n_boxes]
        :param scores: [n_boxes]
        :param score_thres_list:
        :return:
            total_P [n_score_thres]
            TP [n_iou_thres, n_score_thres]
        """
        if score_thres_list is not None:
            return
        if scores is not None:
            total_P = np.zeros_like(scores)
            TP = np.zeros_like(is_tps)
            for i, score in enumerate(scores):
                total_P[i] += 1
                TP[:, i] += is_tps[:, i]
                if i > 0:
                    total_P[i] += total_P[i - 1]
                    TP[:, i] += TP[:, i - 1]
            return total_P, TP

        raise KeyError('ERROR: there must be one not None in (scores, score_thres_list).')

    # 提前设定score阈值列表，在遍历图片过程中直接累积P、TP
    def get_T_TP_with_score_thres_list(self, score_thres_list, mAP_iou_thres_list=None, datasets='train'):
        """
        实现设定好score阈值列表的情况下，每一次图片，累积一次数据，
        遍历数据集后直接得到所有图片的n_gt, total_P [n_scores], TP[n_ious, n_scores]。
        由于本模型只是人脸识别，且用的ssd模型，所以得分是sigmoid得分，
        没有使用多分类的softmax，也无法得到所有的候选预测框。因为当阈值是0或很小时，一张图片的所有锚框都是候选预测框。
        所以只能变通一下，假定得分>=0.5时为候选预测框。
        实现设定好score_thres_list，这样遍历每张图片时就能累加更新需要统计的数据，不需要保存所有图片后统一计算了。
        :param score_thres_list: score阈值列表
        :param mAP_iou_thres_list: 若是None，则默认为0.3
        :param datasets:
        :return:
            n_gt: gt框总数
            total_P: [n_scores] 不同score阈值下的预测框数量
            TP: [n_ious, n_scores]  不同iou阈值、score阈值下的TP数量
            n_pred: 所有图片的候选预测框之和
        """
        sa = self.samples
        ts = self.tensors
        sub_ts = ts.sub_ts[0]
        feed_dict = {ts.training: False}
        run_list = [sub_ts.cla_probs, sub_ts.regs]

        """选定一组iou阈值，并准备好需要存储的数据格式"""
        # 提前设定的参数
        if mAP_iou_thres_list is None: mAP_iou_thres_list = [0.3]
        n_iou_thres = len(mAP_iou_thres_list)
        n_score_thres = len(score_thres_list)
        # 需要累进统计的数据
        n_gt = 0
        total_P = np.zeros(shape=[n_score_thres])  # 表示所有score阈值下的总P数
        TP = np.zeros(shape=[n_iou_thres, n_score_thres])
        n_pred = 0  # 所有图片的候选预测框之和

        """应当记录所有图片的候选预测框is_tps(与本图片中相同class的gt框求iou)
        all_pred_boxes = [class, score, is_tp1, is_tp2, ...]"""
        # 本步统计主要是为了得到score的阈值分布，若提前指定score阈值则可以直接得到本图中各类别的总P和TP。
        # 本模型跳过。

        """逐一计算各类别下的is_tps。不考虑类别，应是二维的[iou阈值数，scores阈值数]"""
        # 由于已经设定了score阈值，直接累加n_gt, total_P, TP
        total_num = sa.train_num if datasets == 'train' else sa.test_num
        # total_num = 10
        for _no_ in range(total_num):
            print(f'\rprocessing {datasets} pic: {_no_}/{total_num} n_pred: {n_pred}...', end='')
            imgs, gt_infos, cla_labels, reg_labels = sa.next_batch(1, datasets=datasets)
            feed_dict[sub_ts.x] = imgs
            scores, regs = self.session.run(run_list, feed_dict)

            # 得分大于0.5的作为候选预测框。
            # 步骤：生成所有锚框-挑出>0.5的作为前景-调整锚框-保留尺寸大于阈值的-按得分排序-NMS
            # 注意若无>0.5或无保留尺寸大于阈值的，则scores和boxes是空
            scores, pred_boxes = get_result(imgs.shape[1:3], scores, regs, score_thres=0.5)
            n_pred += scores.shape[0]

            # 得到各阈值下所有锚框的is_tps [num_iou_thres, num_box]
            is_tps = get_is_tps(gt_infos, pred_boxes, mAP_iou_thres_list)
            # 累积gt
            n_gt += gt_infos.shape[0]
            # 累积total_P和TP
            # 因为score_thres_list不是与scores一一对应的
            # 所以无法使用TP[:, ind_score] = TP[: ind_score-1] + is_tps[ind_score]
            for ind_score, score_thres in enumerate(score_thres_list):
                P_inds = np.where(scores >= score_thres)[0]
                n = P_inds.size
                if n > 0:
                    total_P[ind_score] += n
                    TP[:, ind_score] += np.sum(is_tps[:, :n], axis=1)
        print(f'\nFINISH {datasets} pics {total_num}.')

        return n_gt, total_P, TP, n_pred


if __name__ == '__main__':
    app = App()

    # app.train()
    # for _ in range(10):
    #     app.predict(save_pic=False)
    # # 使用电脑摄像头并实时对摄像头监测，保存检测到的视频。
    # save_dir = "D:/mtcnn_demo.mp4"
    # use_capture(save_dir, app=app)
    # dir = 'E:/TEST/AI/datasets/test/'
    # app.predict_from_dir(img_dir=dir, show_pic=True, save_pic=False, top_boxes=1)
    """检测文件夹下的所有图像"""
    # a_dir = 'E:/TEST/AI/datasets/changeface_video/x/'
    # app.predict_from_dir(img_dir=a_dir, show_pic=True, save_pic=False)
    # pzy_dir = 'E:/TEST/AI/datasets/cnface/piaozhiyan/'
    # app.predict_from_dir(img_dir=pzy_dir, show_pic=True, save_pic=False, show_num=20)
    # dlrb_dir = 'E:/TEST/AI/datasets/cnface/dilireba/'
    # app.predict_from_dir(img_dir=dlrb_dir, show_pic=True, save_pic=False)

    """根据图片所在文件夹，保存相应的人脸位置文件{path_loc:face_locs[[r1, c1, r2, c2], ...]}"""
    # pic_dir = 'E:/TEST/AI/datasets/changeface_video/x/'
    # save_dir = 'E:/TEST/AI/datasets/changeface_video/a_path_locs.txt'
    # app.write_face_loc(pic_dir, save_dir, top_boxes=1)
    #
    # pic_dir = 'E:/TEST/AI/datasets/changeface/piaozhiyan/'
    # save_path = 'E:/TEST/AI/datasets/changeface/pzy_path_locs.txt'
    # app.write_face_loc(pic_dir, save_path, top_boxes=1)
    #
    # pic_dir = 'E:/TEST/AI/datasets/changeface/wangzuxian/'
    # save_path = 'E:/TEST/AI/datasets/changeface/wzx_path_locs.txt'
    # app.write_face_loc(pic_dir, save_path, top_boxes=1)

    app.close()

    print('Finished!')
