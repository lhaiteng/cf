# -*- coding: utf-8 -*-
from arcface.arcface_config import arcface_cfg as cfg
from arcface.arcface_tensors import ArcFace
from arcface.arcface_samples import FaceRecSamples
import tensorflow as tf
import numpy as np
import cv2, os, math, random
import matplotlib.pyplot as plt
import seaborn as sns
from util.lr_generator import LR_decay
from util.arcface_parameter import ArcPara
from util.arcface_utils import get_triu, \
    plot_similarity, heatmap_logit, print_thetas_infos, \
    distribution_angles, distribution_sims, distribution_values, distribution_logits, \
    heatmap_face_weight, heatmap_sim_matrix, heatmap_sim_matrixs, heatmap_matrix, heatmap_matrixs


class App:
    def __init__(self, restore_epoch=99):
        cfg.check_all_dir_paths()
        self.sa = FaceRecSamples()
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.ts = ArcFace()
            conf = tf.ConfigProto()
            conf.allow_soft_placement = True
            self.sess = tf.Session(config=conf)
            ts_paths = []  # (restore_ts_list, save_path)用来恢复模型指定变量的。已保存变量包括待恢复的变量
            # 恢复模型
            print(f'SUCCEED: export_meta_graph to {cfg.save_path}.meta')
            ts_paths.append((self.ts.vars_restore, cfg.save_path + f'-{restore_epoch}'))
            # 保存器
            self.saver_arcface = tf.train.Saver(var_list=self.ts.vars_save, max_to_keep=5)
            self.saver_arcface.export_meta_graph(cfg.save_path + '.meta')

            # 模型初始化、恢复指定变量
            self.init_restore_model(ts_paths)

    def init_restore_model(self, ts_paths):
        # 先初始化所有变量
        print('START: init all vars.')
        self.sess.run(tf.global_variables_initializer())

        for ts_list, path in ts_paths:
            try:
                loader = tf.train.Saver(var_list=ts_list)
                loader.restore(self.sess, path)
                print(f'SUCCEED: restore model from {path}.')
            except:
                print(f'FAILED: restore model from {path}.')

    def train_arcface(self, batch_size=None, start_epoch=0):
        # 写关系图和变量
        writer = tf.summary.FileWriter(cfg.log_dir, tf.get_default_graph())

        # start_epoch = cfg.start_epoch
        num_epoch = cfg.epochs
        if batch_size is None:
            batch_size = cfg.train_batch_size
        # 一个epoch有多少batch
        num_batch = cfg.num_train_batch_expand * (self.sa.train_num // batch_size)
        # 经过多少batch显示一次预测
        show_num = int(cfg.show_num // batch_size)
        # 每个epoch经过多少batch显示一次损失
        num_print_loss = cfg.num_to_print_losses
        train_infos_list = [batch_size, num_batch, num_epoch, num_print_loss, show_num, start_epoch,
                            cfg.num_train_batch_expand]
        self.print_train_infos(*train_infos_list)

        # 训练id
        step = 0  # 表示第几个大batch，而不是epoch!!!
        LR = LR_decay(**cfg.lr_para, epochs=cfg.epochs)
        arc_para = ArcPara(**cfg.arcface_para)
        ts = self.ts
        run_nec = [ts.train_arcface, ts.thetas_angle]  # 2
        run_info = [ts.acc, ts.theta_label, ts.theta_unlabel]  # 3
        run_print = run_nec + run_info  # 有打印
        run_merge = run_nec + [ts.merged]  # 有merge
        run_all = run_print + [ts.merged]  # 有打印有merge
        feed_dict = {ts.keep_prob: cfg.keep_prob, ts.training: True}
        for epoch in range(start_epoch, num_epoch):
            feed_dict[ts.lr] = LR.get_lr(epoch, cate='cosine_decay2')  # lr = LR.get_lr(num_epoch)
            feed_dict[ts.para_arcface] = arc_para.get_para(epoch)
            feed_dict[ts.should_com_center_loss] = False if epoch < cfg.center_stage else True
            cum_acc = 0.  # 本轮的初始累积准确率
            for batch in range(num_batch):
                step = self.sess.run(ts.global_step)
                # 整除时显示画出数据集的夹角分布
                if show_num and step % show_num == 0 and step > 0:
                    self.result_for_show_num()
                # 更新参数
                imgs, labels = self.sa.next_batch(batch_size, cate='train')
                feed_dict.update({ts.inputs: imgs, ts.labels: labels})

                # 打印损失
                if num_print_loss == 0 or batch % num_print_loss == 0 or batch == num_batch - 1:
                    # 每100步记录一次tensorboard
                    run_list = run_all if step % 100 == 0 else run_print
                    run_results = self.sess.run(run_list, feed_dict=feed_dict)
                    cum_acc = cum_acc * 0.95 + run_results[2] * 0.05  # 200步后结果基本符合均值
                    print(f'\rEPOCH {epoch}/{num_epoch} BATCH {batch}/{num_batch} cum_acc {cum_acc:.3%} - '
                          f'acc={run_results[2]:.3%} theta={run_results[3]:.3f} untheta={run_results[4]:.3f}', end='')
                else:
                    # 每100步记录一次tensorboard
                    run_list = run_merge if step % 100 == 0 else run_nec
                    run_results = self.sess.run(run_list, feed_dict=feed_dict)
                    # 每100步记录一次tensorboard
                if step % 100 == 0: writer.add_summary(run_results[-1], step)

            # 每代结束后，保存模型
            self.saver_arcface.save(self.sess, cfg.save_path, write_meta_graph=False, global_step=epoch)
            print(f'\tSave into model {cfg.save_path}')
            # 每代结束后查看训练效果
            self.result_after_one_epoch(epoch)

    # 打印训练参数
    def print_train_infos(self, batch_size, num_batch, num_epoch, num_print_loss, show_num,
                          start_epoch, num_batch_expand):
        print(f'Start training from epoch No.{start_epoch}...')
        print(f'Train pics {self.sa.train_num}.')
        print(f'Train num_epoch: {num_epoch}')
        print(f'Train num_batch_expand: {num_batch_expand}.')
        print(f'Train num_batch: {num_batch} ({num_batch * batch_size} pics/ batch)')
        print(f'Show loss every {num_print_loss} batch / epoch.')
        print(f'Show predict every {show_num} batch, '
              f'after train {show_num * batch_size} pics.')
        print(f'Factor of l1 loss id: {cfg.l1_loss_factor_id}')
        print(f'Factor of l1 loss w: {cfg.l1_loss_factor_id}')
        print(f'Factor of center loss: {cfg.center_loss_factor}')
        print(f'LR para:\n{cfg.lr_para}')
        print(f'Arcface para:\n{cfg.arcface_para}')

    # 训练中整除show_num时的操作
    def result_for_show_num(self):
        # 显示训练集
        self.plot_similarity_from_datasets(suptitle='train', datasets='train')
        self.test_datasets(cate='train', print_compare=False)
        # 显示测试集
        self.plot_similarity_from_datasets(suptitle='test', datasets='test')
        self.test_datasets(cate='test', print_compare=False)
        # 查看cls_weight、centers的情况
        self.distribution_w_cetners()

    # 每代结束后查看训练效果
    def result_after_one_epoch(self, epoch):
        root_dir = cfg.gen_dir
        # 测试一次训练集
        kw = {'plot': False, 'save_file': root_dir + f'E{epoch}_train_similarity.png'}
        self.plot_similarity_from_datasets(suptitle=f'train E{epoch}', datasets='train', **kw)
        kw = {'plot_logit': False, 'plot_distribution': False,
              'save_heatmap_logit_file': root_dir + f'E{epoch}_train_heatmap_logit.png',
              'save_distribution_angle_file': root_dir + f'E{epoch}_train_distribution_angle.png'}
        self.test_datasets(cate='train', **kw)

        # 测试一次测试集
        kw = {'plot': False, 'save_file': root_dir + f'E{epoch}_test_similarity.png'}
        self.plot_similarity_from_datasets(suptitle=f'test E{epoch}', datasets='test', **kw)
        kw = {'plot_logit': False, 'plot_distribution': False,
              'save_heatmap_logit_file': root_dir + f'E{epoch}_test_heatmap_logit.png',
              'save_distribution_angle_file': root_dir + f'E{epoch}_test_distribution_angle.png'}
        self.test_datasets(**kw)

    # 每代结束后的一些测试结果
    def test_datasets(self, batch_size=cfg.test_batch_size, cate='test', **kwargs):
        """
        :param batch_size:
        :param cate:
        :param kwargs:
            print_compare: 默认True
            plot_logit: 默认True
            plot_distribution: 默认True
        :return:
        """

        ts = self.ts

        imgs, labels = self.sa.next_batch(batch_size, cate, should_sort=True)

        feed_dict = {ts.keep_prob: 1, ts.training: False,
                     ts.inputs: imgs, ts.labels: labels,
                     ts.para_arcface: [1, 1, 0, 0]}

        run_list = [ts.arcface_id, ts.acc, ts.theta_label, ts.theta_unlabel, ts.thetas_angle]

        ids, acc, theta_label, theta_unlabel, thetas_angle = self.sess.run(run_list, feed_dict)

        # 标签项的夹角们
        thetas_label_angle = thetas_angle[range(batch_size), labels]
        # 非标签项的夹角们
        # thetas_unlabel_angle = thetas_angle.tolist()
        # thetas_unlabel_angle = np.array([thetas_unlabel_angle[i][:labels_list[i]] + thetas_unlabel_angle[i][labels_list[i] + 1:]
        #                                  for i in range(len(thetas_unlabel_angle))])
        num = thetas_angle.shape[0]
        unlable_inds = np.zeros_like(thetas_angle)
        unlable_inds[range(num), labels] = 1  # 标签项
        thetas_unlabel_angle = thetas_angle[unlable_inds == 0].reshape([num, -1])

        theta_infos_label = np.var(thetas_label_angle), np.max(thetas_label_angle), np.min(thetas_label_angle)
        theta_infos_unlabel = np.var(thetas_unlabel_angle), np.max(thetas_unlabel_angle), np.min(thetas_unlabel_angle)

        print_compare = kwargs.get('print_compare', True)
        if print_compare:
            print('-' * 100)
            print(f'{cate} {batch_size} 个数据')
            print(f'准确率\t{acc:.3%}\n'
                  f'标签项夹角\tmean:{theta_label:.3f}\tvar:{theta_infos_label[0]:.3f}\t'
                  f'max:{theta_infos_label[1]:.3f}\tmin:{theta_infos_label[2]:.3f}\n'
                  f'非标签夹角\tmean:{theta_unlabel:.3f}\tvar:{theta_infos_unlabel[0]:.3f}\t'
                  f'max:{theta_infos_unlabel[1]:.3f}\tmin:{theta_infos_unlabel[2]:.3f}')
            print('-' * 100)
            print_thetas_infos(thetas_angle, labels)

        plot_logit = kwargs.get('plot_logit', True)
        save_heatmap_logit_file = kwargs.get('save_heatmap_logit_file', None)
        if plot_logit or save_heatmap_logit_file:
            kw = {'plot': plot_logit, 'save_file': save_heatmap_logit_file}
            heatmap_logit(logits=thetas_angle, labels=labels,
                          kind='angle', title=f'angle_{cate}_acc: {acc:.3%}', **kw)

        plot_distribution = kwargs.get('plot_distribution', True)
        save_distribution_angle_file = kwargs.get('save_distribution_angle_file', None)
        if plot_distribution or save_distribution_angle_file:
            kw = {'plot': plot_distribution, 'save_file': save_distribution_angle_file}
            distribution_angles(angles=thetas_angle, labels=labels, title=f'angle_{cate}', **kw)

    # 从数据集中抽取图片画两两相似度的热力图
    def heatmap_sim_from_datasets(self, batch_size=32, same=3, datasets='train'):
        """
        从数据集中抽取图片画相似度热力图
        :param batch_size: 抽取图片个数
        :param same: 同类型图片的个数。不足的用batch_size%same来补充。
        :param datasets: 数据集类型
        :return:
        """
        if datasets == 'train':
            b1, b2 = batch_size, 0
        elif datasets == 'test':
            b1, b2 = 0, batch_size
        else:
            b1 = b2 = batch_size // 2

        imgs, labels = [], []
        if datasets is None or datasets in ('train', 'total'):
            n, last = b1 // same, b1 % same
            for _ in range(n):
                _imgs, _labels = self.sa.get_imgs_for_similarity_from_datasets(same, same, datasets='train')
                imgs.extend(_imgs)
                labels.extend(_labels)
            if last:
                _imgs, _labels = self.sa.get_imgs_for_similarity_from_datasets(last, last, datasets='train')
                imgs.extend(_imgs)
                labels.extend(_labels)
        if datasets is None or datasets in ('test', 'total'):
            n, last = b2 // same, b2 % same
            for _ in range(n):
                _imgs, _labels = self.sa.get_imgs_for_similarity_from_datasets(same, same, datasets='test')
                imgs.extend(_imgs)
                labels.extend(_labels)
            if last:
                _imgs, _labels = self.sa.get_imgs_for_similarity_from_datasets(last, last, datasets='test')
                imgs.extend(_imgs)
                labels.extend(_labels)
        imgs, labels = np.asarray(imgs), np.asarray(labels)

        feed_dict = {self.ts.keep_prob: 1, self.ts.training: False,
                     self.ts.inputs: imgs}
        sim_matrix = self.sess.run(self.ts.sim_matrix, feed_dict)
        heatmap_sim_matrix(sim_matrix, labels)

    # 从文件夹中读取图片画两两相似度的热力图
    def heatmap_sim_from_dir(self, img_dir=cfg.test_imgs_dir):
        # 画相似度热力图
        imgs = []
        img_name = [n for n in os.listdir(img_dir)]
        img_path = [img_dir + n for n in img_name]
        img_name = [n[:-4] for n in img_name]
        for p in img_path:
            img = cv2.imread(p)
            img = cv2.resize(img, (cfg.cnn_shape, cfg.cnn_shape))
            imgs.append(img)
        imgs = np.array(imgs)

        feed_dict = {self.ts.keep_prob: 1, self.ts.training: False,
                     self.ts.inputs: imgs}
        sim_matrix = self.sess.run(self.ts.sim_matrix, feed_dict)
        heatmap_sim_matrix(sim_matrix, img_name)

    # 分别画出分类权重、画出中心向量两两相似度的热力图
    def heatmap_sim_weight_centers(self):
        # 权重两两相似度矩阵
        sim_matrix_cls_weight = self.sess.run(self.ts.sim_matrix_cls_weight)  # [id_size, num_cls]
        # centers两两相似度矩阵
        sim_matrix_centers = self.sess.run(self.ts.sim_matrix_centers)
        # 权重和centers两两相似度矩阵
        sim_matrix_w_centers = self.sess.run(self.ts.sim_matrix_w_centers)

        matrixs = [sim_matrix_cls_weight, sim_matrix_centers, sim_matrix_w_centers]
        labels = [range(cfg.num_cls), range(cfg.num_cls), range(cfg.num_cls)]
        num_matrixs = len(labels)
        heatmap_sim_matrixs(matrixs, labels_list=labels, num=num_matrixs)

    # 分别画出分类权重、中心向量两两夹角的分布图
    def distribution_w_cetners(self, **kwargs):
        """
        :param kwargs:
            save_file: 默认不保存
            plot: 默认True
        :return:
        """
        # 分类权重两两夹角
        sim_matrix_cls_weight = self.sess.run(self.ts.sim_matrix_cls_weight)  # [id_size, num_cls]
        sims_cls_weight = get_triu(sim_matrix_cls_weight)
        angels_cls_weight = np.arccos(sims_cls_weight) * 180 / np.pi
        # 中心向量两两夹角
        sim_matrix_centers = self.sess.run(self.ts.sim_matrix_centers)
        sims_centers = get_triu(sim_matrix_centers)
        angels_centers = np.arccos(sims_centers) * 180 / np.pi
        # 权重对应中心向量两两夹角
        sim_matrix_w_centers = self.sess.run(self.ts.sim_matrix_w_centers)
        sims_w_centers = sim_matrix_w_centers[range(cfg.num_cls), range(cfg.num_cls)]
        angels_w_centers = np.arccos(sims_w_centers) * 180 / np.pi

        # 画出分布图
        angles_list = [angels_cls_weight, angels_centers, angels_w_centers]
        name_list = ['cls_weight', 'centers', 'w-centers']
        distribution_angles(angles=angles_list, name_list=name_list, **kwargs)

    # 整体计算指定datasets的acc、radians、labels_list、sims、cls_sims
    def com_datasets(self, datasets=None, batch_size=80,
                     show_infos=True, com_acc=True, com_sim=True, plot_distribution=True,
                     **kwargs):
        """
        计算准确率、绘制夹角分布图。计算数据集的相似度
        :param datasets:
        :param batch_size: 单类别计算时的最大图片数量
        :param show_infos: 是否在计算前显示信息
        :param com_acc: 计算准确率
        :param com_sim: 储存同类型的相似度，并计算均值
        :param plot_distribution: 对scope范围所有的样本求夹角，画出夹角、相似度分布
        :param kwargs:
            save_path 默认None
        :return:
            accs: [train_acc, test_acc]
            train_angles: [n_pics, num_cls]
            names: [n_pics]
            train_sims: [?, ] 分别对训练集和测试集的类内图片，两两求相似度
            train_cls_sims: [num_cls, ] 类内两两相似度的均值
        """
        save_path = kwargs.get('save_path', None)

        accs = []
        train_accs, train_angles, train_labels, train_sims, train_cls_sims = [], [], [], [], []
        if datasets is None or datasets in ('train', 'total'):
            datas = self._com_datasets(datasets='train', batch_size=batch_size,
                                       show_infos=show_infos,
                                       com_acc=com_acc, com_sim=com_sim,
                                       plot_distribution=plot_distribution)
            acc, train_accs, train_angles, train_labels, train_sims, train_cls_sims = datas
            if com_acc: accs.append(acc)
        test_accs, test_angles, test_labels, test_sims, test_cls_sims = [], [], [], [], []
        if datasets is None or datasets in ('test', 'total'):
            datas = self._com_datasets(datasets='test', batch_size=batch_size,
                                       show_infos=show_infos,
                                       com_acc=com_acc, com_sim=com_sim,
                                       plot_distribution=plot_distribution)
            acc, test_accs, test_angles, test_labels, test_sims, test_cls_sims = datas
            if com_acc: accs.append(acc)

        # 画出数据夹角分布
        _save_file = save_path + '{d}_angles.png' if save_path else None
        _kw = {'plot': plot_distribution, 'title': 'train train_angles', 'save_file': _save_file.format(d='train')}
        distribution_angles(train_angles, train_labels, **_kw)
        _kw = {'plot': plot_distribution, 'title': 'test train_angles', 'save_file': _save_file.format(d='test')}
        distribution_angles(test_angles, test_labels, **_kw)

        # 画出各数据集中的类内图片两两相似度分布
        _kw = {'name_list': ['train sims', 'test sims'], 'title': 'similarity between same cls pics',
               'plot': plot_distribution, 'save_file': save_path + 'similarity_between_same_cls_pics.png' if save_path else None}
        distribution_sims([train_sims, test_sims], **_kw)

        # 画出w、centers类间夹角分布
        _kw = {'plot': plot_distribution, 'save_file': save_path + 'angles_w_centers.png' if save_path else None}
        self.distribution_w_cetners(**_kw)

        # 画出各数据集中对应类别的类内平均相似度、准确率曲线
        if plot_distribution:
            plt.figure(figsize=[20, 10])
            plt.subplot(121)
            plt.plot(range(cfg.num_cls), train_cls_sims, marker='o', label='train mean_sim')
            plt.plot(range(cfg.num_cls), test_cls_sims, marker='o', label='test mean_sim')
            plt.legend()
            plt.ylim([-0.1, 1.1])
            plt.title('cls-mean_sim')
            plt.subplot(122)
            plt.plot(range(cfg.num_cls), train_accs, marker='o', label='train mean_acc')
            plt.plot(range(cfg.num_cls), test_accs, marker='o', label='test mean_acc')
            plt.legend()
            plt.ylim([-0.1, 1.1])
            plt.title('cls-mean_acc')
            plt.show()

        datas = [accs,
                 train_accs, train_angles, train_labels, train_sims, train_cls_sims,
                 test_accs, test_angles, test_labels, test_sims, test_cls_sims]

        # 保存数据
        if save_path:
            file_names = ['accs.npy',
                          'train_accs.npy', 'train_angles.npy', 'names.npy', 'train_sims.npy',
                          'train_cls_sims.npy',
                          'test_accs.npy', 'test_angles.npy', 'test_labels.npy', 'test_sims.npy', 'test_cls_sims.npy']
            self.save_com_datasets(save_path, file_names, datas)

        return datas

    def save_com_datasets(self, save_path, file_names, datas):
        save_files = [save_path + fn for fn in file_names]
        for i, save_file in enumerate(save_files):
            np.save(save_file, datas[i])

    def load_com_datasets(self, save_path=None, file_names=None):
        if save_path is None: save_path = cfg.record_dir
        if file_names is None:
            file_names = ['accs.npy',
                          'train_accs.npy', 'train_angles.npy', 'names.npy', 'train_sims.npy',
                          'train_cls_sims.npy',
                          'test_accs.npy', 'test_angles.npy', 'test_labels.npy', 'test_sims.npy', 'test_cls_sims.npy']
        save_files = [save_path + fn for fn in file_names]
        datas = []
        for i, save_file in enumerate(save_files):
            datas.append(np.load(save_file))
        print(f'succeed: load {len(datas)} data.')
        return datas

    def _com_datasets(self, datasets='train', batch_size=80,
                      show_infos=True, com_acc=True, com_sim=True, plot_distribution=True):
        num_pics = self.sa.train_num if datasets == 'train' else self.sa.test_num
        num_cls = cfg.num_cls
        if show_infos:
            print('-' * 100)
            print(f'START:\nsource: {datasets} com_cls: {num_cls} sample_num: {num_pics}\n'
                  f'com acc: {com_acc}  plot distribution: {plot_distribution}\n'
                  f'max batch_size: {batch_size}')
        all_accs, all_angles, all_labels, sims, cls_sims = [], [], [], [], [0] * num_cls
        acc = 0
        num_comed = 0  # 已经计算过的图片个数

        run_list = []
        if com_acc: run_list.append(self.ts.acc)
        if com_sim: run_list.append(self.ts.arcface_id)  # 不能直接用tf求sim_matrix，因为可能一个批次装不完所有图片
        if plot_distribution: run_list.append(self.ts.thetas_angle)
        feed_dict = {self.ts.keep_prob: 1, self.ts.training: False}

        for cls in range(num_cls):  # 按类别对图片计算
            imgs, labels = self.sa.get_imgs_by_class(cls, datasets=datasets)
            ids = []
            # # 防止图片过多，分批进行计算
            # n_batch = math.ceil(imgs.shape[0] / batch_size)
            # 直接把所有图片放入批次中计算
            bs = imgs.shape[0]  # 本批次中的图片个数
            feed_dict.update({self.ts.inputs: imgs, self.ts.labels: labels})
            result = self.sess.run(run_list, feed_dict=feed_dict)

            if com_acc:
                all_accs.append(result[0])
                acc += result[0] * bs
            if com_sim: ids.extend(result[1] if com_acc else result[0])
            if plot_distribution:
                all_angles.extend(result[-1])
                all_labels.extend(labels)
            num_comed += bs  # 累加图片个数
            # 本类图片批次计算完全后，计算本类图片的两两相似度。当图片数目大于1时才计算。
            if com_sim and len(ids) > 1:
                ids = np.array(ids)
                _sim = np.matmul(ids, ids.T)
                _sim = get_triu(_sim)
                sims.extend(_sim)
                cls_sims[cls] = np.mean(sims)
            if show_infos: print(f'\rdatasets:{datasets} cls_logits:{cls}/{num_cls} ACC:{acc / num_comed:.3%}', end='')
        acc /= num_comed
        sim = np.mean(sims)
        if show_infos:
            print(f'\nFINISH: datasets: {datasets} ACC: {acc:.3%} SIM: {sim:.3}')
            print('-' * 100)
        return acc, all_accs, all_angles, all_labels, sims, cls_sims

    # 从datasets中抽取若干图片，画出各图片与第一张图片的相似度
    def plot_similarity_from_datasets(self, num=6, same=3, suptitle=None, datasets='total', **kwargs):
        plot, save_file = kwargs.get('plot', True), kwargs.get('save_file', None)
        if not plot and not save_file: return
        imgs, labels = self.sa.get_imgs_for_similarity_from_datasets(num, same, datasets=datasets)
        self.plot_similarity(imgs, labels, suptitle, **kwargs)

    # 从dir中抽取若干图片，画出各图片与第一张图片的相似度
    def plot_similarity_from_dir(self, num=None, dir=cfg.test_imgs_dir, suptitle='test'):
        # 把文件夹中所有num个图片放在一张图上
        path_names = [(dir + n, n) for n in os.listdir(dir) if os.path.isfile(dir + n)]
        if num is not None: path_names = path_names[:num]
        random.shuffle(path_names)
        imgs, labels = self.get_img_labels(path_names)
        self.plot_similarity(imgs, labels, suptitle=suptitle)

    # 从dir中抽取若干图片，逐步画出各图片与余下图片的相似度
    def plot_similarity_from_dir_every(self, dir=cfg.test_imgs_dir):
        # 对dir中的所有图片逐个与余下所有进行对比
        path_names = [(dir + n, n) for n in os.listdir(dir) if os.path.isfile(dir + n)]
        imgs, labels = self.get_img_labels(path_names)
        num = imgs.shape[0]
        for i in range(num - 1):
            self.plot_similarity(imgs[i:], labels[i:])

    # 与第一张图片，对比其他图片的相似度
    def plot_similarity(self, imgs, labels, suptitle=None, **kwargs):
        ts = self.ts
        feed_dict = {ts.keep_prob: 1, ts.inputs: imgs, ts.training: False, }
        ids = self.sess.run(ts.arcface_id, feed_dict=feed_dict)

        plot_similarity(imgs, ids, labels, suptitle, **kwargs)

    # 根据文件路径和文件名，读取图片和标签，
    def get_img_labels(self, path_names):
        imgs, labels = [], []
        for path, name in path_names:
            try:
                img = cv2.imread(path)
                if img is None:
                    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
                img = cv2.resize(img, (cfg.cnn_shape, cfg.cnn_shape), interpolation=cv2.INTER_AREA)
                imgs.append(img)
                labels.append(name)
            except:
                continue
        return np.array(imgs), np.array(labels)

    def close(self):
        self.sess.close()
        self.sa.close()


if __name__ == '__main__':
    restore_epoch, start_epoch = 99, 0
    app = App(restore_epoch=restore_epoch)

    # # 结束后对cfg.test_imgs_dir内的图片进行对比
    # img_dir = 'E:/TEST/AI/datasets/test_face1/'
    # app.plot_similarity_from_dir_every(dir=img_dir)
    # app.plot_similarity_from_dir_every()

    # app.test_datasets()

    app.sa.remove_img_aug()
    app.com_datasets(com_acc=True, plot_distribution=True)
    app.sa.return_img_aug()

    # 图片对比
    app.heatmap_sim_from_dir()

    app.close()
    print('Finished!')
