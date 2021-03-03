# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from arcface.arcface_config import arcface_cfg as cfg
from arcface.arcface_app import App
from util.arcface_utils import get_triu, get_logits_label_unlabel, \
    distribution_values, distribution_angles, distribution_sims, \
    heatmap_sim_matrix, heatmap_sim_matrixs, heatmap_logit
from util.img_utils import cv2_imread


class Debug:
    def __init__(self, restore_epoch=99):
        self.app = App(restore_epoch=restore_epoch)

    def run(self):
        self.datas = self.app.load_com_datasets()
        accs, train_accs, train_angles, train_labels, train_sims, train_cls_sims, \
        test_accs, test_angles, test_labels, test_sims, test_cls_sims = self.datas
        _save_file = cfg.record_dir + 'angels_w_centers.png'
        self.app.distribution_w_cetners(save_file=_save_file)

    # 计算得到所有列类别的准确率、所有样本的logit夹角和标签项、类内相似度分布
    def com_all_cls(self, datasets='train'):
        """
        :param datasets:
        :return:
            accs: [num_cls] 各类别的准确率列表
            train_angles: [num_pics, num_cls] 各样本的logit夹角
            labels_list: [num_pics] 各样本的标签
            sims: [各类别C_类内图片数_2之和] 各样本与类内样本的相似度列表。
        """
        num_cls = cfg.num_cls
        acc, accs = 0, []
        angles, labels = [], []
        sims = []
        num_pics = 0
        for cls in range(num_cls):
            _num_pic, _acc, _angles, _labels, _sims = self.com_cls(cls, datasets)
            num_pics += _num_pic
            acc += _acc * _num_pic
            accs.append(_acc)
            angles.extend(_angles)
            labels.extend(_labels)
            sims.extend(_sims)
            print(f'\rcls:{cls}/{num_cls} cum_acc:{acc / num_pics:.3%} acc:{_acc:.3%}', end='')
        acc /= num_pics
        mean_sim = np.mean(sims)
        print(f'\ndatasets: {datasets} acc: {acc:.3%} sim: {mean_sim:.3f}.')

        return accs, np.array(angles), np.array(labels), sims

    # 指定类别计算准确率、夹角、相似度分布
    def com_cls(self, cls=0, datasets='train'):
        # 如果类内图片过多，还要分批次进行。此时计算类内相似度不能使用ts.sim_matrix，需要先计算出所有id，再统一计算
        app = self.app

        paths = app.sa.dict_label_path[datasets][str(f' {cls}')]
        num_pics = len(paths)

        imgs = []
        for path in paths:
            try:
                img = cv2_imread(path)
                img = cv2.resize(img, (cfg.cnn_shape, cfg.cnn_shape), interpolation=cv2.INTER_AREA)
                imgs.append(img)
            except:
                continue
        labels = np.array([cls] * len(imgs))
        imgs = np.array(imgs)

        feed_dict = {app.ts.keep_prob: 1, app.ts.training: False,
                     app.ts.inputs: imgs, app.ts.labels: labels}
        run_list = [app.ts.acc, app.ts.thetas_angle, app.ts.cosines, app.ts.sim_matrix]
        acc, angles, cosines, sim_matrix = app.sess.run(run_list, feed_dict)

        # 本批次图片之间的相似度
        sims = get_triu(sim_matrix)

        # self._com_cls_plot(train_angles, cosines, labels_list, sim_matrix)

        return num_pics, acc, angles, labels, sims

    def _com_cls_plot(self, angles, cosines, labels, sim_matrix):
        # 夹角分布
        distribution_angles(angles, labels)
        # 夹角余弦分布
        distribution_sims(cosines, labels)
        # logit热力图
        heatmap_logit(angles, labels, kind='angle', title='angle')
        heatmap_logit(cosines, labels, kind='cosine', title='cosine')
        # 余弦相似度矩阵
        heatmap_sim_matrix(sim_matrix, labels)

    def debug_datsets(self):
        app = self.app

        # 训练集
        app.plot_similarity_from_datasets(num=9, same=6, suptitle=f'train', datasets='train')
        app.test_datasets(cate='train', plot_distribution=True)
        app.heatmap_sim_from_datasets(datasets='train')
        # 测试集
        app.plot_similarity_from_datasets(num=9, same=6, suptitle=f'test', datasets='test')
        app.test_datasets(cate='test', plot_distribution=True)
        app.heatmap_sim_from_datasets(datasets='test')

    def com_datasets(self):
        app = self.app

        # 计算测试集和验证集的准确率、画出所有数据的分布
        app.sa.remove_img_aug()  # 去除数据增强
        app.com_datasets(datasets='total')
        app.sa.return_img_aug()  # 恢复数据增强

    def plot_heatmap_cls_weight(self):
        app = self.app
        ts = app.ts
        # cls_weight
        cls_weight = app.sess.run(ts.cls_weight)
        sns.heatmap(cls_weight)
        plt.title('cls_weight')
        plt.show()

    def close(self):
        self.app.close()


if __name__ == '__main__':
    ## 设置属性防止中文乱码
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    restore_epoch = 99
    debug = Debug(restore_epoch=restore_epoch)
    debug.run()

    debug.close()
    print('Finished!')
