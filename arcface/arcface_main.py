# -*- coding: utf-8 -*-
from arcface.arcface_config import arcface_cfg as cfg
from arcface.arcface_app import App
import numpy as np
import cv2, os, math, random
import matplotlib as mpl
import matplotlib.pyplot as plt


def train(app: App):
    # 预测一次训练集和测试集。刚开始时得到的夹角都是90，使用sns.distplot会报奇异矩阵的错误。
    app.plot_similarity_from_datasets(suptitle=f'train', datasets='train')
    app.test_datasets(cate='train', plot_distribution=False if start_epoch == 0 else True)
    app.plot_similarity_from_datasets(suptitle=f'test', datasets='test')
    app.test_datasets(cate='test', plot_distribution=False if start_epoch == 0 else True)
    # 训练
    app.train_arcface(start_epoch=start_epoch)


def com(app: App):
    # 去除数据增强
    app.sa.remove_img_aug()
    # 计算测试集和验证集的准确率、画出所有数据的分布，并保存相关数据
    rd = cfg.record_dir
    app.com_datasets(save_path=rd)
    # 恢复数据增强
    app.sa.return_img_aug()


if __name__ == '__main__':
    ## 设置属性防止中文乱码
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    # # 不使用科学计数
    # np.set_printoptions(suppress=True)

    restore_epoch, start_epoch = 99, 0
    app = App(restore_epoch=restore_epoch)

    train(app)

    com(app)

    # # 结束后对cfg.test_imgs_dir内的图片进行对比
    # img_dir = 'E:/TEST/AI/datasets/test_face1/'
    # app.plot_similarity_from_dir_every(dir=img_dir)
    # app.plot_similarity_from_dir_every()

    # 图片对比
    app.heatmap_sim_from_dir()

    app.close()
    print('Finished!')
