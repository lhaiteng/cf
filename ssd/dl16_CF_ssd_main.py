# -*- coding: utf-8 -*-
from ssd.dl16_CF_ssd_config import ssd_cfg
from ssd.dl16_CF_ssd_tensors import Tensors
from ssd.dl16_CF_ssd_samples import Samples
from ssd.dl16_CF_ssd_app import App
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time, cv2, os, json


def train(app: App):
    app.train()
    for _ in range(10):
        app.predict(save_pic=False)
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


def main(app: App):
    app.samples.remove_aug()
    for datasets in ('train', 'test'):
        APs, mAP = app.get_mAP(datasets, save_record=False)
        print(f'{datasets} APs: {APs}')
        print(f'{datasets} mAP: {mAP:.2%}')
    app.samples.return_aug()


if __name__ == '__main__':
    app = App()

    # train(app)

    main(app)

    app.close()

    print('Finished!')
