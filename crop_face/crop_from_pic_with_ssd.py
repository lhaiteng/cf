# -*- coding: utf-8 -*-
import numpy as np
import os, shutil, cv2
import tensorflow as tf
from ssd.dl16_CF_ssd_tensors import Tensors


class CropFace:
    def __init__(self):
        graph = tf.Graph()
        with graph.as_default():
            self.ts = Tensors()
            conf = tf.ConfigProto()
            conf.allow_soft_placement = True
            self.sess = tf.Session(config=conf)
            self.saver = tf.train.Saver()
            save_path = '../ssd/models/dl16_CF_ssd_v01'
            try:
                self.saver.restore(self.sess, save_path)
                print(f'Restore model from {save_path} succeed.')
            except:
                raise ValueError('ERROR no trained models.')

    def crop_face_from_root_dir(self, root_dir, remove_old_dir=False, dir_without='face',
                                del_same_size=False, **kwargs):
        dir_names = [n for n in os.listdir(root_dir)
                     if dir_without not in n and os.path.isdir(root_dir + n)]
        for dir_name in dir_names:
            print(f'\nSTART saving face from {dir_name}...')
            save_dir = root_dir + 'face_' + dir_name + '/'
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            pic_dir = root_dir + dir_name + '/'
            self.save_face_with_ssd(pic_dir, save_dir, del_same_size, **kwargs)
            if remove_old_dir:
                shutil.rmtree(pic_dir)
                print(f'FINISH removed dir {pic_dir}.')

    def save_face_with_ssd(self, pic_dir, save_dir, del_same_size, lr=128*2//3, lc=128*2//3, **kwargs):
        pic_paths = [pic_dir + n for n in os.listdir(pic_dir) if os.path.isfile(pic_dir + n)]
        pic_sizes = set()
        if 'pre' in kwargs:
            pre = kwargs['pre']
        else:
            pre = ''
        no_pic = 0
        for path in pic_paths:
            print(f'\rprocessing {path}...', end='')
            # 读取图片，并放缩！！很重要！！！
            img = cv2.imread(path)
            if img is None:
                img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
            if img is None:
                os.remove(path)
                print(f'\nremove {path}.')
                continue
            img, _ = self.resize_img(img, 600, 301)

            _kwargs = {'img': img}  # 默认的参数
            _kwargs.update(kwargs)  # 新添及更新的参数
            face_locs = self.get_face_locs(**_kwargs)
            for loc in face_locs:
                r1, c1, r2, c2 = loc.astype(np.int)
                # # 扩大范围
                # r1, r2 = max(r1 - (r2 - r1) // 3, 0), r2 + (r2 - r1) // 3
                # c1, c2 = max(c1 - (c2 - c1) // 4, 0), c2 + (c2 - r1) // 4
                # 若图片过小，则丢弃
                if abs(r1 - r2) < lr or abs(c1 - c2) < lc:
                    continue
                face = img[r1:r2, c1:c2, :]
                if del_same_size and face.size in pic_sizes:
                    continue
                elif del_same_size:
                    pic_sizes |= {face.size}
                cv2.imwrite(save_dir + f'{pre}{no_pic}.png', face)

                no_pic += 1

    def get_face_locs(self, img, top_boxes=0, **kwargs):
        imgs = np.expand_dims(img, axis=0) if len(img.shape) < 4 else img
        # 使用网络预测
        ts = self.ts
        sub_ts = ts.sub_ts[0]
        feed_dict = {ts.training: False, sub_ts.x: imgs}
        run_list = [sub_ts.ob_scores, sub_ts.ob_boxes]
        ob_scores, ob_boxes = self.sess.run(run_list, feed_dict)
        # # [-1, 4]  r1, c1, r2, c2
        if top_boxes:
            ob_scores = ob_scores[:top_boxes]
            ob_boxes = ob_boxes[:top_boxes]
        # 返回图像中的坐标
        return ob_boxes

    def resize_img(self, img, max, min):
        img_shape = img.shape[:2]

        # 先令最长边不大于600
        long_shape = np.max(img_shape)
        long_resize_ratio = np.minimum(max / long_shape, 1)
        # 再令最短边不小于300
        short_shape = np.min(img_shape) * long_resize_ratio
        resize_ratio = np.maximum(min / short_shape, 1) * long_resize_ratio

        img = cv2.resize(img, (0, 0), fx=resize_ratio, fy=resize_ratio)

        return img, resize_ratio

    def close(self):
        self.sess.close()


if __name__ == '__main__':
    cf = CropFace()

    # # 使用根文件夹
    # for suptitle in ['cnface']:
    #     root_dir = f'E:/TEST/AI/datasets/{suptitle}/'
    #     pre = 'params'
    #     cf.crop_face_from_root_dir(root_dir, top_boxes=1, pre=pre,
    #                                dir_without='face', del_same_size=True)

    # 单一文件夹中的图片
    pic_dir = 'E:/TEST/AI/datasets/cnface/mengjia/'
    save_dir = 'E:/TEST/AI/datasets/cnface_face/face_mengjia/'
    cf.save_face_with_ssd(pic_dir, save_dir=save_dir, pre='params', del_same_size=True, lr=128*2//3, lc=128*2//3)

    cf.close()
