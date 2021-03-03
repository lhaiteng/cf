"""
准备的数据
    img
    gt_infos
    rpn_label  -> proposal_label跟随训练产生，使用tf.py_func
"""
# -*- coding: utf-8 -*-
import matplotlib as mpl
import os, random, json, time
import threading, queue
from ssd.dl16_CF_ssd_utils import *
from ssd.dl16_CF_ssd_config import ssd_cfg
from ssd.dl16_CF_ssd_gt_infos_layer import get_resized_gt
from ssd.dl16_CF_ssd_target_layer import ssd_target_layer


class Samples:
    def __init__(self):
        self.train_path_gts, self.test_path_gts = self.get_path_gts()
        # # [(path_label, gts), (path_label, gts), ...]

        self.train_samples = self.get_data('train')
        self.test_samples = self.get_data('test')

        self.should_stop = False
        # 线程
        self.buffer = queue.Queue(ssd_cfg.train_buffer_capacity)
        self.th = threading.Thread(target=self.produce_train)
        lock = threading.RLock()
        self.has_data = threading.Condition(lock)
        self.has_empty = threading.Condition(lock)
        self.th.start()

        self.should_aug = True

    def get_path_gts(self):
        train_path_gts = []
        test_path_gts = []

        for txt in ssd_cfg.train_path_gts_path:
            if not os.path.isfile(txt):
                print(f'WARNNING cannot find {txt}.')
            print(f'START loading {txt}...')
            with open(txt, 'r') as f:
                train_str = f.readlines()[0].strip()
            dic_train_path_gts = json.loads(train_str)  # {path_loc: gts[(r1, c1, r2, c2), ...], ...}
            train_path_gts += self._dic_to_path_gts(dic_train_path_gts)

        for txt in ssd_cfg.test_path_gts_path:
            if not os.path.isfile(txt):
                print(f'WARNNING cannot find {txt}.')
            print(f'START loading {txt}...')
            with open(txt, 'r') as f:
                test_str = f.readlines()[0].strip()
            dic_test_path_gts = json.loads(test_str)  # {path_loc: gts[(r1, c1, r2, c2), ...], ...}
            test_path_gts += self._dic_to_path_gts(dic_test_path_gts)

        return train_path_gts, test_path_gts

    def _dic_to_path_gts(self, path_gts: dict):
        """
        :param path_gts: {path_loc: gts[(r1, c1, r2, c2), ...], ...}
        :return: [(path_loc, gts), (path_loc, gts), ...]
        """
        new_path_gts = [(k, path_gts[k]) for k in path_gts]

        return new_path_gts

    def produce_train(self):
        while not self.should_stop:
            data = next(self.train_samples)
            with self.has_empty:
                while self.buffer.full():
                    if self.should_stop:
                        return
                    self.has_empty.wait()
                if self.should_stop:
                    return
                self.buffer.put(data)
                self.has_data.notify_all()

    @property
    def train_num(self):
        return len(self.train_path_gts)

    @property
    def test_num(self):
        return len(self.test_path_gts)

    def next_batch(self, batch_size=1, datasets='train'):
        if datasets == 'train':
            with self.has_data:
                while self.buffer.empty():
                    if self.should_stop:
                        return
                if self.should_stop:
                    return
                img, gt_infos, cla_labels, reg_labels = self.buffer.get()
                self.has_empty.notify_all()
        else:
            img, gt_infos, cla_labels, reg_labels = next(self.test_samples)
        return np.expand_dims(img, axis=0), gt_infos, cla_labels, reg_labels

    def get_data(self, cate='train'):
        if cate not in ('train', 'test'):
            raise ValueError("'datasets' should be in ('train', 'test').")

        path_gts = self.train_path_gts if cate == 'train' else self.test_path_gts

        while True:
            random.shuffle(path_gts)
            for img_path, gt_infos in path_gts:
                gt_infos = np.array(gt_infos)
                img = cv2.imread(img_path)
                if img is None:
                    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
                # if img is None:
                #     continue
                # 根据设定的图片最大最小尺寸，自动调节放缩
                img, resize_ratio = self.resize_img(img, ssd_cfg.max_img_size, ssd_cfg.min_img_size)
                img_shape = img.shape[:2]
                gt_infos = self._resize_gts(gt_infos, img_shape, resize_ratio)
                # 若没有新的标记框，则使用下一样本
                if gt_infos.size == 0:
                    continue
                # 返回gt标注框和rpn、fast的锚框信息
                # 根据rpn选中的正负样本提取fast的分类、回归位置和标签
                # 获取样本gt信息
                # 训练时从样本说明文件可直接得到：

                # gt_info [num_boxes, 4] -> r1, c1, r2, c2

                # 根据样本gt信息和图片尺寸，得到所有锚框的分类回归标签
                # cla_labels [sum_anchor_box] -> -1未选中，0负，1正
                # reg_labels [sum_anchor_box, 4] -> tr, tc, th, tw
                cla_labels, reg_labels = ssd_target_layer(img_shape, gt_infos)

                # 数据增强
                if self.should_aug and cate == 'train':
                    img = self._img_augmentation(img)

                yield img, gt_infos, cla_labels, reg_labels

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

    def _resize_gts(self, gt_infos, img_shape, resize_ratio):
        new_gt_infos = []
        for i, gt_loc in enumerate(gt_infos):
            r1, c1, r2, c2 = get_resized_gt(gt_loc, resize_ratio, img_shape)
            # 缩放后的gt需要满足最小尺寸
            if r2 - r1 > ssd_cfg.test_min_size or c2 - c1 > ssd_cfg.test_min_size:
                new_gt_infos.append([r1, c1, r2, c2])

        return np.array(new_gt_infos)

    def _img_augmentation(self, img):

        # 生成所有的随机数
        rm = 100
        random_range = np.random.randint(0, rm + 1, size=4) / rm  # [0, 1]

        # # 是否按照短边裁剪成正方形
        # if ssd_cfg.aug_img_cut:
        #     h, w = img.shape[:2]
        #     min_lin = min(h, w)
        #     # 先裁剪成矩形，再放缩到cnn_shape
        #     h_start = random.randint(0, h - min_lin - 1) if h > min_lin else 0
        #     w_start = random.randint(0, w - min_lin - 1) if w > min_lin else 0
        #     img = img[w_start:w_start + min_lin, h_start:h_start + min_lin, :]

        # 是否进行数据增强
        if random_range[0] < ssd_cfg.aug_img_prob:
            return img

        # # 放缩到cnn_shape+random_add。h和w的add不同，这样就包含了拉伸
        # random_add_h = int(random_range[1] * ssd_cfg.aug_img_add)
        # random_add_w = int(random_range[2] * ssd_cfg.aug_img_add)
        # img = cv2.resize(img, (ssd_cfg.cnn_shape + random_add_w, ssd_cfg.cnn_shape + random_add_h),
        #                  interpolation=cv2.INTER_AREA)
        # # 裁剪到cnn_shape
        # random_add_h = int(random_range[3] * random_add_h)
        # random_add_w = int(random_range[4] * random_add_w)
        # img = img[random_add_h:random_add_h + ssd_cfg.cnn_shape, random_add_w:random_add_w + ssd_cfg.cnn_shape, :]

        # 亮度和对比度
        if ssd_cfg.aug_img_contrast:  # [2-contrast, contrast]
            n = 2 / ssd_cfg.aug_img_contrast - 1
            m = 1 - n
            contrast = (random_range[1] * m + n) * ssd_cfg.aug_img_contrast
            img = img * contrast
        if ssd_cfg.aug_img_brightness:  # [-brightness, brightness]
            brightness = (random_range[2] * 2 - 1) * ssd_cfg.aug_img_brightness
            img = img + brightness
        if ssd_cfg.aug_img_contrast + ssd_cfg.aug_img_brightness:
            img = np.clip(img.astype(int), 0, 255)

        # # 翻转
        # if random_range[7] < ssd_cfg.aug_img_flip:  # 水平翻转
        #     img = cv2.flip(img, 1)
        # if random_range[8] < ssd_cfg.aug_img_flip:  # 垂直翻转
        #     img = cv2.flip(img, 0)
        #
        # # 旋转
        # if ssd_cfg.aug_img_rotate:
        #     angle = (random_range[9] * 2 - 1) * ssd_cfg.aug_img_rotate
        #     height, width = img.shape[:2]
        #     center = (width / 2, height / 2)
        #     M = cv2.getRotationMatrix2D(center, angle, scale=1)
        #     img = cv2.warpAffine(img, M, (width, height))

        # 噪音
        if ssd_cfg.aug_img_noise:
            height, width = img.shape[:2]
            num_noise = int(random_range[3] * ssd_cfg.aug_img_noise * height * width)
            h = np.random.randint(0, height, size=num_noise)
            w = np.random.randint(0, width, size=num_noise)
            img[h, w, :] = 255

        return img

    def close(self):
        self.should_stop = True

        with self.has_data:
            self.has_empty.notify_all()
            self.has_data.notify_all()

        self.th.join()

    def remove_aug(self):
        self.should_aug = False
        for i in range(ssd_cfg.train_buffer_capacity*2):
            self.next_batch()
            time.sleep(0.1)
        print('remove aug.')

    def return_aug(self):
        self.should_aug = True
        for i in range(ssd_cfg.train_buffer_capacity*2):
            self.next_batch()
            time.sleep(0.1)
        print('return aug.')


if __name__ == '__main__':
    ## 设置属性防止中文乱码
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    sa = Samples()
    print(f'train_num: {sa.train_num}')
    print(f'test_num: {sa.test_num}')
    time.sleep(1)

    # sa.test_all_files()
    for cate in ('train', ):
        print(f'{cate} samples...')
        for _ in range(10):
            imgs, gts, cla_labels, reg_labels = sa.next_batch(datasets=cate)
            img = imgs[0].copy()
            total_anchor_boxes = get_all_anchor_boxes(img.shape[:2])
            # 画标记框
            for gt in gts:
                r1, c1, r2, c2 = gt.astype(int)
                img = cv2.rectangle(img, (c1, r1), (c2, r2), [0, 0, 255], thickness=2)
            plt.imshow(img[:, :, ::-1])
            plt.title(f'{cate} with {gts.shape[0]} gts')
            plt.show()
            # # 画前景
            # img2 = img.copy()
            # fg_inds = np.where(cla_labels > 0)[0]
            # for box in total_anchor_boxes[fg_inds]:
            #     r1, c1, r2, c2 = box.astype(int)
            #     img2 = cv2.rectangle(img2, (c1, r1), (c2, r2), [0, 255, 0], thickness=2)
            # plt.imshow(img2[:, :, ::-1])
            # plt.title(datasets + f'fg{fg_inds.size}')
            # plt.show()
            # # 画后景
            # img3 = img.copy()
            # bg_inds = np.where(cla_labels == 0)[0]
            # for box in total_anchor_boxes[bg_inds]:
            #     r1, c1, r2, c2 = box.astype(int)
            #     img3 = cv2.rectangle(img3, (c1, r1), (c2, r2), [255, 0, 0], thickness=2)
            # plt.imshow(img3[:, :, ::-1])
            # plt.title(datasets + f'fg{fg_inds.size}_bg{bg_inds.size}')
            # plt.show()

    time.sleep(5)

    sa.close()
