# -*- coding: utf-8 -*-
import threading, queue, cv2, random, time
from aei.aei_config import aei_cfg as cfg
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


class AEISamples:
    def __init__(self):
        self.train_path, self.test_path = self.get_dirs()
        # # [(path, loc), ...], ...
        self.train_datas = self.get_data(cate='train')
        self.test_datas = self.get_data(cate='test')

        self.should_stop = False

        # 线程
        self.buffer = queue.Queue(cfg.train_buffer_capacity)
        self.th = threading.Thread(target=self.produce_data)
        lock = threading.RLock()
        self.has_data = threading.Condition(lock)
        self.has_empty = threading.Condition(lock)
        self.th.start()

    def get_dirs(self):
        train_path = self._read_path_txt(cfg.path_txt.format(cate='train'))
        test_path = self._read_path_txt(cfg.path_txt.format(cate='test'))

        return train_path, test_path

    def _read_path_txt(self, path):
        with open(path, 'r') as f:
            paths = f.readlines()
        return [p.strip() for p in paths]

    def get_data(self, cate='train'):
        """
        :param paths: [path, path, ...]
        :return:
        """
        paths = self.train_path.copy() if cate == 'train' else self.test_path.copy()
        random_ranges_src, random_ranges_tar = 0, 0
        num_path = len(paths)
        rec_num = int(num_path * cfg.rec_num)  # 样本集中相同图片的最少占比
        while not self.should_stop:
            random.shuffle(paths)  # 随机文件
            # 随机target的序号
            target_inds = np.random.randint(0, num_path, size=[num_path])
            # 随机相同的序号
            _inds = np.random.randint(0, num_path, size=[rec_num])
            target_inds[_inds] = _inds

            for ind, file_path in enumerate(paths):
                if self.should_stop:
                    break
                # 生成所有的随机数，用来做数据增强
                if ind % 5000 == 0:
                    if cate == 'train':
                        random_ranges_src = np.random.randint(0, 1001, size=[5000, 11]) / 1000.
                        random_ranges_tar = np.random.randint(0, 1001, size=[5000, 11]) / 1000.
                    else:
                        random_ranges_src = np.ones(shape=[5000, 1])
                        random_ranges_tar = np.ones(shape=[5000, 1])
                # try:
                source = cv2.imread(file_path)
                if source is None:
                    source = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
                source = self._img_augmentation(source, random_ranges_src[ind % 5000])

                target_ind = target_inds[ind]
                if target_ind == ind:
                    yield source, source, 1
                else:
                    target_path = paths[target_ind]
                    target = cv2.imread(target_path)
                    if target is None:
                        target = cv2.imdecode(np.fromfile(target_path, dtype=np.uint8), -1)
                    target = self._img_augmentation(target, random_ranges_tar[ind % 5000])
                    yield source, target, 0
                # except:
                #     continue

    def _resize_img(self, img_shape, max, min):
        # img_shape = img.shape[:2]

        # 先令最长边不大于600
        long_shape = np.max(img_shape)
        long_resize_ratio = np.minimum(max / long_shape, 1)
        # 再令最短边不小于300
        short_shape = np.min(img_shape) * long_resize_ratio
        resize_ratio = np.maximum(min / short_shape, 1) * long_resize_ratio

        # img = cv2.resize(img, (0, 0), fx=resize_ratio, fy=resize_ratio)

        return resize_ratio

    def _resize_loc(self, loc, resize_ratio, img_shape):
        h, w = img_shape
        r1, c1, r2, c2 = loc

        r1 = int(r1 * resize_ratio + resize_ratio / 2)
        c1 = int(c1 * resize_ratio + resize_ratio / 2)
        r2 = int(r2 * resize_ratio + resize_ratio / 2)
        c2 = int(c2 * resize_ratio + resize_ratio / 2)

        r1, r2 = min(r1, r2), max(r1, r2)
        c1, c2 = min(c1, c2), max(c1, c2)

        r1, c1 = max(r1, 0), max(c1, 0)
        r2, c2 = min(r2, h - 1), min(c2, w - 1)

        return r1, c1, r2, c2

    def _img_augmentation(self, img, random_range):
        # 是否按照短边裁剪成正方形
        if cfg.aug_img_cut:
            h, w = img.shape[:2]
            min_lin = min(h, w)
            # 先裁剪成矩形，再放缩到cnn_shape
            h_start = random.randint(0, h - min_lin - 1) if h > min_lin else 0
            w_start = random.randint(0, w - min_lin - 1) if w > min_lin else 0
            img = img[w_start:w_start + min_lin, h_start:h_start + min_lin, :]

        if random_range[0] > cfg.aug_img_prob:  # 不进行数据增强，直接放缩到224
            img = cv2.resize(img, (cfg.cnn_shape, cfg.cnn_shape), interpolation=cv2.INTER_AREA)
            return img

        # 放缩到cnn_shape+random_add。h和w的add不同，这样就包含了拉伸
        random_add_h = int(random_range[1] * cfg.aug_img_add)
        random_add_w = int(random_range[2] * cfg.aug_img_add)
        img = cv2.resize(img, (cfg.cnn_shape + random_add_w, cfg.cnn_shape + random_add_h),
                         interpolation=cv2.INTER_AREA)
        # 若上一步有放缩拉伸，则再裁剪
        if random_add_h or random_add_w:
            # 裁剪到cnn_shape
            random_add_h = int(random_range[3] * random_add_h)
            random_add_w = int(random_range[4] * random_add_w)
            img = img[random_add_h:random_add_h + cfg.cnn_shape, random_add_w:random_add_w + cfg.cnn_shape, :]

        # 亮度和对比度
        if cfg.aug_img_contrast:
            n = 2 / cfg.aug_img_contrast - 1
            m = 1 - n
            contrast = (random_range[5] * m + n) * cfg.aug_img_contrast
            img = img * contrast
        if cfg.aug_img_brightness:
            brightness = (random_range[6] * 2 - 1) * cfg.aug_img_brightness
            img = img + brightness
        if cfg.aug_img_contrast + cfg.aug_img_brightness:
            img = np.clip(img.astype(int), 0, 255)

        # 翻转
        if random_range[7] < cfg.aug_img_flip_1:  # 水平翻转
            img = cv2.flip(img, 1)
        if random_range[8] < cfg.aug_img_flip_0:  # 垂直翻转
            img = cv2.flip(img, 0)

        # 旋转
        if cfg.aug_img_rotate:
            angle = (random_range[9] * 2 - 1) * cfg.aug_img_rotate
            height, width = img.shape[:2]
            center = (width / 2, height / 2)
            M = cv2.getRotationMatrix2D(center, angle, scale=1)
            img = cv2.warpAffine(img.astype(np.uint8), M, (width, height))

        # 噪音
        if cfg.aug_img_noise:
            height, width = img.shape[:2]
            num_noise = int(random_range[10] * cfg.aug_img_noise * height * width)
            h = np.random.randint(0, height, size=num_noise)
            w = np.random.randint(0, width, size=num_noise)
            img[h, w, :] = 255

        return img

    def produce_data(self):
        while not self.should_stop:
            data = next(self.train_datas)
            with self.has_empty:
                while self.buffer.full():
                    if self.should_stop:
                        return
                    self.has_empty.wait()
                if self.should_stop:
                    return
                self.buffer.put(data)
                self.has_data.notify_all()

    def next_batch(self, batch_size=1, cate='train'):
        srcs = []
        targets = []
        is_recs = []

        for _ in range(batch_size):

            with self.has_data:
                while self.buffer.empty():
                    if self.should_stop:
                        return
                    self.has_data.wait()
                if self.should_stop:
                    return
                if cate == 'train':
                    src, target, is_rec = self.buffer.get()
                elif cate == 'test':
                    src, target, is_rec = next(self.test_datas)
                self.has_empty.notify_all()

            srcs.append(src)
            targets.append(target)
            is_recs.append(is_rec)

        return np.array(srcs), np.array(targets), np.array(is_recs)

    @property
    def train_num(self):
        return len(self.train_path)

    @property
    def test_num(self):
        return len(self.test_path)

    def close(self):
        self.should_stop = True

        # 关键语句
        with self.has_data:
            self.has_empty.notify_all()
            self.has_data.notify_all()

        self.th.join()


if __name__ == '__main__':
    ## 设置属性防止中文乱码
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    sa = AEISamples()
    print('train num:', sa.train_num)
    print('test num:', sa.test_num)
    time.sleep(2)

    # srcs, targets, is_recs = sa.next_batch(10, 'train')
    # for src, target, is_rec in zip(srcs, targets, is_recs):
    #     plt.imshow(np.concatenate([src, target], axis=1)[:, :, ::-1])
    #     plt.datasets(f'train samples_{is_rec}')
    #     plt.show()

    srcs, targets, is_recs = sa.next_batch(10, 'test')
    for src, target, is_rec in zip(srcs, targets, is_recs):
        plt.imshow(np.concatenate([src, target], axis=1)[:, :, ::-1])
        plt.title(f'test samples_{is_rec}')
        plt.show()

    time.sleep(5)

    sa.close()
