# -*- coding: utf-8 -*-
from unet.unet_config import unet_cfg as cfg
import queue, threading, random, cv2, os, time
import numpy as np
import matplotlib.pyplot as plt


class Samples:
    def __init__(self):
        self.train_path, self.test_path = self.get_paths()

        self.train_data_gen = self.get_data(cate='train')
        self.test_data_gen = self.get_data(cate='test')

        self.should_stop = False

        self.buffer = queue.Queue(cfg.train_buffer_capacity)
        self.threading = threading.Thread(target=self.produce_data)
        lock = threading.Lock()
        self.has_data = threading.Condition(lock)
        self.has_empty = threading.Condition(lock)

        self.threading.start()

    def get_paths(self):
        train_path = self._read_txt(cfg.path_txt.format(cate='train'))
        test_path = self._read_txt(cfg.path_txt.format(cate='test'))
        return train_path, test_path

    def _read_txt(self, path):
        with open(path, 'r') as f:
            txt = f.readlines()
        return [t.strip() for t in txt]

    def get_data(self, cate):
        if cate == 'train':
            paths = self.train_path.copy()
        else:
            paths = self.test_path.copy()
        random_ranges = 0
        while not self.should_stop:
            random.shuffle(paths)
            for ind, path in enumerate(paths):
                if ind % 5000 == 0:
                    if cate == 'train':
                        random_ranges = np.random.randint(0, 1001, size=[5000, 11]) / 1000
                    else:
                        random_ranges = np.ones(shape=[5000, 1])
                # try:
                img = cv2.imread(path)
                if img is None:
                    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
                img = self._img_augmentation(img.astype(np.uint8), random_ranges[ind % 5000])
                yield img
                # except:
                #     continue

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
        # 若有放缩拉伸，则再裁剪
        if random_add_h or random_add_w:
            # 裁剪到cnn_shape
            random_add_h = int(random_range[3] * random_add_h)
            random_add_w = int(random_range[4] * random_add_w)
            img = img[random_add_h:random_add_h + cfg.cnn_shape,
                  random_add_w:random_add_w + cfg.cnn_shape, :]

        # 翻转
        if random_range[5] < cfg.aug_img_flip_1:  # 水平翻转
            img = cv2.flip(img, 1)
        if random_range[6] < cfg.aug_img_flip_0:  # 垂直翻转
            img = cv2.flip(img, 0)

        # 旋转
        if cfg.aug_img_rotate:
            angle = (random_range[7] * 2 - 1) * cfg.aug_img_rotate
            height, width = img.shape[:2]
            center = (width / 2, height / 2)
            M = cv2.getRotationMatrix2D(center, angle, scale=1)
            img = cv2.warpAffine(img, M, (width, height))

        # 亮度和对比度
        if cfg.aug_img_contrast:
            n = 2 / cfg.aug_img_contrast - 1
            m = 1 - n
            contrast = (random_range[8] * m + n) * cfg.aug_img_contrast
            img = img * contrast
        if cfg.aug_img_brightness:
            brightness = (random_range[9] * 2 - 1) * cfg.aug_img_brightness
            img = img + brightness
        if cfg.aug_img_contrast + cfg.aug_img_brightness:
            img = np.clip(img, 0, 255)

        # 噪音
        if cfg.aug_img_noise:
            height, width = img.shape[:2]
            num_noise = int(random_range[10] * cfg.aug_img_noise * height * width)
            h = np.random.randint(0, height, size=num_noise)
            w = np.random.randint(0, width, size=num_noise)
            img[h, w, :] = 255

        return img.astype(np.uint8)

    def produce_data(self):
        while not self.should_stop:
            data = next(self.train_data_gen)
            with self.has_empty:
                while self.buffer.full():
                    if self.should_stop:
                        return
                    self.has_empty.wait()
                if self.should_stop:
                    return
                self.buffer.put(data)
                self.has_data.notify_all()  # 注意这里是has_data

    @property
    def train_num(self):
        return len(self.train_path)

    @property
    def test_num(self):
        return len(self.test_path)

    def next_batch(self, batch_size=1, cate='train'):
        imgs = []

        for _ in range(batch_size):
            if cate == 'train':
                with self.has_data:
                    while self.buffer.empty():
                        if self.should_stop:
                            return
                        self.has_data.wait()
                    if self.should_stop:
                        return
                    img = self.buffer.get()
                    self.has_empty.notify_all()  # 注意这里是has_empty
            else:
                img = next(self.test_data_gen)
            imgs.append(img)

        return np.array(imgs)

    def check(self, check_dirs: list):
        """
        检查文件夹里列表中的所有文件夹内的图片能否正确读取
        :param check_dirs:
        :return:
        """
        check_paths = [check_dir + '/' + p
                       for check_dir in check_dirs for p in os.listdir(check_dir)
                       if os.path.isfile(check_dir + '/' + p)]
        if not check_paths:
            check_paths = self.train_path + self.test_path

        random_ranges = 0
        for ind, path in enumerate(check_paths):
            print(f'\rcheck: {path}...', end='')
            if ind % 5000 == 0:
                random_ranges = np.random.randint(0, 1001, size=[5000, 11]) / 1000
                random_ranges[:, 0] = 0  # 需要数据增强
                random_ranges[:, 5] = 0  # 需要水平翻转
                random_ranges[:, 6] = 0  # 需要竖直翻转
            try:
                img = cv2.imread(path)
                if img is None:
                    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
                if img is None:
                    print(f'\nERROR: cannot read from {path}.')
                self._img_augmentation(img, random_ranges[ind % 5000])
            except:
                print(f'\nFAILED: aug {path}.')

    def close(self):
        self.should_stop = True

        with self.has_data:
            self.has_empty.notify_all()
            self.has_data.notify_all()
        self.threading.join()


if __name__ == '__main__':
    sa = Samples()

    # time.sleep(5)
    # y = ['guobiting', 'chenhao', 'zhanghanyun', 'zhaoyihuan']
    # check_dirs = [f'E:/TEST/AI/datasets/cnface/face_{nf}' for nf in y]
    # sa.check(check_dirs)

    for cate in ('train', 'test'):
        imgs1 = np.concatenate(sa.next_batch(3, cate=cate), axis=0)[:, :, ::-1]
        imgs2 = np.concatenate(sa.next_batch(3, cate=cate), axis=0)[:, :, ::-1]
        imgs3 = np.concatenate(sa.next_batch(3, cate=cate), axis=0)[:, :, ::-1]
        imgs = np.concatenate([imgs1, imgs2, imgs3], axis=1)
        plt.imshow(imgs)
        plt.title(cate)
        plt.show()
    for i in range(10000):
        sa.next_batch(128, cate='train')
        print(f'\r{i}', end='')

    sa.close()
