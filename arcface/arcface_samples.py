# -*- coding: utf-8 -*-
import threading, queue, os, cv2, random, time, math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import shutil
from arcface.arcface_config import arcface_cfg
from util.img_utils import get_img_augmentation, cv2_imread


class FaceRecSamples:
    def __init__(self):
        self.total_path_label, self.train_path_label, self.test_path_label, \
        self.dict_label_path = self.get_dirs()
        # # [(path, label), ...], ...
        # # dict_label_path {'total':{label:[path], ...}, 'train':{...}, 'test':{...}}
        self.train_datas = self.get_data(cate='train')
        self.test_datas = self.get_data(cate='test')

        self.should_stop = False
        self.not_aug = False  # 控制是否进行数据增强

        # 线程
        self.buffer = queue.Queue(arcface_cfg.train_buffer_capacity)
        self.th = threading.Thread(target=self.produce_data)
        lock = threading.RLock()
        self.has_data = threading.Condition(lock)
        self.has_empty = threading.Condition(lock)
        self.th.start()

    def get_dirs(self):
        total_path_label = self._read_path_txt(arcface_cfg.path_label_txt.format(datasets='total'))
        train_path_label = self._read_path_txt(arcface_cfg.path_label_txt.format(datasets='train'))
        test_path_label = self._read_path_txt(arcface_cfg.path_label_txt.format(datasets='test'))
        dict_label_path = {'total': self.get_label_path(total_path_label),
                           'train': self.get_label_path(train_path_label),
                           'test': self.get_label_path(test_path_label)}

        return total_path_label, train_path_label, test_path_label, dict_label_path

    def get_label_path(self, path_label):
        label_path = {}
        for path, label in path_label:
            if label not in label_path: label_path[label] = []
            label_path[label].append(path)
        return label_path

    def _read_path_txt(self, path):
        with open(path, 'r') as f:
            _path_label = f.readlines()
        path_label = [p.strip().split(',') for p in _path_label]
        return path_label

    def get_data(self, cate='train'):
        path_label = self.train_path_label.copy() if cate == 'train' else self.test_path_label.copy()
        random_ranges = np.random.randint(0, 1001, size=[5000, 11]) / 1000.
        while not self.should_stop:
            random.shuffle(path_label)  # 随机文件
            for ind, _path_label in enumerate(path_label):
                if self.should_stop:
                    break
                # 生成所有的随机数，用来做数据增强
                if cate == 'train':
                    not_aug = self.not_aug  # 放在循环内是因为在最后计算整个数据集时，需要设定不数据增强
                    if ind % 5000 == 0:
                        random_ranges = np.random.randint(0, 1001, size=[5000, 11]) / 1000.
                else:
                    not_aug = True
                try:  # 可能会删除一些图片。但已验证过，图片全都可用
                    file_path, label = _path_label
                    img = cv2_imread(file_path)
                    img = get_img_augmentation(img.astype(np.uint8), random_ranges[ind % 5000],
                                               not_aug,
                                               **arcface_cfg.img_aug_params)
                    yield img, int(label)
                except:
                    continue

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

    def next_batch(self, batch_size=1, cate='train', should_sort=False):
        imgs = [0 for _ in range(batch_size)]
        labels = [0 for _ in range(batch_size)]

        for ind in range(batch_size):
            with self.has_data:
                while self.buffer.empty():
                    if self.should_stop:
                        return
                    self.has_data.wait()
                if self.should_stop:
                    return
                if cate == 'train':
                    img, label = self.buffer.get()
                elif cate == 'test':
                    img, label = next(self.test_datas)
                self.has_empty.notify_all()
            imgs[ind] = img
            labels[ind] = label

        imgs, labels = np.array(imgs), np.array(labels)
        if should_sort:
            _ind = np.argsort(labels)
            imgs, labels = imgs[_ind], labels[_ind]

        return imgs, labels

    @property
    def train_num(self):
        return len(self.train_path_label)

    @property
    def test_num(self):
        return len(self.test_path_label)

    def close(self):
        self.should_stop = True

        # 关键语句
        with self.has_data:
            self.has_empty.notify_all()
            self.has_data.notify_all()

        self.th.join()

    def check_files(self, save_dir):
        print(f'START: check total samples.')
        for path_label in (self.train_path_label, self.test_path_label):
            for file_path, label in path_label:
                try:
                    img = cv2_imread(file_path)
                    if img is None:
                        print(f'DELETE: read None from {file_path}.')
                        os.remove(file_path)
                        continue
                    img = get_img_augmentation(img, np.random.randint(0, 101, size=11) / 100.,
                                               **arcface_cfg.img_aug_params)
                except:
                    print(f'COPY: cannot augment {file_path}.')
                    shutil.copyfile(file_path, save_dir)
        print(f'FINISH: check total samples.')

    def check_dir(self, dir):
        print(f'START: check dir {dir}.')
        dir_names = [dir + n + '/' for n in os.listdir(dir)
                     if 'face' in n and os.path.isdir(dir + n)]
        for dir_name in dir_names:
            print(f'START: check dir {dir_name}.')
            file_paths = [dir_name + n for n in os.listdir(dir_name) if os.path.isfile(dir_name + n)]
            for file_path in file_paths:
                try:
                    img = cv2_imread(file_path)
                    if img is None:
                        print(f'DELETE: read None from {file_path}.')
                        os.remove(file_path)
                        continue
                    img = get_img_augmentation(img, np.random.randint(0, 101, size=11) / 100.,
                                               **arcface_cfg.img_aug_params)
                except:
                    print(f'DELETE: cannot augment {file_path}.')
                    os.remove(file_path)
            print(f'FINISH: check dir {dir_name}.')
        print(f'FINISH: check dir {dir}.')

    def get_imgs_by_class(self, cls, num=None, datasets='total'):
        """
        # 根据指定cls得到所处datasets的num张图片
        :param cls:
        :param num: 默认None表示全部图片
        :param datasets:
        :return:
        """
        path_dict = self.dict_label_path[datasets]  # {label:[path], ...}
        paths = path_dict[f' {cls}']
        if num is not None: paths = paths[:num]
        imgs = []
        for path in paths:
            try:
                img = cv2_imread(path)
                img = cv2.resize(img, (arcface_cfg.cnn_shape, arcface_cfg.cnn_shape), interpolation=cv2.INTER_AREA)
                imgs.append(img)
            except:
                continue
        return np.array(imgs), [cls] * len(imgs)

    def get_imgs_for_similarity_from_datasets(self, num=6, same=3, datasets='total'):
        # 得到用于计算相似度的几张图片，两张同一人，剩下随机
        path_labels = self.get_paths_for_similarity(num=num, same=same,
                                                    dict_label_path=self.dict_label_path[datasets])
        imgs, labels = [], []
        for path, label in path_labels:
            try:
                img = cv2_imread(path)
                img = cv2.resize(img, (arcface_cfg.cnn_shape, arcface_cfg.cnn_shape), interpolation=cv2.INTER_AREA)
                imgs.append(img)
                labels.append(label)
            except:
                continue
        return np.array(imgs), labels

    def get_paths_for_similarity(self, num, same, dict_label_path):
        # 得到用于计算相似度的几张图片，same张是同一人，剩下随机
        labels = list(dict_label_path.keys()).copy()
        random.shuffle(labels)
        label_path = dict_label_path[labels[0]]  # 相同的标签的文件路径
        label_paths = [dict_label_path[labels[i]] for i in range(1, num - same + 1)]
        path_labels = [(random.choice(label_path), labels[0]) for _ in range(same)]
        path_labels += [(random.choice(label_paths[i]), l) for i, l in enumerate(labels[1:num - same + 1])]
        return path_labels

    def remove_img_aug(self):
        self.not_aug = True
        for _ in range(arcface_cfg.train_buffer_capacity):
            self.next_batch(arcface_cfg.train_batch_size, cate='train')
            time.sleep(0.1)
        print('remove img aug.')

    def return_img_aug(self):
        self.not_aug = False
        for _ in range(arcface_cfg.train_buffer_capacity):
            self.next_batch(arcface_cfg.train_batch_size, cate='train')
            time.sleep(0.1)
        print('return img aug.')


if __name__ == '__main__':
    ## 设置属性防止中文乱码
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    sa = FaceRecSamples()
    print('train num:', sa.train_num)
    print('test num:', sa.test_num)
    time.sleep(1)

    path_dict = sa.dict_label_path['test']
    print(path_dict)


    # # move_dir = 'E:/TEST/AI/datasets/to_be_deled/'
    # # sa.check_files(move_dir)
    # dir = 'E:/TEST/AI/datasets/jpface/'
    # sa.check_dir(dir)

    # imgs, labels1 = sa.get_imgs_for_similarity_from_datasets(datasets='test')
    # num = imgs.shape[0]
    # cols = min(num, 3)
    # rows = math.ceil(num / cols)
    # for ind, img in enumerate(imgs):
    #     plt.subplot(rows, cols, ind + 1)
    #     plt.imshow(img[:, :, ::-1])
    #     plt.title(f'{labels1[ind]}')
    #     plt.axis('off')
    # plt.show()

    print('Train sa...')
    imgs, y = sa.next_batch(16, 'train', should_sort=True)
    for img, label in zip(imgs, y):
        plt.imshow(img[:, :, ::-1])
        plt.title(f'train samples_{label}')
        plt.show()

    # print('Test sa...')
    # imgs, labels_onehot = sa.next_batch(100, 'test')
    # for img, label in zip(imgs, labels_onehot):
    #     plt.imshow(img[:, :, ::-1])
    #     plt.datasets(f'test samples_{label}')
    #     plt.show()

    time.sleep(2)
    sa.close()
