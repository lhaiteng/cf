# -*- coding: utf-8 -*-
import random, cv2, os
import numpy as np


def cv2_imread(file_path):
    # 使用cv2读取图片。首次错误时使用np转换一下再读取
    img = cv2.imread(file_path)
    if img is None:
        img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return img


def get_img_augmentation(img, random_range, not_aug=False,
                         target_shape=128,
                         should_cut=False, aug_prob=0.7, shape_add=16, flip_1=0.5, flip_0=0.1,
                         max_rotate=8, max_contrast=1.2, max_brightness=10, max_noise_ratio=0.001):
    """

    :param img:每次变换一张图片
    :param random_range: [11]  为后续的变换提供参数
    :param not_aug: 决定是否数据增强
    :param target_shape: 变换后的图片尺寸
    :param should_cut: 是否按照短边裁剪成正方形
    :param aug_prob: 概率小于此值时，进行数据增强;否则只放缩到cnn_shape
    :param shape_add: 先放缩到cnn_shape+random_add，再裁剪到cnn_shape。add是random_add的最大值
    :param flip_1: 水平翻转的概率
    :param flip_0: 垂直翻转的概率
    :param max_rotate: 旋转最大角度
    :param max_contrast: 对比度最大值
    :param max_brightness: 亮度增大最大值。255制。
    :param max_noise_ratio: 图片中增加噪音点的最大百分比
    :return:
    """
    # 是否按照短边裁剪成正方形
    if should_cut:
        h, w = img.shape[:2]
        min_lin = min(h, w)
        # 先裁剪成矩形，再放缩到cnn_shape
        h_start = random.randint(0, h - min_lin - 1) if h > min_lin else 0
        w_start = random.randint(0, w - min_lin - 1) if w > min_lin else 0
        img = img[w_start:w_start + min_lin, h_start:h_start + min_lin, :]

    if type(target_shape) in (list, tuple):
        target_shape_r, target_shape_c, channels = target_shape
    elif target_shape is None:
        target_shape_r, target_shape_c, channels = img.shape
    else:
        target_shape_r = target_shape_c = target_shape
    if not_aug or random_range[0] > aug_prob:  # 不进行数据增强，直接放缩到224
        if target_shape is not None:
            img = cv2.resize(img, (target_shape_c, target_shape_r), interpolation=cv2.INTER_AREA)
        return img

    # 放缩到cnn_shape+random_add。h和w的add不同，这样就包含了拉伸
    random_add_h = random_add_w = 0
    if shape_add:
        random_add_h = int(random_range[1] * shape_add)
        random_add_w = int(random_range[2] * shape_add)
        img = cv2.resize(img, (target_shape_c + random_add_w, target_shape_r + random_add_h),
                         interpolation=cv2.INTER_AREA)
    # 若有放缩拉伸，且指定了目标尺寸，则再裁剪
    if (random_add_h or random_add_w) and target_shape is not None:
        # 裁剪到cnn_shape
        random_add_h = int(random_range[3] * random_add_h)
        random_add_w = int(random_range[4] * random_add_w)
        img = img[random_add_h:random_add_h + target_shape_r,
              random_add_w:random_add_w + target_shape_c, :]

    # 翻转
    if random_range[5] < flip_1:  # 水平翻转
        img = cv2.flip(img, 1)
    if random_range[6] < flip_0:  # 垂直翻转
        img = cv2.flip(img, 0)

    # 旋转
    if max_rotate:
        angle = (random_range[7] * 2 - 1) * max_rotate
        height, width = img.shape[:2]
        center = (width / 2, height / 2)
        M = cv2.getRotationMatrix2D(center, angle, scale=1)
        img = cv2.warpAffine(img, M, (width, height))

    # 亮度和对比度
    if max_contrast:
        n = 2 / max_contrast - 1
        m = 1 - n
        contrast = (random_range[8] * m + n) * max_contrast
        img = img * contrast
    if max_brightness:
        brightness = (random_range[9] * 2 - 1) * max_brightness
        img = img + brightness
    if max_contrast + max_brightness:
        img = np.clip(img, 0, 255)

    # 噪音
    if max_noise_ratio:
        height, width = img.shape[:2]
        num_noise = int(random_range[10] * max_noise_ratio * height * width)
        if num_noise:
            h = np.random.randint(0, height, size=num_noise)
            w = np.random.randint(0, width, size=num_noise)
            img[h, w, :] = 255

    return img.astype(np.uint8)


def make_up_dirs(root_dir, random_ranges=None, img_aug_params=None, target_num=60):
    # 根目录文件夹，其下是类别文件夹，再向下是图片
    for ind, dname in enumerate(os.listdir(root_dir)):
        dir_path = root_dir + dname + '/'
        if os.path.isdir(dir_path):
            make_up_dir(dir_path, random_ranges, img_aug_params, target_num)


def make_up_dir(dir_path, random_ranges=None, img_aug_params=None, target_num=60):
    if random_ranges is None: random_ranges = np.random.randint(0, 1001, size=[target_num, 11]) / 1000.
    if img_aug_params is None:
        img_aug_params = {'target_shape': None,  # 只放缩，不裁剪
                          'should_cut': False,  # 是否按照短边裁剪成正方形
                          'aug_prob': 1,  # 概率小于此值时，进行数据增强;否则只放缩到cnn_shape
                          'shape_add': 16,  # 先放缩到cnn_shape+random_add，再裁剪到cnn_shape。add是random_add的最大值
                          'flip_1': 0.01,  # 水平翻转的概率
                          'flip_0': 0.01,  # 垂直翻转的概率
                          'max_rotate': 8,  # 旋转最大角度
                          'max_contrast': 1.1,  # 对比度最大值
                          'max_brightness': 5,  # 亮度增大最大值。255制。
                          'max_noise_ratio': 0.00001}  # 图片中增加噪音点的最大百分比
    fpaths = [dir_path + fpath for fpath in os.listdir(dir_path) if os.path.isfile(dir_path + fpath)]
    num = len(fpaths)
    add_num = target_num - num
    add_no = 0
    if add_num > 0:
        while add_no < add_num:
            root_fpath = fpaths[add_no % num]
            img = cv2.imread(root_fpath)
            if img is None:
                img = cv2.imdecode(np.fromfile(root_fpath, dtype=np.uint8), -1)
            # try:
            img = get_img_augmentation(img, random.choice(random_ranges), **img_aug_params)
            cv2.imwrite(f'{dir_path}Zmakeup_{add_no}.png', img)
            add_no += 1
            # except:
            #     continue
        print(f'make up {add_num} pics for {dir_path}.')
    else:
        print(f'{dir_path} alread has enough pics.')


def del_img_with_dirs(root_dir, keys='Zmakeup'):
    for dname in os.listdir(root_dir):
        dir_path = root_dir + dname + '/'
        if os.path.isdir(dir_path):
            del_img_with_dir(dir_path, keys)


def del_img_with_dir(dir, keys='Zmakeup'):
    for fname in os.listdir(dir):
        file_path = dir + fname
        if keys in fname:
            os.remove(file_path)


if __name__ == '__main__':
    root_dir = 'E:/TEST/AI/datasets/krface_face/'
    make_up_dirs(root_dir)
