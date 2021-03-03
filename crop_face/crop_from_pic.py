# -*- coding: utf-8 -*-
from cyclegan.dl16_CF_cyclegan_samples import CycleGanSamples as Sample
import cv2, os


def crop_pic(save_dir):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    sa = Sample()
    train_num = sa.train_num
    print(f'START to save {train_num} train pics...')
    for ind in range(train_num):
        print(f'\rsaving {ind}...', end='')
        _, img = sa.next_batch(1, 'train')
        save_path = save_dir + f'{ind}.png'
        cv2.imwrite(save_path, img[0])
    print(f'START to save {sa.test_num} test pics...')
    for ind in range(sa.test_num):
        print(f'\rsaving {ind}...', end='')
        _, img = sa.next_batch(1, 'test')
        save_path = save_dir + f'{train_num + ind}.png'
        cv2.imwrite(save_path, img[0])

    sa.close()


if __name__ == '__main__':
    save_dir = 'E:/TEST/AI/datasets/changeface/piaozhiyan_face/'
    crop_pic(save_dir)
