# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def change_train_num(num_dis, num_gen, loss_d, loss_g, d_limit_loss, g_better_loss):
    ex_min, d_min, d_max, ex_max = d_limit_loss  # (-5, -3.5, -2, -0.5)
    g_min, g_max = g_better_loss  # (-1, 0)

    if loss_d < ex_min:
        # self.num_app_train_dis = num_ex_min
        num_dis = max(1, num_dis - 2)
        print(f'判别器平均损失 {loss_d:.3f} 极强，训练次数改为 {num_dis}', end='\t')
    elif loss_d < d_min:
        # self.num_app_train_dis = config.num_to_weaker
        num_dis = max(1, num_dis - 1)
        print(f'判别器平均损失 {loss_d:.3f} 略强，训练次数改为 {num_dis}', end='\t')
    elif loss_d > ex_max:
        # self.num_app_train_dis = num_ex_max
        num_dis += 2
        print(f'判别器平均损失 {loss_d:.3f} 极弱，训练次数改为 {num_dis}', end='\t')
    elif loss_d > d_max:
        # self.num_app_train_dis = config.num_to_stronger
        num_dis += 1
        print(f'判别器平均损失 {loss_d:.3f} 略弱，训练次数改为 {num_dis}', end='\t')
    else:
        print(f'判别器平均损失 {loss_d:.3f} 适中，训练次数保持不变 {num_dis}', end='\t')

    if loss_g < g_min:
        # self.num_app_train_gen = config.num_to_weaker
        num_gen = max(1, num_gen - 1)
        print(f'生成器平均损失 {loss_g:.3f} 略强，训练次数改为 {num_gen}')
    elif loss_g > g_max:
        # self.num_app_train_gen = config.num_to_stronger
        num_gen += 1
        print(f'生成器平均损失 {loss_g:.3f} 略弱，训练次数改为 {num_gen}')
    else:
        print(f'生成器平均损失 {loss_g:.3f} 适中，训练次数保持不变 {num_gen}')

    return num_dis, num_gen


def print_result(epoch, epochs, batch, num_batch, loss_dt=0, loss_dg=0, loss_g=0, loss_cyc=0,
                 loss_iden=0, loss_wgan_gp=0):
    print(f'\rEpoch {epoch + 1}/{epochs}  Batch {batch + 1}/{num_batch}'
          f' - Loss dt = {loss_dt:.6f} dg = {loss_dg:.6f} g = {loss_g:.6f}'
          f' cyc = {loss_cyc:.6f} wgan_gp = {loss_wgan_gp:.6f} identity = {loss_iden:.6f}',
          end='')


def plot_imgs(imgs, name=['test_x', 'test_y', 'train_x', 'train_y']):
    nb = len(imgs)
    plt.figure(figsize=[18, 18])
    for i, img in enumerate(imgs):
        plt.subplot(nb, 1, i + 1)
        plt.imshow(img[:, :, ::-1])
        plt.title(name[i])
    plt.show()


def get_img(imgs, _max):
    # 将图片集imgs转化到0~_max范围
    imgs2 = []
    for img in imgs:
        img = np.squeeze(img)
        img_max = np.max(img)
        img_min = np.min(img)
        img = (img - img_min) / (img_max - img_min) * _max
        img = np.uint8(img)
        imgs2.append(img)
    return imgs2


def get_line(imgs):
    img = None
    for _img in imgs:
        if img is None:
            img = _img
        else:
            img = np.concatenate((img, _img), axis=1)
    return img
