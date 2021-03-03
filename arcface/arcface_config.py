# -*- coding: utf-8 -*-
import os
import numpy as np


class Config:
    def __init__(self):
        self.name = 'dl16_CF_facerec'
        self.version = '49'

        self.vgg16_npy_path = 'E:/TEST/AI/dl/05_transfer/tensorflow_vgg/vgg16.npy'

        self.cnn_shape = 112  # 图片分辨率128

        self.parameter_samples()
        self.parameter_out_model()
        self.parameter_path()
        self.parameters()

    def parameter_samples(self):
        # 数据增强
        self.img_aug_params = {'target_shape': self.cnn_shape,
                               'should_cut': False,  # 是否按照短边裁剪成正方形
                               'aug_prob': 0.5,  # 概率小于此值时，进行数据增强;否则只放缩到cnn_shape
                               'shape_add': 16,  # 先放缩到cnn_shape+random_add，再裁剪到cnn_shape。add是random_add的最大值
                               'flip_1': 0.5,  # 水平翻转的概率
                               'flip_0': 0.01,  # 垂直翻转的概率
                               'max_rotate': 8,  # 旋转最大角度
                               'max_contrast': 1.2,  # 对比度最大值
                               'max_brightness': 20,  # 亮度增大最大值。255制。
                               'max_noise_ratio': 0.002}  # 图片中增加噪音点的最大百分比

    def parameter_out_model(self):
        self.epsilon = 1e-12
        self.epochs = 100
        self.num_cls = self.get_num_cls()
        self.train_batch_size = 64  # 8
        self.test_batch_size = 128  # 64
        self.num_train_batch_expand = 1  # num_batch的扩大倍数
        self.show_num = int(20e4)  # 训练多少图片显示一次预测图片。0为不显示
        self.num_to_print_losses = 0  # 每几个batch打印一次损失。0表示每代都打印
        self.train_buffer_capacity = int(self.train_batch_size * 10)


    def parameters(self):
        # 优化器参数
        # self.beta1 = 0
        # self.beta2 = 0.999
        # 权重滑动更新的参数
        self.ema_decay = 0  # 1-5e-4。0表示不用权重滑动更新
        # 图片色彩通道的滑动平均
        self.move_imgs_decay = 0  # 0.995。0表示不求色彩滑动平均
        # 变量初始化器
        self.init_mean = 0
        self.init_std = 0.02  # 0.02

        self.keep_prob = 0.5
        self.leaky_slop = 0.2

        self.id_size = 128
        self.resnet_filters_base = 64
        self.resnet_layers = (3, 4, 6, 3)  # (3, 4, 6, 3)

        # 学习率参数
        warm_up_stage = 4
        cycle_stage = 1  # 1表示一条cos曲线至结束，lr变化较多阶段更平滑。还需要在main中更改LR.get_lr的cate
        decay_steps = (self.epochs - warm_up_stage) // cycle_stage
        max_lr = 0.01
        self.lr_para = {'start_lr': max_lr / 100, 'lr': max_lr, 'end_lr': max_lr / 100,
                        'warm_up_stage': warm_up_stage,
                        'cycle': True,
                        'decay_steps': decay_steps,
                        'decay_rate': 0.9,
                        'power': 0.5}
        # 权重的l2正则化损失系数
        self.l2_loss_factor = 5e-3
        # features l1 loss
        self.l1_loss_factor_id = 0  # 5e-4
        self.l1_loss_factor_w = 0  # 5e-4
        # center loss
        self.center_loss_factor = 5e-3  # 5e-3
        self.center_alpha = 0.9  # center update rate
        self.center_stage = warm_up_stage  # 只累积prelogit_center而不计算center_loss
        # focal loss
        self.fl_alpha = None  # 1, None表示只用crossentropy loss不用focal loss
        self.fl_gamma = None  # 2, None表示只用crossentropy loss不用focal loss
        # arcface参数 标签项改为(cos(m1*thetas+m2) - m3)，其余不变
        scale_steps = m1_steps = m2_steps = m3_steps = warm_up_stage + decay_steps * 2 // 3
        self.arcface_para = {'scale_start': 2, 'scale_end': 32, 'scale_steps': scale_steps,
                             'm1_start': 1, 'm1_end': 1, 'm1_steps': m1_steps,
                             'm2_start': 0.001, 'm2_end': 0.3, 'm2_steps': m2_steps,
                             'm3_start': 0.001, 'm3_end': 0.2, 'm3_steps': m3_steps}



    def parameter_path(self):
        self.path_label_txt = './arcface_path_label_{datasets}.txt'
        self.test_imgs_dir = 'E:/TEST/AI/datasets/test_face2/'
        self.data_dir = 'E:/TEST/AI/datasets/'
        self.face_names = ['cnface_face', 'jpface_face', 'krface_face']

        self.save_path = f'./models/{self.name}_{self.version}'

        self.log_dir = f'./logs/v{self.version}/'
        self.gen_dir = f'./gen_pics/v{self.version}/'
        self.record_dir = f'./record/v{self.version}/'

    # 检查需要的所有文件夹。因为可能会被别的地方引用，所以使用时要手动检查。
    def check_all_dir_paths(self):
        self.check_dir(self.log_dir)
        self.check_dir(self.gen_dir)
        self.check_dir(self.record_dir)

    def check_dir(self, dir):
        if os.path.isdir(dir):
            print(f'Dir {dir} exists.')
        else:
            os.makedirs(dir)
            print(f'Dir {dir} not exists. Create {dir}')

    def get_gpu_num(self):
        value = os.getenv('CUDA_VISIBLE_DEVICES', '0')
        return len(value.split(','))

    def get_num_cls(self):
        return 782  # 284


arcface_cfg = Config()

if __name__ == '__main__':
    print(f'LR para:\n{arcface_cfg.lr_para}')
    print(f'arcface para:\n{arcface_cfg.arcface_para}')
