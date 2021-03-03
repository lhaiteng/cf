# -*- coding: utf-8 -*-
import os
import numpy as np


class Config:
    def __init__(self):
        self.name = 'dl16_CF_aei'
        self.version = '01'

        self.vgg16_npy_path = 'E:/TEST/AI/dl/05_transfer/tensorflow_vgg/vgg16.npy'

        self.cnn_shape = 128  # 图片分辨率
        self.gpu_num = self.get_gpu_num()

        self.parameter_samples()
        self.parameter_out_model()
        self.parameter_path()
        self.parameter_aei()

    def parameter_samples(self):
        # 数据增强
        self.aug_img_cut = False  # 是否按照短边裁剪成正方形
        self.aug_img_prob = 0.3  # 概率小于此值时，进行数据增强;否则只放缩到cnn_shape
        self.aug_img_add = 32  # 先放缩到cnn_shape+random_add，再裁剪到cnn_shape。add是random_add的最大值
        self.aug_img_flip_1 = 0.3  # 水平翻转的概率
        self.aug_img_flip_0 = 0  # 垂直翻转的概率
        self.aug_img_rotate = 10  # 旋转最大角度
        self.aug_img_contrast = 1.2  # 对比度最大值
        self.aug_img_brightness = 10  # 亮度增大最大值。255制。
        self.aug_img_noise = 0.001  # 图片中增加噪音点的最大百分比

    def parameter_out_model(self):
        self.epsilon = 1e-12
        self.epochs = 200
        self.start_epoch = 0
        self.restore_epoch = 0
        self.train_batch_size = 4
        self.max_batch_size = 64
        self.train_batch = max(2, self.max_batch_size // self.train_batch_size)  # 每更新一次梯度用几个batch
        self.rec_num = 0.65  # 样本集中相同图片的最少占比
        self.show_num = 0  # 50000  # 训练多少图片显示一次预测图片。0为不显示
        self.num_batch_to_print_losses = 0  # 每几个batch打印一次损失。0表示每代都打印
        self.train_buffer_capacity = int(self.max_batch_size * 10)
        self.num_d = 5  # 只训练判别器

        # 读取的模型版本
        self.arcface_save_path = '../arcface/models/dl16_CF_facerec_02-99'  # arcface
        self.unet_save_path = '../unet/models/dl16_CF_unet_01-37'  # unet

    def parameter_aei(self):
        # 优化器参数
        lr = np.linspace(0.00001, 0.0005, 50).tolist()
        self.aei_lr = lr + lr[::-1]
        # self.aei_beta1 = 0
        # self.aei_beta2 = 0.999
        # 变量初始化器
        self.aei_init_mean = 0
        self.aei_init_std = 0.02
        # 损失系数
        self.aei_lambda_wgan_gp = 10
        self.aei_lambda_att = 10
        self.aei_lambda_id = 5
        self.aei_lambda_rec = 10


        # 网络结构参数
        self.aei_id_size = 512
        self.aei_basic_filters = 16
        self.aei_max_fiilters = 256
        # self.aei_level_unet = 7
        # self.aei_unet_filters = [16, 32, 64, 128, 256, 256]  # 比aei_level_unet少1个
        self.aei_ADDGen_filters = (256, 256, 256, 256, 128, 64, 32, 3)  # 比aei_level_unet少1个
        self.aei_discriminater_filters = (16, 32, 64, 128)

        self.aei_keep_prob = 0.7

        self.aei_leaky_slop = 0.2

    def parameter_path(self):
        self.fig_data_dir = 'E:/TEST/AI/datasets/celebA/'

        self.path_txt = './aei_path_{datasets}.txt'

        self.save_path = f'./models/{self.name}_{self.version}'

        self.gen_dir = f'./gen_pics/'

    def check_all_dir_paths(self):
        self.check_dir(self.gen_dir)
        self.check_file(self.path_txt.format(cate='train'))
        self.check_file(self.path_txt.format(cate='test'))

    def check_dir(self, dir):
        if os.path.isdir(dir):
            print(f'Dir {dir} exists.')
        else:
            os.makedirs(dir)
            print(f'Dir {dir} not exists. Create {dir}')

    def check_file(self, file_path):
        if os.path.isfile(file_path):
            print(f'FILE {file_path} exists.')
        else:
            raise FileExistsError(f'ERROR: FILE {file_path} not exists.')

    def get_gpu_num(self):
        value = os.getenv('CUDA_VISIBLE_DEVICES', '0')
        return len(value.split(','))


aei_cfg = Config()
