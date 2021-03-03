# -*- coding: utf-8 -*-
import numpy as np
import os


class Config:
    def __init__(self):
        self.name = 'dl16_CF_unet'
        self.version = '01'

        self.cnn_shape = 128

        self.gpu_num = self.get_gpu_num()

        self.parameter_samples()
        self.parameter_out_model()
        self.parameter_path()
        self.parameters()

    def parameter_samples(self):
        # 数据增强
        self.aug_img_cut = False  # 是否按照短边裁剪成正方形
        self.aug_img_prob = 0.7  # 概率小于此值时，进行数据增强;否则只放缩到cnn_shape
        self.aug_img_add = 32  # 先放缩到cnn_shape+random_add，再裁剪到cnn_shape。add是random_add的最大值
        self.aug_img_flip_1 = 0.5  # 水平翻转的概率
        self.aug_img_flip_0 = 0.1  # 垂直翻转的概率
        self.aug_img_rotate = 15  # 旋转最大角度
        self.aug_img_contrast = 1.2  # 对比度最大值
        self.aug_img_brightness = 20  # 亮度增大最大值。255制。
        self.aug_img_noise = 0.0005  # 图片中增加噪音点的最大百分比

    def parameter_out_model(self):
        self.epsilon = 1e-12
        self.epochs = 100
        self.train_batch_size = 128
        self.max_batch_size = 256
        self.train_batch = max(2, self.max_batch_size // self.train_batch_size)  # 每更新一次梯度用几个batch
        self.show_num = 100000  # 50000  # 训练多少图片显示一次预测图片。0为不显示
        self.num_batch_to_print_losses = 0  # 每几个batch打印一次损失。0表示每代都打印
        self.train_buffer_capacity = int(self.max_batch_size * 10)

    def parameters(self):
        # 优化器参数
        lr = np.linspace(1e-6, 0.0005, 50).tolist()
        self.lr = lr + lr[::-1]
        # self.beta1 = 0
        # self.beta2 = 0.999

        # 变量初始化器
        self.init_mean = 0
        self.init_std = 0.02

        self.keep_prob = 0.7
        self.leaky_slop = 0.2

        self.start_epoch = 0
        self.unet_filters = [16, 32, 64, 128, 256, 512]

    def parameter_path(self):
        self.path_txt = './unet_path_{datasets}.txt'

        self.save_path = f'./models/{self.name}_{self.version}'

        self.gen_dir = f'./gen_pics/'

    def check_all_dir_paths(self):
        self.check_dir(self.gen_dir)
        self.check_file(self.path_txt.format(cate='train'))
        self.check_file(self.path_txt.format(cate='test'))

    def check_dir(self, dir):
        if os.path.isdir(dir):
            print(f'DIR {dir} exists.')
        else:
            os.makedirs(dir)
            print(f'DIR {dir} not exists. Create {dir}')

    def check_file(self, file_path):
        if os.path.isfile(file_path):
            print(f'FILE {file_path} exists.')
        else:
            raise FileExistsError(f'ERROR: FILE {file_path} not exists.')

    def get_gpu_num(self):
        value = os.getenv('CUDA_VISIBLE_DEVICES', '0')
        return len(value.split(','))


unet_cfg = Config()
