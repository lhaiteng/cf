# -*- coding: utf-8 -*-
import os, cv2


class SSDConfig:
    def __init__(self):
        self.name = 'dl16_CF_ssd'
        self.version = 'v01'

        # 样本类别数。还需要再加上背景分类。
        # self.num_classes = 20
        # 仅识别人脸，没有分类一说。

        # 卷积参数
        self.add_conv_filters = 256
        self.num_add_conv = 4  # conv7之后多了多少层  最小样本是71，不改变原图情况下只能再扩展2层
        # 卷积池化的参数p, k, s
        self.conv_layers = [(1, 3, 1), (1, 3, 1), (0, 2, 2),
                            (1, 3, 1), (1, 3, 1), (0, 2, 2),
                            (1, 3, 1), (1, 3, 1), (1, 3, 1), (0, 2, 2),
                            (1, 3, 1), (1, 3, 1), (1, 3, 1), (0, 2, 2),
                            (1, 3, 1), (1, 3, 1), (1, 3, 1)]
        # self.fm_inds = []  # 提取特征图的层号
        self.fm_inds = [12]  # conv4_3
        self.conv_layers += [(1, 3, 1), (0, 1, 1)]  # conv7
        self.fm_inds += [18]  # 提取特征图的层号 conv7
        for _ in range(self.num_add_conv):
            self.conv_layers += [(0, 1, 1), (1, 3, 1), (0, 2, 2)]
            self.fm_inds += [self.fm_inds[-1] + 3]  # 提取特征图的层号
        # 提取的特征图数
        self.num_fms = len(self.fm_inds)

        # 锚框参数
        self.max_img_size = 600  # 长边的最大尺寸
        self.min_img_size = 301  # 短边的最小尺寸
        # 锚框标准面积的单边尺寸
        self.anchor_base_scale = 400
        self.anchor_min = 0.2
        self.anchor_max = 0.9
        # 锚框的高宽比
        self.anchor_ratio = (1, 2, 1 / 2)
        # 每个特征图对应的基础尺寸
        sks = [self.anchor_min / 2]  # conv4_3
        m = self.num_fms - 1  # conv4_3之后的特征图个数
        delta_scale = (self.anchor_max - self.anchor_min) / (m - 1)
        for i in range(m + 1):
            sk = self.anchor_min + delta_scale * i
            sks.append(sk)
        self.anchor_scales = [int(sk * self.anchor_base_scale) for sk in sks]
        # 单点对应的锚框数目
        self.num_anchor_box = len(self.anchor_ratio) + 1

        # 画框参数
        self.box_linewidth = 2
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.fontScale = 0.5
        self.font_width = 1

        # 图片数据增强
        self.aug_img_cut = False  # 是否按照短边裁剪成正方形
        self.aug_img_prob = 0.4  # 概率小于此值时，不进行数据增强
        self.aug_img_add = 0  # 先放缩到224+random_add，再裁剪到224。add是random_add的最大值
        self.aug_img_contrast = 1.2  # 对比度最大值
        self.aug_img_brightness = 10  # 亮度增大最大值。255制。
        self.aug_img_flip = 0  # 随机翻转的概率
        self.aug_img_rotate = 0  # 旋转最大角度
        self.aug_img_noise = 0.001  # 图片中增加噪音点的最大百分比

        # 数据集总文件夹
        # 训练和测试的path_gts文件路径
        self.train_path_gts_path = [# 'E:/TEST/AI/datasets/FDDB/train_path_gts.txt',
                                    'E:/TEST/AI/datasets/WIDER/train_path_locs.txt']
        self.test_path_gts_path = [# 'E:/TEST/AI/datasets/FDDB/test_path_gts.txt',
                                   'E:/TEST/AI/datasets/WIDER/test_path_locs.txt']
        # 迁移学习vgg16的npy保存路径
        self.vgg16_npy_path = 'E:/TEST/AI/dl/05_transfer/tensorflow_vgg/vgg16.npy'
        self.add_dir()
        self.check_dir()

        """训练参数"""

        self.train_start_num_epoch = 0  # 训练开始代数。已生成图片下标+1
        self.train_epochs = 100
        self.train_batch_size = 1
        self.train_num_batch_to_print_losses = 0  # 每几个batch打印一次损失
        self.train_show_num_rate = 500  # 途中显示预测结果的频率
        self.train_buffer_capacity = int(self.train_batch_size * 20)  # 并行存储的最大个数
        self.train_gpu_num = self.get_gpu_num()  # 使用的gpu个数
        self.train_lr = 0.0002
        # 变量初始化参数
        self.train_var_mean = 0
        self.train_var_std = 0.01
        # 回归损失系数
        self.train_lambda_reg = 10  # 回归损失系数

        # ssd_target_layer参数
        # 在ssd_target_layer中使用的正负样本阈值
        self.train_ssd_positive_iou = 0.5
        self.train_ssd_negative_iou_high = 0.45
        self.train_ssd_negative_iou_lo = 0.1
        self.train_negative_big_ratio = 0.8
        # 训练时，每张图片使用的正负样本总数
        self.train_ssd_batch_size = 128
        self.train_ssd_fg_fraction = 0.5

        """测试参数"""

        # get_result参数
        # 预测框的最小尺寸
        self.test_min_size = 16  # 如果更改，则make_WIDER_path_txt也要更新。
        # 保留得分高于阈值的结果
        self.test_result_score_thres = 0.9
        # 根据得到的结果进行NMS的阈值
        self.test_result_nms_iou = 0.3

    def get_gpu_num(self):
        value = os.getenv('CUDA_VISIBLE_DEVICES', '0')
        return len(value.split(','))

    def add_dir(self):
        # model保存路径
        self.save_path = f'./models/{self.name}_{self.version}'
        # log保存路径
        self.log_dir = f'./logs/{self.name}_{self.version}/'
        # 生成文件保存路径
        self.gen_dir = f'./gen_pics/{self.name}_{self.version}/'

    def check_dir(self):
        for dir in (self.log_dir, self.gen_dir):
            self._check_dir(dir)

    def _check_dir(self, dir):
        if os.path.isdir(dir):
            print(f'Dir {dir} exists.')
        else:
            os.makedirs(dir)
            print(f'Dir {dir} not exists. Create {dir}')


ssd_cfg = SSDConfig()
