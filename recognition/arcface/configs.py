# -*- coding: utf-8 -*-
from easydict import EasyDict as edict

"""通用配置"""
config = edict()

config.img_size = 128
config.regularizer = None  # 'l2_regularizer'
config.regular_factor = 0  # 5e-4
config.embedding_size = 128
config.act_type = 'leaky_relu'
config.focal_loss_alpha = 0  # 0.75
config.focal_loss_gamma = 0  # 2

# 初始化器
initializer = edict()

initializer.xavier = edict()
initializer.xavier.initializer = 'xavier'

initializer.truncated_normal = edict()
initializer.truncated_normal.initializer = 'truncated_normal'
initializer.truncated_normal.init_mean = 0
initializer.truncated_normal.init_std = 0.02

"""优化器配置"""
optimizer = edict()

optimizer.adam = edict()
optimizer.adam.optimizer_name = 'AdamOptimizer'

optimizer.momentum = edict()
optimizer.momentum.optimizer_name = 'MomentumOptimizer'
optimizer.momentum.momentum = 0.9

"""网络配置"""
network = edict()

network.resnet_18 = edict()
network.resnet_18.net_name = 'resnet'
network.resnet_18.num_layers = 18

network.resnet_50 = edict()
network.resnet_50.net_name = 'resnet'
network.resnet_50.num_layers = 50

network.resnet_100 = edict()
network.resnet_100.net_name = 'resnet'
network.resnet_100.num_layers = 100

network.darknet_53 = edict()
network.darknet_53.net_name = 'darknet'
network.darknet_53.num_layers = 53
network.darknet_53.base_filter = 32  # 32
network.darknet_53.addition_layer_type = 3

network.darknet_27 = edict()
network.darknet_27.net_name = 'darknet'
network.darknet_27.num_layers = 27
network.darknet_27.base_filter = 16  # 32
network.darknet_27.addition_layer_type = 3





"""数据集配置"""
dataset = edict()

dataset.cfp = edict()
dataset.cfp.dataset_name = 'cfp'
dataset.cfp.num_cls = 500  # 284 + 500
dataset.cfp.num_train_pics = 5500
dataset.cfp.num_test_pics = 1500

dataset.faces = edict()
dataset.faces.dataset_name = 'faces'
dataset.faces.num_cls = 284  # 284 + 498
dataset.faces.num_train_pics = 13854
dataset.faces.num_test_pics = 4531

"""损失配置"""
loss = edict()

loss.arcface = edict()
loss.arcface.loss_name = 'arcface_loss'
loss.arcface.loss_s = 64
loss.arcface.loss_m1 = 1
loss.arcface.loss_m2 = 0.5
loss.arcface.loss_m3 = 0.

loss.combind0 = edict()
loss.combind0.loss_name = 'margin_loss'
loss.combind0.loss_s = 64
loss.combind0.loss_m1 = 1
loss.combind0.loss_m2 = 0.3
loss.combind0.loss_m3 = 0.2
loss.combind0.arcface_loss_type = 1

loss.combind1 = edict()
loss.combind1.loss_name = 'margin_loss'
loss.combind1.loss_s = 18
loss.combind1.loss_m1 = 0.8
loss.combind1.loss_m2 = 0.1
loss.combind1.loss_m3 = 0.2

loss.combind2 = edict()
loss.combind2.loss_name = 'margin_loss'
loss.combind2.loss_s = 64
loss.combind2.loss_m1 = 1
loss.combind2.loss_m2 = 0.15
loss.combind2.loss_m3 = 0.1
loss.combind2.arcface_loss_type = 1



"""默认配置"""
default = edict()

# 默认网络、数据集、损失
default.network = 'darknet_27'  # 'darknet_53', 'resnet_100'
default.dataset = 'cfp'  # ['cfp', 'faces']
default.loss = 'combind2'
default.initializer = 'xavier'
default.optimizer = 'momentum'
# 版本
default.version = ['27', str, 'train version']
default.restore_version = ['27-99', str, 'restore version']
default.batch_size = [64, int, 'num of batch size']
default.num_epoch = [100, int, 'num of epoch']
# 学习率
default.max_lr = [1e-3, float, 'max learning rate']
default.end_epoch = [80, int, 'end epoch of learning rate decay']
default.lr_decay = ['cosine', str, 'lr decay method']
# 保留率
default.keep_prob = [0.5, float, 'dropout keep prob']
# 储存文件夹
default.model_dir = ['../record/mdoels/', str, 'output model dir']
default.log_dir = [f'../record/logs/v{default.version[0]}/', str, 'output log dir']
default.gen_pic_dir = [f'../record/gen_pics/v{default.version[0]}/', str, 'output pic dir']


# 把选择的网络、数据集、损失载入配置
def update_config(_network=default.network, _dataset=default.dataset,
                  _loss=default.loss, _initializer=default.initializer,
                  _optimizer=default.optimizer):
    # 初始化器
    config.initializer = _initializer
    for k, v in initializer[_initializer].items():
        config[k] = v
        if k in default:
            default[k] = v
    # 优化器
    config.optimizer = _optimizer
    for k, v in optimizer[_optimizer].items():
        config[k] = v
        if k in default:
            default[k] = v


    config.network = _network
    for k, v in network[_network].items():
        config[k] = v
        if k in default:
            default[k] = v
    config.dataset = _dataset
    for k, v in dataset[_dataset].items():
        config[k] = v
        if k in default:
            default[k] = v
    config.loss = _loss
    for k, v in loss[_loss].items():
        config[k] = v
        if k in default:
            default[k] = v


if __name__ == '__main__':
    update_config(default.network, default.dataset, default.loss,
                  default.initializer, default.optimizer)
    print(config)
