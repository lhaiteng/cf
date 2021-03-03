# -*- coding: utf-8 -*-
import os, random, json, cv2


# 得到文件夹dir所有子文件
def get_all_paths(dir):
    paths = []
    for f in os.listdir(dir):
        if os.path.isfile(dir + f):
            paths.append(dir + f)
        else:
            paths += get_all_paths(dir + f + '/')
    return paths


def make_cfp_path_label(data_dir, save_dir):
    labels = [p for p in os.listdir(data_dir) if os.path.isdir(data_dir + p)]
    path_label = []
    for label in labels:
        label_dir = data_dir + label + '/'
        for side in os.listdir(label_dir):
            if os.path.isdir(label_dir + side):
                side_dir = label_dir + side + '/'
                path_label += [f'{side_dir}{f}, {int(label) - 1}' for f in os.listdir(side_dir) if
                               os.path.isfile(side_dir + f)]
    path_label = [p + '\n' for p in path_label]
    random.shuffle(path_label)
    num = len(path_label)
    print(f'SAVE {num} path_label...')
    with open(save_dir + 'path_label.txt', 'w') as f:
        f.writelines(path_label)
    num_test = int(num * 0.1)
    num_train = num - num_test
    print(f'SAVE {num_train} train_path_label...')
    with open(save_dir + 'train_path_label.txt', 'w') as f:
        f.writelines(path_label[:num_train])
    print(f'SAVE {num_test} test_path_label...')
    with open(save_dir + 'test_path_label.txt', 'w') as f:
        f.writelines(path_label[num_train:])


def make_FDDB_path_txt(root_dir, ell_rec_ratio=1):
    anno_dir = root_dir + 'FDDB-folds/'
    pic_dir = root_dir + 'originalPics/'
    file_name = 'FDDB-fold-{ind}-ellipseList.txt'
    path_gts = []  # [(pic_path, gts[(r1, c1, r2, c2), ...]), (pic_path, gts[(r1, c1, r2, c2), ...]), ...]
    for ind in range(1, 11):
        pre = '0' if ind < 10 else ''
        file_path = anno_dir + file_name.format(ind=pre + str(ind))
        with open(file_path, 'r') as f:
            annos = f.readlines()
        annos = [a.strip() for a in annos]
        print(len(annos))
        j = 0
        while j < len(annos):
            pic_path = pic_dir + annos[j] + '.jpg'
            num_faces = int(annos[j + 1])
            gts = []
            for k in range(num_faces):
                gt = _get_rec_gt(annos[j + 2 + k], ell_rec_ratio)
                gts.append(gt)
            path_gts.append((pic_path, gts))
            j += 2 + num_faces

    random.shuffle(path_gts)

    num_pics = len(path_gts)
    print(f'NUM of all pics: {num_pics}')
    num_test = int(num_pics * 0.01)
    test_path_gts = path_gts[:num_test]
    train_path_gts = path_gts[num_test:]
    print(f'NUM of train pics: {len(train_path_gts)}')
    print(f'NUM of test pics: {len(test_path_gts)}')

    train_path_gts = {p[0]: p[1] for p in train_path_gts}
    train_str = json.dumps(train_path_gts, ensure_ascii=False)
    train_path_gts_txt = root_dir + 'train_path_gts.txt'
    with open(train_path_gts_txt, 'w') as f:
        f.writelines(train_str)
    print(f'SUCCESS write train_path_gts into {root_dir}.')

    test_path_gts = {p[0]: p[1] for p in test_path_gts}
    test_str = json.dumps(test_path_gts, ensure_ascii=False)
    test_path_gts_txt = root_dir + 'test_path_gts.txt'
    with open(test_path_gts_txt, 'w') as f:
        f.writelines(test_str)
    print(f'SUCCESS write test_path_gts into {root_dir}.')


def _get_rec_gt(annos, ell_rec_ratio=1):
    l = annos.split(' ')
    rc, rd, ag, cx, cy = [float(i) for i in l[:5]]
    # # 半长轴, 半短轴, 偏角, 中心点x， 中心点y
    r1, c1, r2, c2 = ell_to_rec(rc, rd, ag, cx, cy, ell_rec_ratio)
    return r1, c1, r2, c2


def ell_to_rec(rc, rd, ag, cx, cy, ell_rec_ratio=1):
    x1 = cx - ell_rec_ratio * rd
    y1 = cy - ell_rec_ratio * rc
    x2 = cx + ell_rec_ratio * rd
    y2 = cy + ell_rec_ratio * rc
    return y1, x1, y2, x2


def make_WIDER_path_txt(root_dir, scale=1):
    """
    文件夹准备：root_dir/{wider_face_split、WIDER_train、WIDER_val}/
    生成train和test(根据val生成，因为原版的test无标签)的路径文件，放在root_dir目录下
    生成格式: {path1:[gt1, gt2, ..], path2:[gt1, gt2, ...], ...}
    :param root_dir:
    :param scale:
    :return:
    """

    file = root_dir + 'wider_face_split/wider_face_train_bbx_gt.txt'
    pic_dir = root_dir + 'WIDER_train/images/'
    path_locs = _get_path_locs_from_file(file, pic_dir, scale)  # dict
    path_locs = json.dumps(path_locs)
    with open(root_dir + 'train_path_locs.txt', 'w') as f:
        f.writelines(path_locs)
    print(f'SUCCESS write train_path_locs into {root_dir}')

    file = root_dir + 'wider_face_split/wider_face_val_bbx_gt.txt'
    pic_dir = root_dir + 'WIDER_val/images/'
    path_locs = _get_path_locs_from_file(file, pic_dir, scale)  # dict
    path_locs = json.dumps(path_locs)
    with open(root_dir + 'test_path_locs.txt', 'w') as f:
        f.writelines(path_locs)
    print(f'SUCCESS write test_path_locs into {root_dir}')


def _get_path_locs_from_file(file_path, pic_dir, scale=1):
    with open(file_path, 'r') as f:
        files = f.readlines()
    files = [f.strip() for f in files]

    path_locs = {}
    j = 0
    while j < len(files):
        pic_path = pic_dir + files[j]
        num_faces = int(files[j + 1])
        if num_faces == 0:
            j += 3
            continue
        path_locs[pic_path] = []
        for k in range(num_faces):
            left, top, w, h = [int(i) for i in files[j + 2 + k].split()[:4]]
            dw, dh = w * scale - w, h * scale - h
            gt = top - dh / 2, left - dw / 2, top + h + dh / 2, left + w + dw / 2
            path_locs[pic_path].append(gt)
        j += 2 + num_faces

    return path_locs


def make_CycleGAN_path_loc_txt(txt, train, test):
    """
    写下的和txt一样，也是{path: locs, path: locs}。locs可能包含不止一个loc
    :param txtA:
    :param trainA:
    :param testA:
    :return:
    """
    print(f'START {txt}...')
    with open(txt, 'r') as f:
        path_locs = f.readlines()[0].strip()
    path_locs = json.loads(path_locs)  # {path: locs}
    # 先把path和locs分成(path, loc)对
    _path_locs = []  # [(path, loc), (path, loc), ...]
    for p in path_locs:
        for loc in path_locs[p]:
            _path_locs.append((p, loc))
    random.shuffle(_path_locs)
    num = len(_path_locs)
    num_test = int(num * 0.1)
    # 在把同path的loc合并成{path: locs, path:locs, ...}
    train_path_locs = _merge_path_dict(_path_locs[num_test:])
    test_path_locs = _merge_path_dict(_path_locs[:num_test])
    train_str = json.dumps(train_path_locs)
    test_str = json.dumps(test_path_locs)
    _save_json_str(train_str, train)
    _save_json_str(test_str, test)
    print(f'total num: {num}')
    print(f'train num: {len(train_path_locs)}')
    print(f'test num: {len(test_path_locs)}')


def _merge_path_dict(path_locs_list):
    path_locs = {}
    for path, loc in path_locs_list:
        if path in path_locs:
            path_locs[path].append(loc)
        else:
            path_locs[path] = [loc]

    return path_locs


def _save_json_str(string, txt_path):
    with open(txt_path, 'w') as f:
        f.writelines(string)


def make_CycleGan():
    txt = 'E:/TEST/AI/datasets/changeface_video/a_path_locs.txt'
    train = txt[:-4] + '_train.txt'
    test = txt[:-4] + '_test.txt'
    make_CycleGAN_path_loc_txt(txt, train, test)

    txt = 'E:/TEST/AI/datasets/changeface/pzy_path_locs.txt'
    train = txt[:-4] + '_train.txt'
    test = txt[:-4] + '_test.txt'
    make_CycleGAN_path_loc_txt(txt, train, test)

    txt = 'E:/TEST/AI/datasets/changeface/wzx_path_locs.txt'
    train = txt[:-4] + '_train.txt'
    test = txt[:-4] + '_test.txt'
    make_CycleGAN_path_loc_txt(txt, train, test)


def make_path_txt(all_dirs, pic_dirs, save_path, with_name='face'):
    train_paths = []
    test_paths = []
    for all_dir in all_dirs:
        pic_dirs += [all_dir + n + '/' for n in os.listdir(all_dir)
                     if with_name in n and os.path.isdir(all_dir + n)]
    for pic_dir in pic_dirs:
        if not os.path.isdir(pic_dir):
            print(f'WARNNING: not dir {pic_dir}')
            continue
        print(f'\rSTART getting paths from {pic_dir}...', end='')
        pic_paths = [pic_dir + p for p in os.listdir(pic_dir) if os.path.isfile(pic_dir + p)]
        for _ in range(3):
            random.shuffle(pic_paths)
        total_num = len(pic_paths)
        test_num = int(total_num * 0.1)
        test_paths += pic_paths[:test_num]
        train_paths += pic_paths[test_num:]
    print(f'\nSTART writing into {save_path.format(cate="train")}...')
    _celebA_save(train_paths, save_path.format(cate='train'))
    print(f'START writing into {save_path.format(cate="test")}...')
    _celebA_save(test_paths, save_path.format(cate='test'))


def _celebA_save(pic_paths, save_path):
    pic_paths = [p + '\n' for p in pic_paths]
    with open(save_path, 'w') as f:
        f.writelines(pic_paths)


# 得到人脸识别的标签txt
def make_label_txt_arcface(root_dir, dataset_names, save_paths, **kwargs):
    """
    得到人脸识别的标签txt
    :param root_dir: 存放数据的根文件夹 'E:/TEST/AI/datasets/'
    :param dataset_names: 数据集名称列表 [cfp-dataset, cnface_face, ...]
    :param save_paths: '../arcface/arcface_path_label_{datasets}.txt'
    :param kwargs:
        min_test_num: 最小测试样本比例，默认0.1
        max_num_per_cls: 每个类别最大样本数，默认0表示取全部。
    :return:
    """
    pic_types = ('png', 'jpg')  # 读取文件的类型
    print(f'START get labels from {dataset_names}.')
    train_labels, test_labels, total_labels = [], [], []
    min_pics_cls, min_pics_cls_path = float('inf'), ''
    # 获取类别文件夹
    cls_dirs = []  # 所有的类别文件夹
    for _dname in dataset_names:
        if 'cfp' in _dname:  # 若是cfp-dataset
            _set_dir = root_dir + _dname + '/Data/Images/'
            _cls_dirs = [_set_dir + _clsname + '/' for _clsname in os.listdir(_set_dir)
                         if os.path.isdir(_set_dir + _clsname)]
        else:  # 其他数据集
            _set_dir = root_dir + _dname + '/'
            _cls_dirs = [_set_dir + _clsname + '/' for _clsname in os.listdir(_set_dir)
                         if os.path.isdir(_set_dir + _clsname)]
        cls_dirs += _cls_dirs
        print(f'There are {len(_cls_dirs)} labels in {_dname}')
    num_cls = len(cls_dirs)
    print(f'There are {num_cls} labels in total.')
    print('-' * 100)
    print(f'all cls dirs:\n{cls_dirs}')
    print('-' * 100)
    for _ in range(3):
        random.shuffle(cls_dirs)
    print('START loading pics...')
    max_num_per_cls = kwargs.get('max_num_per_cls', 0)
    for label_ind, _cls_dir in enumerate(cls_dirs):
        print(f'\rstart loading pics from {_cls_dir}...', end='')
        path_label = [f'{p}, {label_ind}\n' for p in get_all_paths(_cls_dir) if p[-3:] in pic_types]
        random.shuffle(path_label)
        if max_num_per_cls: path_label = path_label[:max_num_per_cls]
        test_labels += path_label[:2]  # 保证每个标签至少有两个测试样本
        train_labels += path_label[2:6]  # 保证每个标签至少有四个训练样本
        total_labels += path_label[6:]
        if min_pics_cls > len(path_label): min_pics_cls, min_pics_cls_path = len(path_label), _cls_dir
    print(f'\n标签具有的具有最少样本数：{min_pics_cls}\n路径：{min_pics_cls_path}')
    print('-' * 100)

    for _ in range(3):
        random.shuffle(total_labels)

    # 挑选至满足最低测试样本数量要求
    min_test_num = kwargs.get('min_test_num', 0.2)
    total_num = len(total_labels)
    test_num = max(int(min_test_num * total_num) - len(test_labels), 1)
    test_labels += total_labels[:test_num + 1]
    train_labels += total_labels[test_num + 1:]

    # 最终标签数量
    train_num, test_num = len(train_labels), len(test_labels)
    total_num = train_num + test_num

    print('CONCLUSION:')
    coclusion = {'num_cls': num_cls, 'max_num_per_cls': max_num_per_cls, 'min_pics_cls': min_pics_cls,
                 'total_num': total_num, 'train_num': train_num, 'test_num': test_num}
    print(f'num cls: {num_cls}\tmax num per cls: {max_num_per_cls}\tmin pics cls: {min_pics_cls}')
    print(f'total num: {total_num}\ttrain num: {train_num}\ttest num: {test_num}')
    print('-' * 100)
    print(f'START writing total total_labels...')
    with open(save_paths.format(datasets='total'), 'w') as f:
        f.writelines(train_labels + test_labels)
    print(f'START writing train total_labels...')
    with open(save_paths.format(datasets='train'), 'w') as f:
        f.writelines(train_labels)
    print(f'START writing test total_labels...')
    with open(save_paths.format(datasets='test'), 'w') as f:
        f.writelines(test_labels)
    print(f'FINISH writing all total_labels.')
    print('-' * 100)


if __name__ == '__main__':
    """面部识别"""
    # cfp
    # data_dir = 'E:/TEST/AI/datasets/cfp/Images/'
    # save_dir = 'E:/TEST/AI/datasets/cfp/Images/'
    # make_cfp_path_label(data_dir, save_dir)

    """做面部检测的数据集标签"""
    # # FDDB
    # root_dir = 'E:/TEST/AI/datasets/FDDB/'
    # FDDB_ell_rec_ratio = 1.1  # 把样本椭圆标记变成方框的缩放因子
    # make_FDDB_path_txt(root_dir, FDDB_ell_rec_ratio)
    # # WIDER
    # root_dir = 'E:/TEST/AI/datasets/WIDER/'
    # scale = 1  # 标注框的缩放因子
    # make_WIDER_path_txt(root_dir, scale)

    """做cyclegan的面部。是从path_locs文件中摘出的path_loc"""
    # make_CycleGan()

    """
    仅提取文件的路径，不施加标签
    适用于：UNet、AEI
    """
    # # all_dirs = ['E:/TEST/AI/datasets/jpface/',
    # #             'E:/TEST/AI/datasets/cnface/']
    # # pic_dirs = []
    # all_dirs = ['E:/TEST/AI/datasets/jpface/']
    # cns = ('puzhiyan', 'dilireba', 'maidina', 'liqin', 'yangmi', 'liushishi', 'guanxiaotong',
    #        'zhaoliying', 'liutao', 'tongliya', 'gulinazha', 'jiangxin', 'qiwei', 'jingtian',
    #        'wangzuxian', 'chenyanxi', 'zhuyin', 'dongqing', 'qiushuzhen', 'tianfuzhen', 'chengxiao',
    #        'chenhao', 'zhaoyihuan', 'zhanghanyun', 'guobiting')
    # pic_dirs = [f'E:/TEST/AI/datasets/cnface/face_{n}/' for n in cns]
    # save_path = '../aei/aei_path_{datasets}.txt'
    # make_path_txt(all_dirs, pic_dirs, save_path, with_name='face')

    """
    根据提取文件夹下所有文件夹的图片，并赋标签
    适用于arcface
    """
    root_dir = 'E:/TEST/AI/datasets/'
    dataset_names = []
    # dataset_name.extend([f'{n}face_face' for n in ('cn', 'jp', 'kr')])
    dataset_names.append('cfp-dataset')
    save_paths = '../arcface/arcface_path_label_{datasets}.txt'
    kw = {'max_num_per_cls': 30,  # 每个类别的最大样本数
          }
    make_label_txt_arcface(root_dir, dataset_names, save_paths, **kw)  # 文件夹名含有face
