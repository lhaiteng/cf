"""对gt框聚类得到先验框长宽"""
import numpy as np
import matplotlib.pyplot as plt


class Yolo_proposal:
    def __init__(self, boxes_path):
        self.boxes = np.load(boxes_path)
        self.num_boxes = self.boxes.shape[0]

    # 聚合聚类。类间距离定义为类中心的距离。
    def get_agg_proposal(self, update_method=np.median, stop_num_clusters=9):
        # 初始时每个样本是一个簇
        clses = np.arange(self.num_boxes)  # 每个样本所属的簇
        num_clusters = self.num_boxes  # 簇个数
        # 簇中心
        clusters = np.array([update_method(self.boxes[clses == _cls], axis=0)
                             for _cls in range(num_clusters)])

        # 族个数大于终止条件，循环
        while num_clusters > stop_num_clusters:
            print(f'\rnum_clusters = {num_clusters}', end='')
            # 簇间距
            ious, distances = self.get_distances(clusters, clusters, method='iou')
            # 注意，自身的distances存在0，会影响取最小值操作，令其为1
            distances[range(num_clusters), range(num_clusters)] = 1

            # 距离最近的两个簇，需要合并
            ind = np.argmin(distances)
            i, j = ind % num_clusters, ind // num_clusters
            i, j = np.minimum(i, j), np.maximum(i, j)
            # 合并后簇序号为i。更新样本所属簇序号
            clses[clses == j] = i
            for _i in range(j + 1, num_clusters):
                clses[clses == _i] = _i - 1

            num_clusters -= 1
            # 更新簇中心
            clusters = np.array([update_method(self.boxes[clses == _cls], axis=0)
                                 for _cls in range(num_clusters)])
            if num_clusters % 100 == 0:
                plt.plot(clusters[:, 0], clusters[:, 1], '.')
                plt.title(f'agglomeration_{num_clusters}')
                plt.axis([0, 1, 0, 1])
                plt.show()
        else:
            print()
        return clusters, clses

    # kmeans聚类
    def get_kmeans_proposal(self, num_clusters=9, update_method=np.median,
                            stop_step=1e5, stop_diff=1e-8):
        clses = np.zeros(self.num_boxes)  # 簇编号

        # 挑选初始簇中心
        clusters = self.get_start_kmeans_proposal(num_clusters, method='k++')
        print(f'start_clusters:\n{clusters}')

        n = 0
        while n < stop_step:
            print(f'\rn={n}\t', end='')
            # 计算距离
            ious, distances = self.get_distances(self.boxes, clusters, method='iou')
            # 更新簇
            new_cls = np.argmin(distances, axis=1)  # 新的簇编号
            # 是否中止更新
            diff = np.sum(clses != new_cls) / self.num_boxes  # 不相同的簇比率
            if diff < stop_diff:  # if (clses == new_cls).all():
                print(f'diff={diff:.6f} break.')
                break
            # 更新簇中心
            clses = new_cls
            clusters = np.array([update_method(self.boxes[clses == i], axis=0)
                                 for i in range(num_clusters)])
            n += 1
        else:
            print(f'diff={diff:.6f}')
        return clusters, clses

    def get_start_kmeans_proposal(self, num_clusters, method=''):
        clusters = np.zeros([num_clusters, 2])
        if method == 'k++':
            start_ind = np.random.choice(range(self.num_boxes), 1)
            clusters[0] = self.boxes[start_ind]
            for i in range(1, num_clusters):
                distances = self.get_distances(self.boxes, clusters[:i, :])
                choice_ind = np.argmax(np.sum(distances, axis=1))
                clusters[i] = self.boxes[choice_ind]
        else:
            start_ind = np.random.choice(range(self.num_boxes), num_clusters, replace=False)
            clusters = self.boxes[start_ind]
        return clusters

    def get_distances(self, boxes, clusters, method='iou'):
        assert method == 'iou'

        gt_w, gt_h = boxes[:, 0], boxes[:, 1]
        clusters_w, clusters_h = clusters[:, 0], clusters[:, 1]

        inter_w = np.minimum(gt_w[:, np.newaxis], clusters_w[np.newaxis, :])
        inter_h = np.minimum(gt_h[:, np.newaxis], clusters_h[np.newaxis, :])

        inter_areas = inter_w * inter_h
        gt_areas = gt_w * gt_h
        clusters_areas = clusters_w * clusters_h

        ious = inter_areas / (gt_areas[:, np.newaxis] + clusters_areas[np.newaxis, :] - inter_areas)

        distances = 1 - ious  # iou越大，距离越小
        return ious, distances

    # 可视化对比簇中心位置
    def plot(self, clusters, clses, title=''):
        boxes = self.boxes
        plt.scatter(boxes[:, 0], boxes[:, 1], s=2, c=clses)
        plt.scatter(clusters[:, 0], clusters[:, 1], s=20, c='k')
        if title: plt.title(title)
        plt.axis([0, 1, 0, 1])
        plt.show()


if __name__ == '__main__':
    num_boxes = 1000
    boxes_path = './record/test19_gt_wh.npy'
    data = np.random.normal(0.5, 0.2, 10000)
    data = data[data > 0]
    data = data[data <= 1]
    datawh = np.random.choice(data, 2 * num_boxes).reshape([-1, 2])
    np.save(boxes_path, datawh)

    yolo_proposal = Yolo_proposal(boxes_path)

    clusters, clses = yolo_proposal.get_kmeans_proposal(num_clusters=9)
    yolo_proposal.plot(clusters, clses, title='kmeans')
    clusters, clses = yolo_proposal.get_kmeans_proposal(num_clusters=9)
    yolo_proposal.plot(clusters, clses, title='kmeans')
    clusters, clses = yolo_proposal.get_kmeans_proposal(num_clusters=9)
    yolo_proposal.plot(clusters, clses, title='kmeans')

    clusters, clses = yolo_proposal.get_agg_proposal(stop_num_clusters=9)
    yolo_proposal.plot(clusters, clses, title='agglomeration')
    agg_clusters_path = './record/test19_agg_clusters.npy'
    agg_clses = './record/test19_agg_clses.npy'
    np.save(agg_clusters_path, clusters)
    np.save(agg_clses, clses)

    clusters = np.load(r'E:\TEST\ChangeFace\tests\record\test19_agg_clusters.npy')
    clses = np.load(r'E:\TEST\ChangeFace\tests\record\test19_agg_clses.npy')
    plt.plot(range(num_boxes), clses)
    plt.show()
    for i in range(9):
        print(f'{i}: {np.sum(clses == i)}')
