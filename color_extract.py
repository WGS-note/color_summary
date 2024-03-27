# coding:utf-8
# @author: wangguisen
# @Time: 2023/11/16 16:41
# @File: color_extract.py
import numpy as np
import time
import matplotlib.pyplot as plt
import cv2
from scipy.spatial.distance import cdist
from functools import partial, lru_cache
from dataclasses import dataclass
import pandas as pd
from typing import List, Dict

'''   配色方案   '''
WEBCOLOR_NAME = pd.read_csv('./assets/colors.csv', usecols=[0, 1, 2, 3])  # id | 颜色名称en | 颜色名称zh | HEX
WEBCOLORS = cv2.imread('./assets/webcolors.png')

'''   获取每个像素点的近距离质心索引   '''
def get_color_distance(colors, target, color_convert=partial(cv2.cvtColor, code=cv2.COLOR_BGR2HSV), blur_block=1):
    '''
    获取每个像素点的近距离质心索引
    Args:
        colors: 输入图像 BRG [H, W, C]
        target: 140种配色 [140, 3]
        color_convert: 色系转换方式
        blur_block: 模糊滑块大小

    Returns: 图像像素点与配色的距离 [H//blur_block * W//blur_block, 140]
    '''
    colors = np.lib.stride_tricks.sliding_window_view(colors, window_shape=(blur_block, blur_block), axis=(0, 1))[::blur_block, ::blur_block]
    colors = np.mean(colors, axis=(-2, -1)).astype(np.uint8)  # [H//blur_block, W//blur_block, 3]

    # colors = cv2.GaussianBlur(colors, (blur_block, blur_block), 0, 0).astype(np.uint8)
    # colors = cv2.blur(colors, (blur_block, blur_block)).astype(np.uint8)

    colors = color_convert(colors)
    target = color_convert(target)

    d_color = target.shape[-1]
    distance = cdist(colors.reshape(-1, d_color), target.reshape(-1, d_color), metric='euclidean')  # [H//blur_block * W//blur_block, 140]

    return distance


'''   取距离近的80%的像素点   '''
def color_idx_filtration(distance, closest_centroids, factor=0.8):
    pixs = distance.shape[0]
    distance_sort = np.array([distance[i, closest_centroids[i]] for i in range(pixs)])

    cut = int(pixs * factor)
    distance_sort, closest_centroids = sort_multiple_pairs(distance_sort, closest_centroids)
    closest_centroids = closest_centroids[:cut]

    print('原像素点数量: %d, 取距离近的%d%%后: %d' % (pixs, factor * 100, len(closest_centroids)))

    return np.array(closest_centroids)


'''   得到图像embedding   '''
def get_color_emb(distance, centroids_len=140, ignore_mode=None, ignore_factor=0.05):
    '''
    得到图像embedding，每个emb value代表质心配色出现的占比。
    Args:
        distance: 像素距离数组 [H//blur_block * W//blur_block, 140]
        centroids_len: 140种配色RGB(webcolors)
        ignore_mode: 在配色中，忽略占比不足的方式，默认不忽略
        ignore_factor: 忽略比例

    Returns: 140种配色在目标图像中出现的占比  [140, ]
    '''
    centroid_counts = np.zeros(centroids_len, dtype=int)
    closest_centroids = np.argmin(distance, axis=1)  # [H//blur_block * W//blur_block, ]

    for centroid_index in closest_centroids:
        centroid_counts[centroid_index] += 1

    centroid_proportion = centroid_counts / np.sum(centroid_counts)

    if not ignore_mode:
        return centroid_proportion
    elif ignore_mode == 'ignore':
        centroid_proportion = ignore_proportion(centroid_proportion=centroid_proportion, ignore_factor=ignore_factor)
    elif ignore_mode == 'grand_total_ignore':
        centroid_proportion = total_ignore_proportion(centroid_proportion=centroid_proportion, centroids_len=centroids_len, ignore_factor=ignore_factor)
    else:
        raise ValueError('the parameter is not supported.')

    return centroid_proportion


'''   占比不足的忽略   '''
def ignore_proportion(centroid_proportion, ignore_factor=0.05):
    ignore_index = np.where(centroid_proportion < ignore_factor)[0]
    centroid_proportion[ignore_index] = 0.

    return centroid_proportion


'''   累计占比不足的忽略   '''
def total_ignore_proportion(centroid_proportion, centroids_len, ignore_factor=0.05):

    index = range(centroids_len)   # 140
    centroid_proportio_sort, index = sort_multiple_pairs(centroid_proportion, index)

    ignore_index = []
    total = 0.
    for i in range(centroids_len):
        if total >= ignore_factor:
            break
        total += centroid_proportio_sort[i]
        ignore_index.append(index[i])

    centroid_proportion[ignore_index] = 0.

    return centroid_proportion


def color_img_show(centroid_proportion, colors, labels_str=None, title=None, index=None, save_path='./assets/color_pie.jpg',
                   cur_pro=None, cur_cols=None, res_min_idx=None):

    colors = colors.reshape(-1, 3)

    histogram, labels, cen_colors = [], [], []
    for i in range(len(centroid_proportion)):
        if centroid_proportion[i] != 0.:
            histogram.append(centroid_proportion[i])
            if labels_str is not None:
                labels.append('cu: {}'.format(labels_str[i]))
            else:
                labels.append(str(i+1))
            # cen_colors.append(colors[i]/255.)
            # cen_colors.append(colors[i])
            cen_colors.append(rgb_to_hex(colors[i]))

    histogram, labels, cen_colors = sort_multiple_pairs(histogram, labels, cen_colors, reverse=True)

    plt.figure(figsize=(10, 10))
    # fig, axs = plt.subplots(1, 2, figsize=(6, 6))

    plt.subplot(211)
    if index is not None:
        plt.pie(histogram, colors=cen_colors[index, :], autopct='%1.1f%%')
    else:
        plt.pie(histogram, colors=cen_colors, labels=labels, autopct='%1.1f%%')

    # histogram = np.array(histogram) * 100
    # plt.bar(labels, histogram, color=cen_colors)
    # # plt.barh(labels, np.array(histogram) * 100, color=cen_colors)
    # for a, b in zip(labels, histogram):
    #     plt.text(a, b, '{:.2f}%'.format(b), ha='center', va='bottom', fontsize=7)

    # 加个柱形图：当前合并的是那两个，合并后颜色
    if cur_pro:
        tmp_pl = [rgb_to_hex(i) for i in cur_cols]
        plt.subplot(212)
        plt.bar(range(3), np.array(cur_pro) * 100, color=tmp_pl)


    plt.title(title)
    plt.axis('equal')
    plt.savefig(save_path)
    plt.show()

'''   RGB -> HEX   '''
def rgb_to_hex(rgb):
    r, g, b = rgb
    r = str(hex(int(max(0, min(255, r)))))[-2:].replace('x', '0')
    g = str(hex(int(max(0, min(255, g)))))[-2:].replace('x', '0')
    b = str(hex(int(max(0, min(255, b)))))[-2:].replace('x', '0')

    hex_string = '#{}{}{}'.format(r, g, b)
    return hex_string


'''   BGR -> HSVcone   '''
def bgr2hsvcone(img, r=2):

    arr_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h = arr_hsv[..., 0] / 180. * 2
    s = arr_hsv[..., 1] / 255.
    v = arr_hsv[..., 2] / 255.

    x = r * np.cos(h * np.pi) * s * v
    y = r * np.sin(h * np.pi) * s * v

    return np.stack((x, y, v), axis=-1)


def sort_multiple_pairs(*args, reverse=False):
    pairs = sorted(zip(*args), reverse=reverse)
    arrs = [np.array(item) for item in zip(*pairs)]

    return arrs


@dataclass
class Centroids:
    xids: np.ndarray

    def __hash__(self):
        x = self.xids
        return hash(x.data.tobytes())

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __add__(self, other):
        new_xids = np.concatenate([self.xids, other.xids], axis=0)
        return Centroids(new_xids)

class ColorSummary():
    def __init__(self, centroid_proportion, target_img, num=5):
        # 忽略占比为0的
        # centroid_proportion = np.array(centroid_proportion)
        index_boole = centroid_proportion[:] > 0
        self.weights = centroid_proportion[index_boole]
        self.colors = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB).squeeze(axis=1)[index_boole, :]
        self.cones = bgr2hsvcone(target_img, r=10).squeeze(axis=1)[index_boole, :]
        self.C = len(self.weights)
        self.num = num  # 概述个数
        self.loss_history = []
        # print('忽略占比为0后个数: ', len(self.weights))

        self.centroid_lst = [Centroids(xids=np.array([i,])) for i in range(self.C)]

    def merge_similar_colors(self, img_show=False):
        while self.C > self.num:

            min_delta = np.inf
            min_loss = np.inf
            best_pair = (0, 0)

            for j in range(self.C):
                for i in range(j + 1, self.C):
                    # 计算△误差
                    tmp_centroid = self.centroid_lst[j] + self.centroid_lst[i]
                    e_loss = self.cal_e_loss(tmp_centroid)
                    j_loss = self.cal_e_loss(self.centroid_lst[j])
                    i_loss = self.cal_e_loss(self.centroid_lst[i])
                    delta_loss = e_loss - j_loss - i_loss

                    if delta_loss < min_delta:
                        min_loss = e_loss
                        min_delta = delta_loss
                        best_pair = (j, i)

            self.C -= 1
            c2 = self.centroid_lst.pop(best_pair[1])
            c1 = self.centroid_lst.pop(best_pair[0])
            new_centroid = c1 + c2
            self.centroid_lst.append(new_centroid)

            if img_show:
                loss = self.cal_loss(self.C, 0.1, min_loss)

                print('要合并的索引是：{}, 它的△误差最小：{}'.format(best_pair, min_delta))
                title = 'loss: %.6f, C: %d, e_loss: %.6f, e_loss change: %.6f' % (loss, self.C, min_loss, min_delta)
                print(title)

                self.loss_history.append(loss)
                color_img_show(centroid_proportion=self.get_weigths(self.centroid_lst),
                               colors=self.get_colors(self.centroid_lst),
                               save_path='./assets_pie/clu_color_pie{}.jpg'.format(self.C),
                               title=title,
                               cur_pro=[self.get_weight(c2), self.get_weight(c1), self.get_weight(new_centroid)],
                               # todo检查用，将当前合并的两个画出来
                               cur_cols=[self.get_color(c2), self.get_color(c1), self.get_color(new_centroid)],
                               res_min_idx=best_pair,
                               )

        if img_show:
            plt.plot(self.loss_history)
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.title("Loss Curve")
            plt.savefig('./assets_pie/loss.jpg')
            plt.show()

    @lru_cache
    def cal_new_centroid(self, centroid):
        c_cone = np.sum(self.weights[centroid.xids].reshape(-1, 1) * self.cones[centroid.xids], axis=0) / np.sum(self.weights[centroid.xids])
        return c_cone

    @lru_cache(maxsize=100*100)
    def cal_e_loss(self, centroid):
        e_loss = np.sum(self.weights[centroid.xids].reshape(-1, 1) * np.sqrt(np.square(self.cones[centroid.xids, :] - self.cal_new_centroid(centroid))))
        return e_loss

    @lru_cache(maxsize=100*100)
    def cal_loss(self, C, alpha=0.1, e_loss=0.):
        return alpha * C + e_loss

    @lru_cache
    def get_weight(self, centroid):
        return np.sum(self.weights[centroid.xids])

    def get_weigths(self, centroid_lst):
        weights = np.array([self.get_weight(centroid) for centroid in centroid_lst])
        return weights

    @lru_cache
    def get_color(self, centroid):
        return np.sum(self.weights[centroid.xids].reshape(-1, 1) * self.colors[centroid.xids], axis=0) / self.get_weight(centroid)

    def get_colors(self, centroid_lst):
        colors = np.array([self.get_color(centroid) for centroid in centroid_lst])
        return colors

    def min_distance(self, centroid):
        cen = self.cal_new_centroid(centroid).reshape(-1, 3)
        distance = cdist(self.cones[centroid.xids], cen)
        near_xid = centroid.xids[np.argmin(distance, axis=0)]
        return near_xid

    def get_color_summary(self, webcolor_name: pd.DataFrame):
        '''
        Returns: [{'prob': .4f, 'name': {'en': , 'zh_cn': }, 'RGB':}, {}]
        '''
        color_summary_lst = []
        sort_prob = []

        for cen in self.centroid_lst:
            color_summary = {}
            color_summary["proportion"] = self.get_weight(cen)
            color_summary["RGB"] = self.get_color(cen).tolist()
            near_xid = self.min_distance(cen)
            idx, en, zh_cn, hex = webcolor_name[webcolor_name['HEX'] == rgb_to_hex(self.colors[near_xid].squeeze())].values[0]
            color_summary["name"] = {'en': en, 'zh_cn': zh_cn}
            color_summary["id"] = int(idx)
            color_summary_lst.append(color_summary)

            sort_prob.append(color_summary["proportion"])

        color_summary_lst = sorted(color_summary_lst, key=lambda x: x['proportion'], reverse=True)

        return color_summary_lst

def img_color_emb(image: np.ndarray) -> np.ndarray:
    distance = get_color_distance(image, target=WEBCOLORS, color_convert=partial(bgr2hsvcone, r=10), blur_block=8)  # HSV cone
    centroid_proportion = get_color_emb(distance)
    return centroid_proportion

def img_color_summary(color_proportion: np.ndarray, num=8) -> List[dict]:
    csy = ColorSummary(color_proportion, WEBCOLORS, num=num)
    csy.merge_similar_colors(img_show=False)
    color_summary_lst = csy.get_color_summary(webcolor_name=WEBCOLOR_NAME)
    return color_summary_lst

if __name__ == '__main__':

    sss = time.time()

    image = cv2.imread('./assets/d2.png')

    '''   得到距离近的配色质心索引   '''
    distance = get_color_distance(image, target=WEBCOLORS, color_convert=partial(bgr2hsvcone, r=10), blur_block=8)   # HSV cone

    '''   得到embedding   '''
    centroid_proportion = get_color_emb(distance)

    color_img_show(centroid_proportion, colors=np.array(cv2.cvtColor(WEBCOLORS, cv2.COLOR_BGR2RGB)), save_path='./assets/clu_color_pie.jpg')

    '''   色系概述   '''
    csy = ColorSummary(centroid_proportion, WEBCOLORS, num=8)
    csy.merge_similar_colors(img_show=False)
    color_summary_lst = csy.get_color_summary(webcolor_name=WEBCOLOR_NAME)
    print(color_summary_lst)

    print('Done, use time: ', time.time() - sss)

    # '''   画图   '''
    sss = time.time()
    wei, cols = [], []
    for cen in color_summary_lst:
        wei.append(cen['proportion'])
        cols.append(cen['RGB'])
    color_img_show(np.array(wei), colors=np.array(cols), save_path='./assets/clu_color_pie.jpg')
    print('Done, use time: ', time.time() - sss)
    print()
    print()


