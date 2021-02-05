import os
import cv2
import numpy as np
import pickle as pkl
from tqdm import tqdm
import imutils
import matplotlib.pyplot as plt


def Generate_data(src_base, src_pt, tgt_size, pick_num):
    if os.path.exists(src_pt):
        with open(src_pt, 'rb') as f:
            data = pkl.load(f)
    else:
        data = {'img': [], 'pixel': []}
        for p in tqdm(os.listdir(src_base)[:pick_num]):
            pt = os.path.join(src_base, p)
            im = cv2.imread(pt)
            if im is None:
                continue
            im = cv2.resize(im, (tgt_size, tgt_size))
            pixel = np.mean(np.reshape(
                im[int(tgt_size*0.5):, :, :], (-1, 3)), axis=0)  # 只取头发部分颜色
            data['img'].append(p)
            data['pixel'].append(pixel)
        data['pixel'] = np.array(data['pixel'])
        with open(src_pt, 'wb') as f:
            pkl.dump(data, f)
    return data


def Match_img(pixel, data):
    diff = np.abs(data['pixel'] - pixel)
    diff = np.mean(diff, axis=-1)
    min_index = np.argmin(diff)
    return min_index

def Match_img_v2(pixel, data):
    sum_ = np.sum(data['pixel'], axis=-1)
    ab_ids = np.where(sum_ == 0)
    diff = np.abs(data['pixel'] - pixel)
    diff = np.mean(diff, axis=-1)
    diff[ab_ids] = 1e+4
    min_index = np.argmin(diff)
    return min_index


if __name__ == '__main__':

    src_base = 'cropped'  # 动漫人脸路径
    src_pt = 'data.pkl' # 临时数据保存文件
    tgt_size = 32 # 每张小图的大小
    pick_num = 30000  # 只取10000张
    width = 100 # 放缩原图的宽度
    without_repeat = True # 是否允许重复

    data = Generate_data(src_base, src_pt, tgt_size, pick_num) # 生成临时数据（每张小图的平均像素）
    print(data['pixel'].shape)

    tgt_image_pt = 'demo.jpg' # 原图
    raw_image = cv2.imread(tgt_image_pt) # 读取原图
    tgt_image = imutils.resize(raw_image, width=width)  # 100 * 32

    w, h = tgt_image.shape[:2]
    # 遍历像素
    results = []
    for i in tqdm(range(w)):
        base_im = []
        for j in range(h):
            pixel = tgt_image[i, j] # 获取原图(i, j)位置像素
            if without_repeat:
                match_id = Match_img_v2(pixel, data)
                data['pixel'][match_id] = np.zeros((3, ))
            else:
                match_id = Match_img(pixel, data)
            match_pt = os.path.join(src_base, data['img'][match_id])
            match_im = cv2.imread(match_pt)
            base_im.append(cv2.resize(match_im, (tgt_size, tgt_size)))
        results.append(np.hstack(base_im))

    final = np.vstack(results)
    cv2.imwrite('result.png', final)
    print(final.shape)
    plt.imshow(raw_image[:, :, [2, 1, 0]])
    plt.show()
    plt.imshow(final[:, :, [2, 1, 0]])
    plt.show()
    
