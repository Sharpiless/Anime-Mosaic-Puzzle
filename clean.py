import os
import cv2
from tqdm import tqdm

src_base = 'cropped'
# 清理无法读取的图像
for p in tqdm(os.listdir(src_base)):
    pt = os.path.join(src_base, p)
    im = cv2.imread(pt)
    if im is None:
        os.remove(pt)
        print('-[INFO] remove', p)
