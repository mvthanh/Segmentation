import numpy as np
import cv2
import os


def read_data(path, size=(64, 64), flag=0):
    res = []
    for file in os.listdir(path):
        img_path = os.path.join(path, file)
        res.append(img_path)
    res = np.sort(res)
    res_img = []
    for img_path in res:
        curr_img = cv2.imread(img_path, flag)/255.0
        curr_img = cv2.resize(curr_img, size, interpolation=cv2.INTER_CUBIC)
        res_img.append(curr_img)
    res_img = res_img[:256]
    return np.array(res_img).reshape((len(res_img), size[0], size[1], 3 if flag == 1 else 1))


a = read_data('./Data/Training_Images')
print(a.shape)
