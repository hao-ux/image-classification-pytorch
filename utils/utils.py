import os
import numpy as np

# ----------------------------------- #
# 转为RGB图像
# ----------------------------------- #
def cvtColor(image):
    if len(np.shape(image)) != 3 or np.shape(image)[2] != 3:
        image = image.convert('RGB')
    return image 

#----------------------------------------#
#   预处理训练图片
#----------------------------------------#
def preprocess_input(x):
    x /= 255
    x -= np.array([0.485, 0.456, 0.406])
    x /= np.array([0.229, 0.224, 0.225])
    return x

#----------------------------------------#
#   获得种类
#----------------------------------------#
def get_classes(path):
    with open(path, 'r', encoding='utf-8') as f:
        c = list(map(lambda x: x.strip(), f.readlines()))
    return c