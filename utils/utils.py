import numpy as np

# ----------------------------------- #
# 转为RGB图像
# ----------------------------------- #
def cvtColor(image):
    if len(np.shape(image)) != 3 or np.shape(image)[2] != 3:
        image = image.convert('RGB')
    return image 