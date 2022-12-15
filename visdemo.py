from PIL import Image
from utils.image_aug import (
    Cutout, RandomHorizontalVerticalFlip, Resize, RandomRotate, RandomBright, RandomContrast,
    RandomSaturation, RandomHue
)
from utils.utils import cvtColor
from matplotlib import pyplot as plt
import numpy as np


aug_names = {
    'cutout': Cutout(), 'randomhorizontalverticalflip': RandomHorizontalVerticalFlip(), 'resize': Resize(224),
    'randomrotate': RandomRotate(), 'randomrright': RandomBright(), 'randomcontrast': RandomContrast(), 
    'randomsaturation': RandomSaturation(), 'randomhue': RandomHue()
}

# ------------------------------------- #
# 图像增强可视化类
# ------------------------------------- #
class DataAugVision(object):
    def __init__(self, funcname):
        self.funcname = funcname
        if isinstance(funcname, list):
            self.name_func = []
            for i in range(len(self.funcname)):
                self.name_func.append(aug_names[funcname[i]])
        else:
            self.name_func = aug_names[funcname]
        
    def vision(self, img):
        img1 = img.copy()
        if isinstance(self.funcname, list):
            for i in range(len(self.funcname)):
                img = self.name_func[i](img)
            plot(img1, img, f'transformed by {self.funcname}', False)
        else:
            img = self.name_func(img)
            plot(img1, img, f'transformed by {self.funcname}')
        

def plot(img1, img, text='', is_title=True):
    plt.subplot(1, 2, 1)
    plt.imshow(np.array(img1))
    plt.title('original img')

    plt.subplot(1, 2, 2)
    plt.imshow(np.array(img))
    if is_title:
        plt.title(text)
    plt.show()

if __name__ == '__main__':
    img = Image.open('./img/flower4.jpg')
    img = cvtColor(img)
    
    vis = DataAugVision(['randomhue', 'randomrotate', 'randomhorizontalverticalflip', 'randomsaturation'])
    vis.vision(img)
    
    