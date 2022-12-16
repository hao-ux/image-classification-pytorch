import torch
import numpy as np
from PIL import Image

from utils.image_aug import Cutout, RandomHorizontalVerticalFlip, Resize, RandomRotate, RandomBright, RandomContrast, RandomSaturation, RandomHue
from utils.utils import cvtColor, preprocess_input

class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, file_list,input_shape,prob=0.8,phase='train'):
        self.file_list = file_list
        self.input_shape = input_shape
        self.phase = phase
        self.prob = prob
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        file_path = self.file_list[index].split(';')[1].split()[0]
        image = Image.open(file_path)
        image = cvtColor(image)
        if self.phase == 'train':
            image = self.img_aug(image, self.prob)
        image = np.transpose(preprocess_input(np.array(image).astype(np.float32)), [2, 0, 1])
        y = int(self.file_list[index].split(';')[0])
        return image, y
        
    def img_aug(self, img, prob=0.8):
        if isinstance(self.input_shape, (list, tuple)):
            self.input_shape = self.input_shape[0]
        aug_list = [
            Cutout(prob=prob), RandomHorizontalVerticalFlip(), Resize(self.input_shape),
            RandomRotate(prob=prob),RandomBright(prob, 0.5), RandomContrast(prob),
            RandomSaturation(prob), RandomHue(prob, 20)
        ]
        for aug in aug_list:
            img = aug(img)
        return img
        


