import argparse
import torch
import numpy as np
import torch.nn as nn
import os

from nets.mobileone import reparameterize_model
from nets import get_model_from_name
from utils.utils import get_classes, cvtColor, preprocess_input
from utils.image_aug import Resize
from utils.metrics import evaluteTop1_5

class Eval(object):
    def __init__(self, model_name, model_path, input_shape=None, is_cuda=True):
        if input_shape is None:
            self.input_shape = [224, 224]
        self.model_name = model_name
        self.model_path = model_path
        self.is_cuda = is_cuda

        self.class_names = get_classes('./classes.txt')
        self.num_classes = len(self.class_names)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.model_name in ['mobileone']:
            self.model = get_model_from_name[model_name](num_classes=self.num_classes, variant="s0")
            self.model.load_state_dict(torch.load(self.model_path, map_location=device))
            self.model = reparameterize_model(self.model)
        self.model.eval()
        print(f"Load {self.model_name} sucessfully")
        if self.is_cuda:
            self.model = nn.DataParallel(self.model)
            self.model = self.model.cuda()
    
    def detect(self, image):
        image = cvtColor(image)
        image_data = Resize(self.input_shape)(image)
        image_data  = np.transpose(np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0), (0, 3, 1, 2))
        with torch.no_grad():
            photo   = torch.from_numpy(image_data)
            if self.is_cuda:
                photo = photo.cuda()
            preds   = torch.softmax(self.model(photo)[0], dim=-1).cpu().numpy()
        return preds
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default='mobileone',help='select model',choices=['mobileone']
    )
    parser.add_argument(
        "--model_path", type=str, default="weights/mobileone-16e-s0-flower.pth", help="select model path"
    )
    parser.add_argument(
        "--output_dir", type=str, default="eval_out", help="select metrics output dir"
    )

    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    classfication = Eval(args.model_name, args.model_path)
    with open("./valid_cls.txt","r") as f:
        lines = f.readlines()
    top1, top5, Recall, Precision = evaluteTop1_5(classfication, lines, args.output_dir)
    print("top-1 accuracy = %.2f%%" % (top1*100))
    print("top-5 accuracy = %.2f%%" % (top5*100))
    print("mean Recall = %.2f%%" % (np.mean(Recall)*100))
    print("mean Precision = %.2f%%" % (np.mean(Precision)*100))
    
    
