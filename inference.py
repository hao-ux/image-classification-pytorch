import torch
import torch.nn as nn
import numpy as np
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import time

from nets import get_model_from_name
from nets.mobileone import reparameterize_model
from utils.utils import get_classes
from utils.image_aug import Resize
from utils.utils import get_classes, cvtColor, preprocess_input

def softmax(x):
    x = np.exp(x) / np.sum(np.exp(x))
    return x

class Infer(object):
    def __init__(self, model_name, model_path, onnxruntime=False, input_shape=None, is_cuda=True, onnx_path=None):
        if input_shape is None:
            self.input_shape = [224, 224]
        
        self.is_cuda = is_cuda
        self.model_name = model_name
        self.model_path = model_path
        self.class_names = get_classes('./classes.txt')
        self.num_classes = len(self.class_names)
        self.onnxruntime = onnxruntime
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not onnxruntime:
            if self.model_name in ['mobileone']:
                self.model = get_model_from_name[model_name](num_classes=self.num_classes, variant="s0")
                self.model.load_state_dict(torch.load(self.model_path, map_location=device))
                self.model = reparameterize_model(self.model)
            self.model.eval()
            print(f"Load {self.model_name} sucessfully")
            if self.is_cuda:
                self.model = nn.DataParallel(self.model)
                self.model = self.model.cuda()
        else:
            import onnxruntime as ort
            self.session_option = ort.SessionOptions()
            self.session_option.log_severity_level = 3
            self.ort_session = ort.InferenceSession(onnx_path,providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'], sess_options=self.session_option)
            self.model_inputs = self.ort_session.get_inputs()
            self.input_names = [self.model_inputs[i].name for i in range(len(self.model_inputs))]
            self.model_outputs = self.ort_session.get_outputs()
            self.output_names = [self.model_outputs[i].name for i in range(len(self.model_outputs))]
            print(f"Load  onnx model of{self.model_name} sucessfully")
            
    
    def detect(self, image):
        image = cvtColor(image)
        image_data = Resize(self.input_shape)(image)
        image_data  = np.transpose(np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0), (0, 3, 1, 2))
        with torch.no_grad():
            photo   = torch.from_numpy(image_data)
            if self.is_cuda:
                photo = photo.cuda()
            preds = torch.softmax(self.model(photo)[0], dim=-1).cpu().numpy()
        class_name  = self.class_names[np.argmax(preds)]
        probability = np.max(preds)
        plt.subplot(1, 1, 1)
        plt.imshow(np.array(image))
        plt.title('Class:%s Probability:%.3f' %(class_name, probability))
        plt.show()
        return class_name
    
    def onnx_detect(self, image):
        print(f"{'-'*20}onnx_detect{'-'*20}")
        old_image = image.copy()
        image = cvtColor(image)
        image_data = Resize(self.input_shape)(image)
        image_data  = np.transpose(np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0), (0, 3, 1, 2))
        outputs = self.ort_session.run(self.output_names, {input_name: image_data for input_name in self.input_names})
        outputs = softmax(outputs)
        class_name  = self.class_names[np.argmax(outputs)]
        probability = np.max(outputs)
        plt.subplot(1, 1, 1)
        plt.imshow(np.array(old_image))
        plt.title('Class:%s Probability:%.3f' %(class_name, probability))
        plt.show()
        return class_name

        
    
    def get_fps(self, n=50):
        if not self.onnxruntime:
            print("Get fps")
            with torch.no_grad():
                photo   = torch.from_numpy(np.random.randn(1, 3, 224, 224).astype(np.float32))
                if self.is_cuda:
                    photo = photo.cuda()
                start = time.time()
                for _ in range(n):
                    preds = self.model(photo)
                end = time.time()
                print(f"time: {(end-start)/n}, fps: {n/(end-start)}")
        else:
            print("Get onnx model fps")
            start = time.time()
            for _ in range(n):
                outputs = self.ort_session.run(self.output_names, {input_name: np.random.randn(1, 3, 224, 224).astype(np.float32) for input_name in self.input_names})
            end = time.time()
            print(f"time: {(end-start)/n}, fps: {n/(end-start)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default='mobileone',help='select model',choices=['mobileone']
    )
    parser.add_argument(
        "--model_path", type=str, default="weights/mobileone-16e-s0-flower.pth", help="select model path"
    )
    parser.add_argument(
        "--infer_onnx", type=int, default=0, choices=[1,0]
    )
    parser.add_argument(
        "--get_fps", type=int, default=0, choices=[1,0]
    )
    parser.add_argument(
        "--model_onnx", type=str, default='./weights/mobileone-16e-s0-flower.onnx'
    )

    args = parser.parse_args()
    image = Image.open('./img/flower4.jpg')
    infer = Infer(args.model_name, args.model_path, args.infer_onnx, onnx_path=args.model_onnx)
    if not args.get_fps:
        class_name = (
            infer.onnx_detect(image) if args.infer_onnx else infer.detect(image)
        )
        print("Prediction:", class_name)
    if args.get_fps:
        infer.get_fps()
    