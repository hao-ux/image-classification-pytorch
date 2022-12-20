import torch
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

#----------------------------------------#
#   初始化权值
#----------------------------------------#
def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    f'initialization method [{init_type}] is not implemented'
                )
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print(f'initialize network with {init_type} type')
    net.apply(init_func)
    
#----------------------------------------#
#   打印参数
#----------------------------------------#
def show_config(config):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in config.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)

