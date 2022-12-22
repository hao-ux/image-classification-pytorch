import torch
import os
from torch.cuda.amp import GradScaler as GradScaler
from torch.cuda.amp import autocast
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from utils.dataloader import ClassificationDataset, detection_collate
from utils.utils import get_classes, weights_init, show_config
from utils.loss import (
    Poly1CrossEntropyLoss, Poly1FocalLoss, LabelSmoothSoftmaxCE, JointLoss
)
from nets import get_model_from_name
from utils.callbacks import LossHistory
from nets.mobileone import reparameterize_model
from utils.optimizer import Ranger

config = {
    'is_cuda'                  : True,         
    'fp16'                     : True,              # 混合精度训练  
    'classes_path'             : './classes.txt',   # 种类
    'input_shape'              : [224, 224],        
    'model_name'               : 'mobileone',
    'pretrained_weights'       : False,              # 是否需要预训练权重
    'model_path'               : '',                # 整个模型的权重
    'batch_size'               : 16,
    'Epochs'                   : 400,
    'learning_rate'            : 1e-2,
    'optimizer_type'           : 'SGD',
    'lr_decay_type'            : 'Cosine',
    'num_worker'               : 4,
    'save_dir'                 : './logs',          # 保存权重以及损失的文件夹
    'save_period'              : 10,                # 每隔10Epochs保存一次权重
    'loss_func_name'           : 'Poly_loss',        # 损失函数
    'data_aug'                 : 'original'
}

# ---------------------------------------------------- #
# model_name                 可选：mobileone、ghostnetv2
# optimizer_type             可选：SGD、Adam、Ranger
# loss_func_name
# 可选：Poly_loss、PolyFocal、CE、LabelSmoothSoftmaxCE
# 若设置为是双损失函数，则'loss_func_name'设成列表形式
# 如：'loss_func_name': [('Poly_loss', 'LabelSmoothSoftmaxCE'), (0.9, 0.1)]
# 后面一个元组为对应损失函数的权重
# data_aug                   可选：original、randaugment
# lr_decay_type              可选：Cosine
# ---------------------------------------------------- #


if __name__ == '__main__':
    
    is_cuda              = config['is_cuda']
    fp16                 = config['fp16']
    classes_path         = config['classes_path']
    input_shape          = config['input_shape']
    model_name           = config['model_name']
    pretrained_weights   = config['pretrained_weights']
    model_path           = config['model_path']
    batch_size           = config['batch_size']
    learning_rate        = config['learning_rate']
    optimizer_type       = config['optimizer_type']
    lr_decay_type        = config['lr_decay_type']
    num_worker           = config['num_worker']
    save_dir             = config['save_dir']
    save_period          = config['save_period']
    loss_func_name       = config['loss_func_name']
    Epochs               = config['Epochs']
    data_aug             = config['data_aug']
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    loss_func_dict = {
            'Poly_loss': Poly1CrossEntropyLoss(num_classes), 'PolyFocal': Poly1FocalLoss(num_classes),
            'CE': nn.CrossEntropyLoss(),
            'LabelSmoothSoftmaxCE': LabelSmoothSoftmaxCE()
        }
    if isinstance(loss_func_name, str):
        loss_func = loss_func_dict[loss_func_name]
    else:
        first_loss, first_loss_weight = loss_func_dict[loss_func_name[0][0]], loss_func_name[1][0]
        second_loss, second_loss_weight = loss_func_dict[loss_func_name[0][1]], loss_func_name[1][1]
        loss_func = JointLoss(first_loss, second_loss, first_loss_weight, second_loss_weight)
    if model_name in ['mobileone']:
        model = get_model_from_name[model_name](num_classes=num_classes, variant="s0", pretrained=pretrained_weights, inference_mode=False)
    else:
        model = get_model_from_name[model_name](num_classes=num_classes, pretrained=pretrained_weights)
    if not pretrained_weights:
        weights_init(model)
    if model_path != "":
        print(f'Load weights {model_path}.')
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
        print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
        print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")

    loss_history = LossHistory(save_dir, model, input_shape=input_shape)
    scaler = GradScaler() if fp16 else None
    model_train = model.train()
    if is_cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()
    
    with open('./train_cls.txt', encoding='utf-8') as f:
        train_lines = f.readlines()
    with open('./valid_cls.txt', encoding='utf-8') as f:
        val_lines   = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)
    np.random.seed(10101)
    np.random.shuffle(train_lines)
    np.random.seed(None)
    
    train_dataset   = ClassificationDataset(train_lines, input_shape, phase='train', data_aug=data_aug)
    val_dataset     = ClassificationDataset(val_lines, input_shape, phase='valid')
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_worker, pin_memory=True, 
                                drop_last=True, collate_fn=detection_collate)
    valid_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_worker, pin_memory=True, 
                                drop_last=True, collate_fn=detection_collate)
    
    show_config(config)
    wanted_step = 3e4 if optimizer_type == "SGD" else 1e4
    total_step  = num_train // batch_size * Epochs
    if total_step <= wanted_step:
        wanted_epoch = wanted_step // (num_train // batch_size) + 1
        print("\n\033[1;33;44m[Warning] 使用%s优化器时，建议将训练总步长设置到%d以上。\033[0m"%(optimizer_type, wanted_step))
        print("\033[1;33;44m[Warning] 本次运行的总训练数据量为%d，Unfreeze_batch_size为%d，共训练%d个Epoch，计算出总训练步长为%d。\033[0m"%(num_train, batch_size, Epochs, total_step))
        print("\033[1;33;44m[Warning] 由于总训练步长为%d，小于建议总步长%d，建议设置总世代为%d。\033[0m"%(total_step, wanted_step, wanted_epoch))
    optimizer = {
            'Adam'  : optim.Adam(model_train.parameters(), learning_rate, betas = (0.9, 0.999), weight_decay=5e-4),
            'SGD'   : optim.SGD(model_train.parameters(), learning_rate, momentum = 0.9, nesterov=True),
            'Ranger': Ranger(model_train.parameters(), learning_rate)
        }[optimizer_type]
    scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer,T_max =  Epochs)
    epoch_step      = num_train // batch_size
    epoch_step_val  = num_val // batch_size
    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")
    
    
    for epoch in range(Epochs):
        total_loss      = 0
        total_accuracy  = 0
        val_loss        = 0
        val_accuracy    = 0
        print("Start training!")
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epochs}',postfix=dict,mininterval=0.3)
        model_train.train()
        for idx, batch in enumerate(train_dataloader):
            if idx >= epoch_step: 
                break
            images, targets = batch
            with torch.no_grad():
                if is_cuda:
                    images  = images.cuda()
                    targets = targets.cuda()
            optimizer.zero_grad()
            if not fp16:
                outputs = model_train(images)
                loss_value = loss_func(outputs, targets)
                loss_value.backward()
                optimizer.step()
            else:
                with autocast():
                    outputs = model_train(images)
                    loss_value = loss_func(outputs, targets)
                scaler.scale(loss_value).backward()
                scaler.step(optimizer)
                scaler.update()
                
            total_loss += loss_value.item()
            with torch.no_grad():
                accuracy = torch.mean((torch.argmax(F.softmax(outputs, dim=-1), dim=-1) == targets).type(torch.FloatTensor))
                total_accuracy += accuracy.item()
            pbar.set_postfix(**{'total_loss': total_loss / (idx + 1), 
                                'accuracy'  : total_accuracy / (idx + 1), 
                                'lr'        : scheduler.get_last_lr()[0]})
            pbar.update(1)
        scheduler.step()
        pbar.close()
        print('Finsh Training!')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epochs}',postfix=dict,mininterval=0.3)
        model_eval = model_train.eval()
        if model_name in ['mobileone']:
            model_eval = reparameterize_model(model_eval)
        for idx, batch in enumerate(valid_dataloader):
            if idx >= epoch_step_val:
                break
            images, targets = batch
            with torch.no_grad():
                if is_cuda:
                    images  = images.cuda()
                    targets = targets.cuda()

                optimizer.zero_grad()
                outputs = model_eval(images)
                loss_value = loss_func(outputs, targets)
                val_loss    += loss_value.item()
                accuracy        = torch.mean((torch.argmax(F.softmax(outputs, dim=-1), dim=-1) == targets).type(torch.FloatTensor))
                val_accuracy    += accuracy.item()
            pbar.set_postfix(**{'total_loss': val_loss / (idx + 1),
                                'accuracy'  : val_accuracy / (idx + 1), 
                                'lr'        : scheduler.get_last_lr()[0]})
            pbar.update(1)
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
        print('Epoch:' + str(epoch + 1) + '/' + str(Epochs))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
        
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epochs:
            torch.save(model.state_dict(), os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)))
        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))
        
            
                
                    

    
        
    
        
    