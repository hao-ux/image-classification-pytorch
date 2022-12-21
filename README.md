简体中文 | [English](README_en.md)

## 图像分类模型

### 1. 训练

#### 1.1 预训练权重

| 模型 | 链接 |
| - | - |
| [mobileone](https://github.com/apple/ml-mobileone) | [s0](https://github.com/hao-ux/image-classification-pytorch/releases/download/weights/mobileone_s0_unfused.pth.tar)、[s1](https://github.com/hao-ux/image-classification-pytorch/releases/download/weights/mobileone_s1_unfused.pth.tar)、[s2](https://github.com/hao-ux/image-classification-pytorch/releases/download/weights/mobileone_s2_unfused.pth.tar)、[s3](https://github.com/hao-ux/image-classification-pytorch/releases/download/weights/mobileone_s3_unfused.pth.tar)、[s4](https://github.com/hao-ux/image-classification-pytorch/releases/download/weights/mobileone_s4_unfused.pth.tar) |
| [ghostnetv2](https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/ghostnetv2_pytorch) | 宽度：[1.0](https://github.com/hao-ux/image-classification-pytorch/releases/download/weights/ck_ghostnetv2_10.pth.tar) |

花朵图像分类数据集
链接：https://pan.baidu.com/s/1zs9U76OmGAIwbYr91KQxgg
提取码：bhjx

1. 数据集文件结构
    ```txt
    - data
        - train            # 训练集
            - flower0
            - flower1
            - ...
        - test             # 验证集
            - flower0
            - flower1
            - ...
    ```
    运行`python process_datasets_path.py`命令，将会生成`train_cls.txt`和`valid_cls.txt`，这是训练时所需要的。

2. 训练的参数配置在`train.py`中，预训练权重下载到`weights`文件夹。
    ```python
    config = {
    'is_cuda'                  : True,         
    'fp16'                     : True,              # 混合精度训练  
    'classes_path'             : './classes.txt',   # 种类
    'input_shape'              : [224, 224],        
    'model_name'               : 'mobileone',       # ghostnetv2, mobileone
    'pretrained_weights'       : True,              # 是否需要预训练权重
    'model_path'               : '',                # 整个模型的权重
    'batch_size'               : 16,
    'Epochs'                   : 400,
    'learning_rate'            : 1e-2,
    'optimizer_type'           : 'SGD',
    'lr_decay_type'            : 'Cosine',
    'num_worker'               : 4,
    'save_dir'                 : './logs',          # 保存权重以及损失的文件夹
    'save_period'              : 10,                # 每隔10Epochs保存一次权重
    'loss_func'                : 'Poly_loss'        # 损失函数
    }
    ```

    mobileone网络结构的参数，运行：
    ```txt
    python summary.py --backbone mobileone
    ```
    单GPU训练，运行：
    ```txt
    python train.py
    ```


### 2. 评估

运行：
```txt
python eval.py --model_name mobileone --model_path weights/mobileone-16e-s0-flower.pth --output_dir eval_out
```
其中，`model_name`表示要评估的图像分类模型，`model_path`表示权重路径，`output_dir`表示保存评估结果的文件夹。

### 3. 推理

预测图片运行：
```txt
python inference.py --model_name mobileone --model_path weights/mobileone-16e-s0-flower.pth
```

### 4. 部署

本仓库暂时只支持onnxruntime部署。
1. 导出onnx，运行：
    ```txt
    python export_onnx.py --model_name mobileone --model_path weights/mobileone-16e-s0-flower.pth --output_path weights/mobileone-16e-s0-flower.onnx
    ```
    其中，output_path表示onnx导出的路径。

2. 使用onnxruntime推理图片，运行：
    ```txt
    python inference.py --model_name mobileone --model_onnx ./weights/mobileone-16e-s0-flower.onnx --infer_onnx 1
    ```


## 参考

1. https://github.com/bubbliiiing/classification-pytorch
2. https://github.com/apple/ml-mobileone
3. https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/ghostnetv2_pytorch
