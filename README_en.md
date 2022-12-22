[简体中文](README.md) | English

## Image classification model

### Updating
- 2022.12.22, the `RandAugment` data augmentation method was added.

### 1. Training

#### 1.1 Pre-trained weights

| model | link |
| - | - |
| [mobileone](https://github.com/apple/ml-mobileone) | [s0](https://github.com/hao-ux/image-classification-pytorch/releases/download/weights/mobileone_s0_unfused.pth.tar)、[s1](https://github.com/hao-ux/image-classification-pytorch/releases/download/weights/mobileone_s1_unfused.pth.tar)、[s2](https://github.com/hao-ux/image-classification-pytorch/releases/download/weights/mobileone_s2_unfused.pth.tar)、[s3](https://github.com/hao-ux/image-classification-pytorch/releases/download/weights/mobileone_s3_unfused.pth.tar)、[s4](https://github.com/hao-ux/image-classification-pytorch/releases/download/weights/mobileone_s4_unfused.pth.tar) |
| [ghostnetv2](https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/ghostnetv2_pytorch) | width：[1.0](https://github.com/hao-ux/image-classification-pytorch/releases/download/weights/ck_ghostnetv2_10.pth.tar) |

Flower image classification data set
Link: https://pan.baidu.com/s/1zs9U76OmGAIwbYr91KQxgg
Code: bhjx

[An Improved One millisecond Mobile Backbone](https://arxiv.org/pdf/2206.04040.pdf) |
[GhostNetV2: Enhance Cheap Operation with Long-Range Attention](https://openreview.net/pdf/6db544c65bbd0fa7d7349508454a433c112470e2.pdf) |

1. Dataset file structure
    ```txt
    - data
        - train            # Training set
            - flower0
            - flower1
            - ...
        - test             # Validation set
            - flower0
            - flower1
            - ...
    ```
    Running`python process_datasets_path.py` will generate `train_cls.txt` and `valid_cls.txt`, which are needed for training.

2. The training parameters are configured in `train.py`, and the pre-trained weights are downloaded to the `weights`folder.
    ```python
    config = {
        'is_cuda'                  : True,         
        'fp16'                     : True,              # Mixed precision training  
        'classes_path'             : './classes.txt',   # category
        'input_shape'              : [224, 224],        
        'model_name'               : 'mobileone',
        'pretrained_weights'       : False,              # Whether pre-trained weights are required
        'model_path'               : '',                # Whether we need to pre-train weights Weights for the entire model
        'batch_size'               : 16,
        'Epochs'                   : 400,
        'learning_rate'            : 1e-2,
        'optimizer_type'           : 'SGD',
        'lr_decay_type'            : 'Cosine',
        'num_worker'               : 4,
        'save_dir'                 : './logs',          # Folder to save weights and losses
        'save_period'              : 10,                # The weights are saved every 10 epochs
        'loss_func_name'           : 'Poly_loss',        # loss function
        'data_aug'                 : 'original'
    }

    # ---------------------------------------------------- #
    # model_name                 optional：mobileone、ghostnetv2
    # optimizer_type             optional：SGD、Adam、Ranger
    # loss_func_name
    # optional：Poly_loss、PolyFocal、CE、LabelSmoothSoftmaxCE
    # If it is a double loss function, then 'loss_func_name' is set to a list
    # like：'loss_func_name': [('Poly_loss', 'LabelSmoothSoftmaxCE'), (0.9, 0.1)]
    # The last tuple is the weight of the corresponding loss function
    # data_aug                   optional：original、randaugment
    # lr_decay_type              optional：Cosine
    # ---------------------------------------------------- #
    ```

    Network structure parameters of mobileone , run:
    ```txt
    python summary.py --backbone mobileone
    ```
    Single GPU training, run:
    ```txt
    python train.py
    ```


### 2. Evaluate

run:
```txt
python eval.py --model_name mobileone --model_path weights/mobileone-16e-s0-flower.pth --output_dir eval_out
```
`model_name` denotes the image classification model to evaluate, `model_path` denotes the weight path, `output_dir` denotes the folder where the evaluation results are saved.

### 3. Inference

Predict the image run:
```txt
python inference.py --model_name mobileone --model_path weights/mobileone-16e-s0-flower.pth
```

### 4. Deploy

This repository only supports onnxruntime deployment for now.
1. Export onnx and run:
    ```txt
    python export_onnx.py --model_name mobileone --model_path weights/mobileone-16e-s0-flower.pth --output_path weights/mobileone-16e-s0-flower.onnx
    ```
    Here, `output_path` denotes the path exported by onnx.

2. To predict the image using the onnxruntime, run:
    ```txt
    python inference.py --model_name mobileone --model_onnx ./weights/mobileone-16e-s0-flower.onnx --infer_onnx 1
    ```


## Reference

1. https://github.com/bubbliiiing/classification-pytorch
2. https://github.com/apple/ml-mobileone