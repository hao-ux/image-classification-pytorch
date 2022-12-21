[简体中文](README.md) | English

## Image classification model

### 1. Training

#### 1.1 Pre-trained weights

| model | link |
| - | - |
| [mobileone](https://github.com/apple/ml-mobileone) | [s0](https://github.com/hao-ux/image-classification-pytorch/releases/download/weights/mobileone_s0_unfused.pth.tar)、[s1](https://github.com/hao-ux/image-classification-pytorch/releases/download/weights/mobileone_s1_unfused.pth.tar)、[s2](https://github.com/hao-ux/image-classification-pytorch/releases/download/weights/mobileone_s2_unfused.pth.tar)、[s3](https://github.com/hao-ux/image-classification-pytorch/releases/download/weights/mobileone_s3_unfused.pth.tar)、[s4](https://github.com/hao-ux/image-classification-pytorch/releases/download/weights/mobileone_s4_unfused.pth.tar) |
| [ghostnetv2](https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/ghostnetv2_pytorch) | width：[1.0](https://github.com/hao-ux/image-classification-pytorch/releases/download/weights/ck_ghostnetv2_10.pth.tar) |

Flower image classification data set
Link: https://pan.baidu.com/s/1zs9U76OmGAIwbYr91KQxgg
Code: bhjx

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
    'model_name'               : 'mobileone',       # ghostnetv2, mobileone
    'pretrained_weights'       : True,              # Whether pre-training weights are required重
    'model_path'               : '',                # Weights for the entire model
    'batch_size'               : 16,
    'Epochs'                   : 400,
    'learning_rate'            : 1e-2,
    'optimizer_type'           : 'SGD',
    'lr_decay_type'            : 'Cosine',
    'num_worker'               : 4,
    'save_dir'                 : './logs',          # Save the weights and loss的文件夹
    'save_period'              : 10,                # The weights are saved every 10 epochs
    'loss_func'                : 'Poly_loss'        # loss function
    }
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
3. https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/ghostnetv2_pytorch
