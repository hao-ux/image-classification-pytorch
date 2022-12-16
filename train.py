
config = {
    'is_cuda'                  : True,
    'fp16'                     : True,
    'classes_path'             : './classes.txt',
    'input_shape'              : [224, 224],
    'backbone'                 : 'mobileone',
    'pretrained_weights'       : True,
    'model_path'               : '',
    'batch_size'               : 32,
    'learning_rate'            : 1e-2,
    'optimizer_type'           : 'SGD',
    'lr_decay_type'            : 'Cosine',
    'num_worker'               : 4,
    'save_period'              : 10,
    'loss_func'                : 'Poly_loss'
}

