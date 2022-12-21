import argparse
import torch
import onnx

from nets.mobileone import reparameterize_model
from nets import get_model_from_name
from utils.utils import get_classes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default='ghostnetv2',help='select model',choices=['mobileone', 'ghostnetv2']
    )
    parser.add_argument(
        "--model_path", type=str, default="weights/ghostnetv2.pth", help="select model path"
    )
    parser.add_argument(
        "--output_path", type=str, default="weights/ghostnetv2.onnx", help="select export dir"
    )
    parser.add_argument(
        "--input_shape", type=int, default=224, help="input_shape"
    )
    parser.add_argument(
        "--simplify", type=bool, default=True, help="if simplify onnx"
    )

    args = parser.parse_args()
    class_names = get_classes('./classes.txt')
    num_classes = len(class_names)
    if args.model_name in ['mobileone']:
        model = get_model_from_name[args.model_name](num_classes=num_classes, variant="s0")
        model.load_state_dict(torch.load(args.model_path))
        model = reparameterize_model(model)
    else:
        model = get_model_from_name[args.model_name](num_classes=num_classes)
        model.load_state_dict(torch.load(args.model_path))
    model.eval()
    img= torch.zeros(1, 3, args.input_shape, args.input_shape).to('cpu')
    input_layer_names   = ["images"]
    output_layer_names  = ["output"]
    print(f'Starting export with onnx {onnx.__version__}.')
    torch.onnx.export(model,
                        img,
                        f               = args.output_path,
                        verbose         = False,
                        opset_version   = 12,
                        training        = torch.onnx.TrainingMode.EVAL,
                        do_constant_folding = True,
                        input_names     = input_layer_names,
                        output_names    = output_layer_names,
                        dynamic_axes    = None)
    model_onnx = onnx.load(args.output_path)
    onnx.checker.check_model(model_onnx)

    if args.simplify:
        import onnxsim
        print(f'Simplifying with onnx-simplifier {onnxsim.__version__}.')
        model_onnx, check = onnxsim.simplify(
            model_onnx,
            dynamic_input_shape=False,
            input_shapes=None)
        assert check, 'assert check failed'
        onnx.save(model_onnx, args.output_path)

    print(f'Onnx model save as {args.output_path}')
    
    
    