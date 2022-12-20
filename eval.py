import argparse

class Eval(object):
    def __init__(self, model_name, model_path, output_dir, eval_dir):
        self.model_name = model_name
        self.model_path = model_path
        self.output_dir = output_dir
        self.eval_dir = eval_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default='mobilenet',help='select model',choices=['mobilenet']
    )
    parser.add_argument(
        "--model_path", type=str, default="weights/mobilenet_s0_dog.pth", help="select model path"
    )
    parser.add_argument(
        "--output_dir", type=str, default="metrics_out", help="select metrics output dir"
    )
    parser.add_argument(
        "--eval_dir", type=str, default="valid_cls.txt"
    )
    args = parser.parse_args()
    
    