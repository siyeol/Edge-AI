import torch
import torchvision
import argparse

# from models.vgg11_pt import VGG
#from models.vgg16_pt import VGG


# Argument parser
parser = argparse.ArgumentParser(description='PyTorch to ONNX conversion')
parser.add_argument('--model_type', type=str, default='VGG', help='model type for conversion')
parser.add_argument('--pytorch_model_path', type=str, default='VGG11.pt', help='location of the pytorch model')
parser.add_argument('--onnx_model_path', type=str, default='VGG11_pt.onnx', help='location to store the converted onnx model')
parser.add_argument('--model', type=str, default="VGG11", help='Model to train')
args = parser.parse_args()

model_to_convert = args.model


if (model_to_convert == "VGG11"):
    from models.vgg11_pt import VGG
elif (model_to_convert == "VGG16"):    
    from models.vgg16_pt import VGG


model = VGG()

model.load_state_dict(torch.load(args.pytorch_model_path, map_location=torch.device('cpu')))

random_input = torch.randn(1,3,32,32)
torch.onnx.export(model, random_input, args.onnx_model_path, export_params=True, opset_version=10)
