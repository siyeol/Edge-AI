
import torch
import torchvision
import argparse
import torch.nn as nn
import argparse


parser = argparse.ArgumentParser(description='M1')

parser.add_argument('--model', type=str, default="5percent_pruned_zeroelim", help='Model to convert')
args = parser.parse_args()


model = torch.load(args.model + ".pt", map_location=torch.device('cpu'))


random_input = torch.randn(1,3,32,32)
torch.onnx.export(model, random_input, args.model + ".onnx", export_params=True, opset_version=10)

