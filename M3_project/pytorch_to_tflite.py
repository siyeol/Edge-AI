import torch
import torchvision
import argparse
import torch.nn as nn
import tensorflow as tf
import numpy as np
from torch.autograd import Variable
from pytorch2keras import pytorch_to_keras

parser = argparse.ArgumentParser(description='M3')

parser.add_argument('--model', type=str, default="0.9thr_pruned", help='Model to convert')
args = parser.parse_args()


model = torch.load(args.model + ".pt", map_location=torch.device('cpu'))


random_input = torch.randn(1,3,32,32)
torch.onnx.export(model, random_input, args.model + ".onnx", export_params=True, opset_version=10)




input_np = np.random.uniform(0, 1, (1, 3, 32, 32))
input_var = Variable(torch.FloatTensor(input_np))

k_model = pytorch_to_keras(model, input_var, [(3, 32, 32,)], change_ordering=True,  verbose=True)


k_model.summary()


optim=True
# Convert the model
converter = tf.lite.TFLiteConverter.from_keras_model(k_model)
if optim:
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the model
with open(args.model + ".tflite",'wb') as f:
    f.write(tflite_model)
