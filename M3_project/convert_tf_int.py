import torch
import torchvision
import argparse
import torch.nn as nn
import tensorflow as tf
import numpy as np
from torch.autograd import Variable
from pytorch2keras import pytorch_to_keras
import numpy as np
import os
# import tqdm
from PIL import Image
import logging
assert float(tf.__version__[:3]) >= 2.3

parser = argparse.ArgumentParser(description='M3')

parser.add_argument('--model', type=str, default="0.9thr_pruned", help='Model to convert')
args = parser.parse_args()
# logging.getLogger("tensorflow").setLevel(logging.DEBUG)


model = torch.load(args.model + ".pt", map_location=torch.device('cpu'))



input_np = np.random.uniform(0, 1, (1, 3, 32, 32))
input_var = Variable(torch.FloatTensor(input_np))

k_model = pytorch_to_keras(model, input_var, [(3, 32, 32,)], change_ordering=True,  verbose=True)


k_model.summary()


(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

mean = np.array((0.4914, 0.4822, 0.4465))
std = np.array((0.2023, 0.1994, 0.2010))

test_images = ((test_images.astype(np.float32) / 255.0 - mean) / std).astype(np.float32)





def representative_data_gen():
  for input_value in tf.data.Dataset.from_tensor_slices(test_images).batch(1).take(10000):
    yield [input_value]



# train_images_list = np.empty((1000,32,32,3))

# for filename in (os.listdir("./test_deployment")):
#     with Image.open(os.path.join("./test_deployment", filename)).resize((32, 32)) as img:
#         # normalize image
#         input_image = (np.float32(img) / 255. - mean) / std

#         input_image = np.expand_dims(np.float32(input_image), axis=0)

#         # train_images_list.append(input_image)
#         # np.append(train_images, input_image)
#         data = np.asarray(input_image)
#         # print(data.shape)
#         train_images_list = np.append(train_images_list, data, axis=0)

optim=True
# Convert the model
converter = tf.lite.TFLiteConverter.from_keras_model(k_model)
if optim:
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# converter.inference_input_type = tf.uint8
# converter.inference_output_type = tf.uint8

tflite_model = converter.convert()

# Save the model
with open(args.model + "stat" + ".tflite",'wb') as f:
    f.write(tflite_model)
