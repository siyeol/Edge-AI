import time
import argparse
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import tflite_runtime.interpreter as tflite

# TODO: add argument parser
parser = argparse.ArgumentParser(description='EE379K HW4')

# TODO: add one argument for selecting VGG or MobileNet-v1 models
parser.add_argument('--model', type=str, default="VGG11", help='Model to train')
args = parser.parse_args()

# TODO: Modify the rest of the code to use the arguments correspondingly

tflite_model_name = args.model + ".tflite" # TODO: insert TensorFlow Lite model name

# Get the interpreter for TensorFlow Lite model
interpreter = tflite.Interpreter(model_path=tflite_model_name)

# Very important: allocate tensor memory
interpreter.allocate_tensors()

# Get the position for inserting the input Tensor
input_details = interpreter.get_input_details()
# Get the position for collecting the output prediction
output_details = interpreter.get_output_details()

# Label names for CIFAR10 Dataset
label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

test_correct = 0
test_total =0
total_inference_time = 0

for filename in tqdm(os.listdir("./test_deployment")):
  with Image.open(os.path.join("./test_deployment", filename)).resize((32, 32)) as img:
    
    
    input_image = np.float32(img) / 255.
    
    input_image = np.expand_dims(np.float32(input_image), axis=0)
    
    

    # Set the input tensor as the image
    interpreter.set_tensor(input_details[0]['index'], input_image)

    # Run the actual inference
    start = time.time()
    interpreter.invoke()
    per_image_inference_time = time.time()-start

    total_inference_time += per_image_inference_time

    # Get the output tensor
    pred_tflite = interpreter.get_tensor(output_details[0]['index'])

    # Find the prediction with the highest probability
    top_prediction = np.argmax(pred_tflite[0])

    # Get the label of the predicted class
    pred_class = label_names[top_prediction]

    test_total += 1
    if pred_class in filename:
      test_correct += 1

test_accuracy = 100. * test_correct / test_total
print("Test Accuracy : ",test_accuracy)
print('Total time for inference : %.4f seconds' % (total_inference_time))
print("Sec/Acc : ",total_inference_time,"/",test_accuracy)
