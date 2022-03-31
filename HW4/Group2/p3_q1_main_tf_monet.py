import tensorflow as tf
import argparse
import matplotlib.pyplot as plt
from torchsummary import summary
import numpy as np
import time

# Argument parser
parser = argparse.ArgumentParser(description='EE379K HW4 - Starter TensorFlow code')
# Define the mini-batch size, here the size is 128 images per batch
parser.add_argument('--batch_size', type=int, default=128, help='Number of samples per mini-batch')
# Define the number of epochs for training
parser.add_argument('--epochs', type=int, default=100, help='Number of epoch to train')
# Define the model class
parser.add_argument('--model_class', type=str, default="MobileNetV1", help='Model to train')
args = parser.parse_args()

epochs = args.epochs
batch_size = args.batch_size
model_to_train = args.model_class

random_seed = 1
tf.random.set_seed(random_seed)

# TODO: Insert your model here
if (model_to_train == "MobileNetV1"):
    from models.mobilenet_tf import MobileNetv1

# used to limit the memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


# use GPU
with tf.device("/device:GPU:0"):
  model = MobileNetv1()
  print(model.summary())

  # TODO: Load the training and testing datasets
  (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
  
  # TODO: Convert the datasets to contain only float values
  train_images_norm = train_images.astype('float32')
  test_images_norm = test_images.astype('float32')

  # TODO: Normalize the datasets
  train_images_norm = train_images_norm / 255.0
  test_images_norm = test_images_norm / 255.0

  # TODO: Encode the labels into one-hot format
  train_labels = tf.keras.utils.to_categorical(train_labels)
  test_labels = tf.keras.utils.to_categorical(test_labels)

  # TODO: Configures the model for training using compile method
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy',
                metrics=['accuracy'])


  # TODO: Train the model using fit method
  start = time.time()
  history = model.fit(train_images_norm, train_labels, epochs=epochs, batch_size=batch_size,
                      validation_data=(test_images_norm, test_labels), verbose=1)
  end = time.time()

  total_training_time = end - start

  print("Total time: ", total_training_time, "seconds")


  test_loss, test_acc = model.evaluate(test_images_norm, test_labels, verbose=2)
  print("test_acc :", test_acc)
  print("==", model_to_train, "==")
  print(history.history['val_accuracy'])


  # TODO: Save the weights of the model in .ckpt format
  model.save_weights("./models/monet/mobilenet_tf.ckpt")

