import tensorflow as tf
import argparse

# Argument parser
parser = argparse.ArgumentParser(description='EE379K HW4 - Starter TensorFlow code')
# Define the mini-batch size, here the size is 128 images per batch
parser.add_argument('--batch_size', type=int, default=128, help='Number of samples per mini-batch')
# Define the number of epochs for training
parser.add_argument('--epochs', type=int, default=100, help='Number of epoch to train')
args = parser.parse_args()

epochs = args.epochs
batch_size = args.batch_size

random_seed = 1
tf.random.set_seed(random_seed)

# TODO: Insert your model here
model = None

# TODO: Load the training and testing datasets

# TODO: Convert the datasets to contain only float values

# TODO: Normalize the datasets

# TODO: Encode the labels into one-hot format

# TODO: Configures the model for training using compile method

# TODO: Train the model using fit method

# TODO: Save the weights of the model in .ckpt format
