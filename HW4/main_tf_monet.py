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
parser.add_argument('--model_class', type=str, default="VGG11", help='Model to train')
args = parser.parse_args()

epochs = args.epochs
batch_size = args.batch_size
model_to_train = args.model_class

random_seed = 1
tf.random.set_seed(random_seed)

# TODO: Insert your model here
if (model_to_train == "VGG11"):
    from models.vgg11_tf import VGG
elif (model_to_train == "VGG16"):
    from models.vgg16_tf import VGG
elif (model_to_train == "MobileNet"):
    from models.mobilenet_tf import MobileNetv1

with tf.device("/device:GPU:0"):
    model = VGG()
    print(model.summary())

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # use cuda if available
    # model.to(device)        # put model to cuda
    # summary(model, (3, 32, 32))

    # TODO: Load the training and testing datasets

    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    # class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # Prepare the training dataset.
    # train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    # train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    # # Prepare the testing dataset.
    # test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    # test_dataset = test_dataset.batch(batch_size)

    # TODO: Convert the datasets to contain only float values
    train_images_norm = train_images.astype('float32')
    test_images_norm = test_images.astype('float32')

    # TODO: Normalize the datasets
    train_images_norm = train_images_norm / 255.0
    test_images_norm = test_images_norm / 255.0

    # TODO: Encode the labels into one-hot format
    # ORIGINAL
    # train_labels = tf.python.keras.utils.np_utils.to_categorical(train_labels)
    # test_labels = tf.python.keras.utils.np_utils.to_categorical(test_labels)

    # PROBABLY THE CORRECT VERSION
    train_labels = tf.keras.utils.to_categorical(train_labels)
    test_labels = tf.keras.utils.to_categorical(test_labels)

    # train_labels_onehot = tf.one_hot(train_labels, depth=len(np.unique(y_train)))
    # test_labels_onehot = tf.one_hot(test_labels, depth=len(np.unique(y_train)))

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

    # plt.plot(history.history['accuracy'], label='accuracy')
    # plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.grid()
    # plt.legend(loc='lower right')

    test_loss, test_acc = model.evaluate(test_images_norm, test_labels, verbose=2)
    print("test_acc :", test_acc, "%")
    print("==", model_to_train, "==")
    print(history.history['val_accuracy'])

    # TODO: Save the weights of the model in .ckpt format
    model.save_weights("./models/monet/mobilenet_tf.ckpt")

