from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Conv2D,Activation,MaxPooling2D


def VGG():
    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding='same', input_shape=(32, 32, 3), name='block1_conv1'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    model.add(Conv2D(128, (3, 3), padding='same', name='block2_conv1'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    model.add(Conv2D(256, (3, 3), padding='same', name='block3_conv1'))
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same', name='block3_conv2'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

    model.add(Conv2D(512, (3, 3), padding='same', name='block4_conv1'))
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same', name='block4_conv2'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

    model.add(Conv2D(512, (3, 3), padding='same', name='block5_conv1'))
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same', name='block5_conv2'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

    model.add(Flatten())

    model.add(Dense(512, name='fc1'))
    model.add(Activation('relu'))

    model.add(Dense(512, name='fc2'))
    model.add(Activation('relu'))

    model.add(Dense(10, name='fc3'))
    model.add(Activation('softmax'))
    return model

