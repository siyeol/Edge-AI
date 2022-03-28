import tensorflow as tf
from models.vgg16_tf import VGG

new_model = VGG()
new_model.load_weights("./models/vgg16/vgg16_tf.ckpt")

converter = tf.lite.TFLiteConverter.from_keras_model(new_model)
tflite_model = converter.convert()

with open("./models/vgg16/VGG16.tflite", 'wb') as f:
    f.write(tflite_model)