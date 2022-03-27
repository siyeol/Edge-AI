import tensorflow as tf
from vgg11_tf import VGG

new_model = VGG()
new_model.load_weights("vgg11_tf.ckpt")

converter = tf.lite.TFLiteConverter.from_keras_model(new_model)
tflite_model = converter.convert()

with open('vgg11.tflite', 'wb') as f:
    f.write(tflite_model)