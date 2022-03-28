import tensorflow as tf
from models.vgg11_tf import VGG

new_model = VGG()
new_model.load_weights("./models/vgg11/vgg11_tf.ckpt")

converter = tf.lite.TFLiteConverter.from_keras_model(new_model)
tflite_model = converter.convert()

with open("./models/vgg11/VGG11.tflite", 'wb') as f:
    f.write(tflite_model)