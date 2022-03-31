import tensorflow as tf
from models.mobilenet_tf import MobileNetv1

new_model = MobileNetv1()
new_model.load_weights("./models/monet/mobilenet_tf.ckpt")

converter = tf.lite.TFLiteConverter.from_keras_model(new_model)
tflite_model = converter.convert()

with open("./models/monet/Mobilenetv1.tflite", 'wb') as f:
    f.write(tflite_model)