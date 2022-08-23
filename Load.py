import tensorflow as tf

converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file('vit_l16.h5')
tflite_model = converter.convert()
open("vit_l16.tflite", "wb").write(tflite_model)