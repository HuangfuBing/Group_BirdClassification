import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2 as cv

#参看文章：https://blog.csdn.net/qq_36926037/article/details/106112072
def image_process(image_path):
    tf.compat.v1.disable_eager_execution()
    image=cv.imread(image_path)
    image=cv.resize(image,(224,224))
    image=tf.convert_to_tensor(image)
    image=tf.reshape(image,[1,224,224,3])
    image = tf.cast(image, dtype=np.float32)
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    image = image.eval(session=sess)  # 转化为numpy数组
    return image

image_path="F:/test/2.jpg"
image=image_process(image_path)
print(image)
interpreter = tf.lite.Interpreter("./model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details)
print(output_details)
interpreter.set_tensor(input_details[0]['index'], image)#传入的数据必须为ndarray类型
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
w = np.argmax(output_data)#值最大的位置
print(w)#第279类

