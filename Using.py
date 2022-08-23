import keras.models
import tensorflow as tf
import numpy as np
import cv2 as cv
import csv as csv

#读csv文件
def Load_csv(csv_file_name="test.csv"):
    """
    从CSV文件中读取数据信息
    :param csv_file_name: CSV文件名
    :return: Data：二维数组
    """
    csv_reader = csv.reader(open(csv_file_name))
    Data=[]
    for row in csv_reader:
        Data.append(row)
    print("Read All!")
    return Data

#参看文章：https://blog.csdn.net/qq_36926037/article/details/106112072
def image_process(image_path):
    tf.compat.v1.disable_eager_execution()
    image=cv.imread(image_path)
    image=cv.resize(image,(112,112))
    image=tf.convert_to_tensor(image)
    image=tf.reshape(image,[1,112,112,3])
    image = tf.cast(image, dtype=np.float32)
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    image = image.eval(session=sess)  # 转化为numpy数组
    return image

data=Load_csv("./birds latin names.csv")
image_path="./148.jpg"
image=image_process(image_path)
print(image)
interpreter = tf.lite.Interpreter("./vit_l16.tflite")
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
print("鸟类俗名：",data[w+1][1]," ","鸟类学名:",data[w+1][2])#第100


