import tensorflow as tf
from tensorflow.keras import backend, optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GaussianNoise, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from vit_keras import vit
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.python.client import device_lib
TRAIN_PATH = "../Dataset/train"
VALID_PATH = "../Dataset/valid"
TEST_PATH = "../Dataset/test"
#分配显存
config = ConfigProto()
#先分配较少的显存，按需增加
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

#设计动态分配显存的函数
def allocate_gpu_memory(gpu_number=0):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if physical_devices:
        try :
            print("Found {} GPU(s)".format(len(physical_devices)))
            #设置为可见，tf只在可见的设备上分配工作
            #见https://andy6804tw.github.io/2021/08/18/tensorflow-gpu-memory-growth/
            tf.config.set_visible_devices(physical_devices[gpu_number],'GPU')
            tf.config.experimental.set_memory_growth(physical_devices[gpu_number],True)
            print("#{} GPU(s) is allocated".format(gpu_number))
        except RuntimeError as e:
            print(e)
    else:
        print("一块GPU都没有还想跑机器学习？")

allocate_gpu_memory()
print(device_lib.list_local_devices())

#用ImageDataGenerator进行数据处理，生成Image Tensor
train_datagen = ImageDataGenerator(
    rescale = 1/255,
    horizontal_flip=True,#随机水平翻转
    rotation_range=15,#随机旋转度数范围
    zoom_range=0.1,#随机缩放
)
valid_datagen = ImageDataGenerator(
    rescale=1/255,
)
test_datagen = ImageDataGenerator(
    rescale=1/255,
)
#读入数据
train_generator = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=(224,224),
    batch_size=16,
    color_mode='rgb',
    class_mode='sparse',#展开成一维
    shuffle=True,
)
validation_generator = valid_datagen.flow_from_directory(
    VALID_PATH,
    target_size=(224, 224),
    batch_size=16,
    color_mode='rgb',
    class_mode='sparse',
)
test_generator = test_datagen.flow_from_directory(
    TEST_PATH,
    target_size=(224, 224),
    batch_size=16,
    color_mode='rgb',
    class_mode='sparse',
)

#看一下图片
def plotImages(images_arr):
    fig,axes = plt.subplots(1,5,figsize=(20,20))#（1，5）
    axes=axes.flatten()
    for img,ax in zip(images_arr,axes):#打包元组
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

augmented_images=[train_generator[0][0][0] for i in range(5)]
plotImages(augmented_images)
backend.clear_session()
#配置模型
#ViT模型思考角度和卷积神经网络略有差异
#https://zhuanlan.zhihu.com/p/491848581
#https://arxiv.org/pdf/2010.11929.pdf
vit_model = vit_keras.vit.vit_l32(
    image_size=224,
    pretrained=True,#使用预训练的模型
    include_top=False,
    pretrained_top=False,#如果不如此指定会出现错误
)

print(len(vit_model.layers))
print(vit_model.layers)

#学习率预热
#https://blog.csdn.net/sinat_36618660/article/details/99650804
def scheduler(epoch:int,lr:float) -> float:
    if epoch!= 0 and epoch % 7 == 0:
        return lr*0.1
    else :
        return lr

lr_scheduler_callback = LearningRateScheduler(scheduler)
finetune_at = 28

#数据量少，数据相似度低：冻结预训练模型的初始层
for layer in vit_model.layers[:finetune_at-1]:
    layer.trainable = False

num_classes = len(validation_generator.class_indices)

#在ViT网络的最前端插入一个线性层并输出softmax
noise = GaussianNoise(0.01,input_shape=(224,224,3))
head = Dense(num_classes,activation="softmax")
#序贯模型
model = Sequential()
model.add(noise)
model.add(vit_model)
model.add(head)
#定义优化器、损失函数、准确率
#核心训练部分见https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile
model.compile(optimizer=op.optimizers.adam_v2.Adam(),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

#开始训练
#history attribute is a record of training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values
history = model.fit(
    train_generator,
    epochs=20,#default
    validation_data=validation_generator,
    verbose=1,#default
    shuffle=True,
    callbacks=[#在每个epoch的开始或结束执行特定的操作
        EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True),
        lr_scheduler_callback,
    ]
)
#获得损失值和评估值的字典来生成图表,便于控制调参
history_dict = history.history
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
acc_values = history_dict["accuracy"]
val_acc_values = history_dict["val_accuracy"]
epochs = range(1, len(history_dict["accuracy"]) + 1)
#可视化操作
plt.plot(epochs, loss_values, "bo", label="train")
plt.plot(epochs, val_loss_values, "b", label="valid")
plt.title("Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.plot(epochs, acc_values, "bo", label="train")
plt.plot(epochs, val_acc_values, "b", label="valid")
plt.title("Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

#保存模型
model.save('BirdClassification.h5')
