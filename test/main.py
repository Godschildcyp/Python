# TensorFlow and tf.keras
import tensorflow as tf
import keras
#import cv2
from keras.models import load_model
# Helper libraries
import numpy as np
import random
# 读取数据
from tensorflow.examples.tutorials.mnist import input_data

model_test = []
########################################################################################################################
# 训练CNN模型
########################################################################################################################
def trainFashionMNISTCNNModel():
    MNIST_data_folder = "dataSet"
    fashion_mnist = input_data.read_data_sets(MNIST_data_folder, one_hot=True)

    # 划分训练集与测试集
    # (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    temp_train_images, temp_train_labels = fashion_mnist.train.images, fashion_mnist.train.labels
    temp_test_images, temp_test_labels = fashion_mnist.test.images, fashion_mnist.test.labels

    # plt.figure()
    # for i in range(10):
    #     print(temp_test_images[i])
    #     plt.subplot(2, 5, i + 1)
    #     plt.imshow(np.array(temp_test_images[i]).reshape(28, 28))
    #     plt.pause(0.000001)
    # plt.show()

    # label转换为整形label
    train_labels = np.zeros((temp_train_labels.shape[0], 1))
    test_labels = np.zeros((temp_test_labels.shape[0], 1))

    for i in range(temp_train_labels.shape[0]):
        for j in range(temp_train_labels.shape[1]):
            if temp_train_labels[i][j] == 1:
                train_labels[i, 0] = j
                break

    for i in range(temp_test_labels.shape[0]):
        for j in range(temp_test_labels.shape[1]):
            if temp_test_labels[i][j] == 1:
                test_labels[i, 0] = j
                break

    train_images = temp_train_images.reshape(-1, 28, 28, 1).astype('float32')
    train_images = train_images[:, :, :, 0]
    test_images = temp_test_images.reshape(-1, 28, 28, 1).astype('float32')
    test_images_T = test_images[:, :, :, 0]

    # 数据处理归一化

    train_images = train_images / 255.0
    test_images = test_images_T / 255.0

    #  CNN建模

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    
    # 采用序列模式。
    # 第一层是Flatten，将28 * 28
    # 的像素值，压缩成一行(784, )
    # 第二层是Dense，全连接层。激活函数使用relu
    # 第三层还是Dense，因为是多分类问题，激活函数使用softmax
    # 在keras里，layers里包含所有的层类型。其中还包括，
    # 卷积层，Conv2D
    # Dropout层，Dropout
    # MaxPool1D, 最大池化层


    # 编译
    # 建模后就是编译。编译的参数主要是：
    # optimizer，优化方法，这里用Adam
    # loss，损失函数，这里用稀疏类别交叉熵（多类的对数损失），sparse是指稀疏矩阵
    # metrics，评估模型在训练和测试时的性能的指标
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # 拟合
    # 拟合是训练参数，拟合训练数据的过程。主要参数：
    # 训练数据
    # 训练标签
    # 训练次数
    model.fit(train_images, train_labels, epochs=5)

    # # save architecture
    # json_string = model.to_json()
    # open('./my_model_architecture.json', 'w').write(json_string)
    # # save weights
    # model.save_weights('./my_model_weights.h5')
    #
    # my_model = model_from_json(open('./my_model_architecture.json').read())
    # my_model.load_weights('./my_model_weights.h5')
    model.save('my_model.h5')

    model_test = load_model('my_model.h5')
    model_test.compile(optimizer=tf.train.AdamOptimizer(),
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])

    # 评估
    test_loss, test_acc = model_test.evaluate(test_images, test_labels)
    print('Test accuracy:', test_acc)

    return model_test,test_images_T,test_labels

########################################################################################################################
# 测试图片加高斯噪声
########################################################################################################################
def GaussianNoise(src,means,sigma):
    NoiseImg=src
    rows=NoiseImg.shape[0]
    cols=NoiseImg.shape[1]
    for i in range(rows):
        for j in range(cols):
            NoiseImg[i,j]=NoiseImg[i,j]+random.gauss(means,sigma)
            if  NoiseImg[i,j]< 0:
                 NoiseImg[i,j]=0
            elif  NoiseImg[i,j]>255:
                 NoiseImg[i,j]=255
    return NoiseImg

########################################################################################################################
# AI test
########################################################################################################################
def aiTest(images,shape):
    generate_images = np.zeros(shape)
    # 对图片进行处理
    for i in range(shape[0]):
        temp = images[i,:,:]
        noiseImg = GaussianNoise(temp, 10, 5)
        generate_images[i,:,:] = noiseImg

    return generate_images


########################################################################################################################
# 主函数
########################################################################################################################
if __name__ == '__main__':
    # 训练CNN模型
    model_test, test_images_T, test_labels = trainFashionMNISTCNNModel()
    # 选取1000张测试数据
    test_images_T = test_images_T[0:1000,:]
    test_labels = test_labels[0:1000,:]
    # 处理图像进行黑盒测试
    generate_images = aiTest(test_images_T, test_images_T.shape)
    # 进行黑盒测试
    test_loss, test_acc = model_test.evaluate(generate_images, test_labels)
    print('AI test accuracy:', test_acc)
