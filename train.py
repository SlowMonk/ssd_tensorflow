
import os
import numpy as np
import tensorflow as tf
import cv2
import itertools
import time
from Image import *
from  tensorflow.keras import backend as K
from create_pascal_dataset import  PascalVOCDataset
from model import VGGBase,MyModel

isprint = True

'''
'''
def run_train2(dataset,num_epochs = 2):
    start_time = time.perf_counter()

    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # 채널 차원을 추가합니다.
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]
    model = MyModel()
    print('x_train',type(x_train), 'y_train->',type(y_train))
    for _ in tf.data.Dataset.range(num_epochs):
        for image, target in dataset:  # (batch_size (N), 300, 300, 3)
            # print(type(image), type(x_train))
            predicted_locs = model(image)  # (N, 8732, 4), (N, 8732, n_classes)
            print(predicted_locs)
            pass
            break
        pass
    tf.print("실행 시간:", time.perf_counter() - start_time)


def run_train(dataset, num_epochs=2):
    start_time = time.perf_counter()

    model = VGGBase()

    for _ in tf.data.Dataset.range(num_epochs):
        for image,target in dataset: # (batch_size (N), 300, 300, 3)
            image = np.array(image)
            target = np.array(target)
            print(type(image), type(target),image.shape,target.shape)
            predicted_locs, predicted_socres = model(image)# (N, 8732, 4), (N, 8732, n_classes)
            print(predicted_locs,predicted_socres)
            pass
            break
        pass
    tf.print("실행 시간:", time.perf_counter() - start_time)
def train():
    if isprint:print(tf.__version__)
    batch_size= 256

    #dataset test0
    images,boxes,labels,difficulties= PascalVOCDataset()
    boxes = tf.ragged.constant(boxes)
    dataset = tf.data.Dataset.from_tensor_slices((images,boxes))
    run_train(dataset.map(resize_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(1).prefetch(tf.data.experimental.AUTOTUNE))

def main():
    train()
if __name__ =='__main__':
    main()
