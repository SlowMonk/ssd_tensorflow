
from create_pascal_dataset import  PascalVOCDataset
import os
import numpy as np
import tensorflow as tf
import cv2
import itertools
import time
from  tensorflow.keras import backend as K
'''
'''
isprint = True



def resize_image(image,bbox):

    img = tf.io.read_file(image)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32)
    newSize = (300, 300)

    img = tf.image.resize(img,newSize)

    return img,bbox
def run_train(dataset, num_epochs=2):
    start_time = time.perf_counter()

    for _ in tf.data.Dataset.range(num_epochs):
        #for _,__ in dataset:
        #    print(_,__)
        #    break
        #    pass
        #for _ in dataset:
        #    print(_)
        pass
    tf.print("실행 시간:", time.perf_counter() - start_time)

def train():
    if isprint:print(tf.__version__)
    batch_size= 256

    #dataset test0
    images,boxes,labels,difficulties= PascalVOCDataset()
    print('boxes->',boxes)
    boxes = tf.ragged.constant(boxes)
    print('boxes after->', boxes)


    print(type(images),type(boxes))

    dataset = tf.data.Dataset.from_tensor_slices((images,boxes))
    #function_to_map = lambda x,y: resize_image(x,y)
    run_train(dataset.map(resize_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(2).prefetch(tf.data.experimental.AUTOTUNE))

#image, bbox = next(iter(dataset))
def main():
    train()

if __name__ =='__main__':
    main()
