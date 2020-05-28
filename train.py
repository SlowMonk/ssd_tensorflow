
from create_pascal_dataset import  PascalVOCDataset
import os
import numpy as np
import tensorflow as tf
import cv2
import itertools
'''
ds = tf.data.Dataset.from_tensorslice
ds = ds.map(func)
ds = ds.batch()
ds = ds.prefetch()
/media/jake/mark-4tb3/input/datasets/pascal/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/2008_000259.jpg
/media/jake/mark-4tb3/input/datasets/pascal/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/
'''
isprint = True
def func(images):
    images = np.array(images)
    newSize = [300,300]
    if isprint:print('image_A->',images[0].decode("utf-8"))
    #print('boxes->',boxes)


    image = cv2.imread(images[0].decode("utf-8"))
    if isprint:print(image.shape)
    image = np.array(image)
    if isprint:print('A')
    scale_x = newSize[0] / image.shape[1]
    scale_y = newSize[1] / image.shape[0]
    if isprint:print('B')
    image = cv2.resize(image, (newSize[0], newSize[1]))
    if isprint:print('C',image.shape)
    return tf.convert_to_tensor(image,dtype=tf.uint8)

def train():
    if isprint:print(tf.__version__)

    images,boxes,labels,difficulties= PascalVOCDataset()
    boxes = tf.ragged.constant(boxes)
    dataset = tf.data.Dataset.from_tensor_slices((images,boxes)).shuffle(100).batch(1)
    dataset = dataset.map(lambda image,box: tf.py_function(func=func, inp = [image],Tout=tf.string))
    for  i,b in dataset:
        print(i,b)
        break

def main():
    train()
if __name__ =='__main__':
    main()
