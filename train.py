import os
import numpy as np
import tensorflow as tf
import cv2
import itertools
import time
from Image import *
from  tensorflow.keras import backend as K
from create_pascal_dataset import  PascalVOCDataset
from model import VGGBase,MyModel,SSD,MultiBoxLoss
from ssd_utils import resize_image_bbox

isprint = True

'''
'''

def run_train(dataset, num_epochs=1):
    start_time = time.perf_counter()

    #model = VGGBase()
    model = SSD(n_classes=20)
    tf.print('prios_cxcy->',model.priors_cxcy)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy)

    for _ in tf.data.Dataset.range(num_epochs):
        for idx,(images,boxes,labels) in enumerate(dataset): # (batch_size (N), 300, 300, 3)

            images = np.array(images)
            labels = np.array(labels)
            boxes = np.array(list(boxes))

            if isprint: tf.print(type(images), type(labels),images.shape,labels.shape)
            predicted_locs, predicted_socres = model(images)# (N, 8732, 4), (N, 8732, n_classes)

            #if isprint:
            #        tf.print("============================================================")
            #        tf.print('predicted_locs->',predicted_locs.shape)
            ##        tf.print('predicted_socres->',predicted_socres.shape)
             #       tf.print('image ->',images.shape)
             #       tf.print('boxes->',boxes)
             #       tf.print('labels->',labels.shape)
             #       tf.print('labels->',labels)
            #find_jaccard_overlap  model.forward
            loss = criterion(predicted_locs,predicted_socres,boxes,labels)
            pass
            if idx ==0: break
        pass
    tf.print("실행 시간:", time.perf_counter() - start_time)
def train():
    if isprint:print(tf.__version__)
    batch_size= 256
    images,boxes,labels,difficulties,new_boxes= PascalVOCDataset()
    new_boxes = list(new_boxes)
    boxes = tf.ragged.constant(boxes)
    labels = tf.ragged.constant(labels)
    new_boxes = tf.ragged.constant(new_boxes)
    dataset = tf.data.Dataset.from_tensor_slices((images,new_boxes,labels))
    #run_train(dataset.map(resize_image_bbox, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(1).prefetch(tf.data.experimental.AUTOTUNE))
    run_train(dataset.map(resize_image_bbox, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(1).prefetch(tf.data.experimental.AUTOTUNE))


def main():
    train()
if __name__ =='__main__':
    main()
