import tensorflow as tf
import PIL
import matplotlib.pyplot as plt
from Image import *
import numpy as np
from PIL import Image
import torch


def resize_image_bbox(image,boxes,labels):
    #print(image)
    img = tf.io.read_file(image)
    img = tf.image.decode_jpeg(img, channels=3)

    #img2 = tf.io.read_file('/home/jake/Pictures/example.png/')
    #img2 = tf.image.decode_jpeg(img2,channels=3)
    #print('tf_shape->',img.shape,img2.shape)
    img = tf.cast(img, tf.float32)
    #print('cast->',img.shape)
    newSize = (300, 300)

    new_img = tf.image.resize(img,newSize)
    #print('boxes->',boxes)
    #print('labels->',labels)


    #old_dims = tf.TensorArray([300,300,300,300])
    return new_img,boxes,labels


def resize_image(image):

    img = tf.io.read_file(image)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32)
    newSize = (300, 300)

    img = tf.image.resize(img,newSize)
    print(img.shape)

    return img

def cxcy_to_xy(cxcy):
    """
    Convert bounding boxes from center-size coordinates (c_x, c_y, w, h) to boundary coordinates (x_min, y_min, x_max, y_max).

    :param cxcy: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    :return: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    """
    return tf.concat([cxcy[:, :2] - (cxcy[:, 2:] / 2),  # x_min, y_min
                      cxcy[:, :2] + (cxcy[:, 2:] / 2)], 1)  # x_max, y_max

    #detect(original_image, min_score=0.2, max_overlap=0.5, top_k=200).show()
#main()
def find_intersection(set_1, set_2):
    """
    Find the intersection of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """
    print("==========================================find_intersection==========================================")
    set_1 = tf.cast(set_1.to_tensor(),tf.float64)
    #set_2 = tf.cast(set_1,tf.float32)
    #print('set1->',set_1)
    #print('set_2->',set_2)
    set1 = tf.expand_dims(set_1,axis=-1)
    #set2 = tf.expand_dims(set_2,axis=-1)
    set2 = set_2
    #print('set1->',set1)
    #print('set2->',set2)
    #print('set_1[:,0]->',set1[:,0])
    #print('set_2[:,0->',set2[:,0])
    x_min = tf.math.maximum(set1[:, 0], set2[:, 0])
    y_min = tf.math.maximum(set1[:, 1], set2[:, 1])
    x_max = tf.math.minimum(set1[:, 2], set2[:, 2])
    y_max = tf.math.minimum(set1[:, 3], set2[:, 3])
    dx = tf.math.maximum(x_max - x_min, 0)
    dy = tf.math.maximum(y_max - y_min, 0)
    #print('dx->',dx)
    #print('dy->',dy)
    #print('dx * dy->',dx * dy)
    return dx * dy

    #lower_bounds = tf.(set_1[:,:2],set_2[:,:2])
    #return tf.sets.intersection(set_1,set_2)
def find_jaccard_overlap(set_1, set_2):
    """
    Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """
    print("========================================find_jaccard_overlap=========================================")

    # Find intersections
    intersection = find_intersection(set_1, set_2)  # (n1, n2)
    #print('intersection->', intersection)
    set_1 = tf.cast(set_1.to_tensor(), tf.float64)
    set_1 = tf.expand_dims(set_1,axis=-1)
    #print('set_1->',set_1)
    #print('set_1->',set_1[:,2] - set_1[:,0])
    #print('set_2->',set_2)
    #print('set_2->',set_2[:,3])
    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    #print('areas_set1,areas_set2->',areas_set_1,areas_set_2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    #union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)
    union = areas_set_1 + areas_set_2 - intersection

    return intersection / union  # (n1, n2)
    #return intersection