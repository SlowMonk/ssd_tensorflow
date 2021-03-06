'''

# From tensorflow/models/research/

$ wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

$ tar -xvf VOCtrainval_11-May-2012.tar

$ python object_detection/dataset_tools/create_pascal_tf_record.py \

    --label_map_path=object_detection/data/pascal_label_map.pbtxt \

    --data_dir=VOCdevkit --year=VOC2012 --set=train \

    --output_path=pascal_train.record

http://solarisailab.com/archives/2603  텐서플로우 tfrecord 파일을 이용해서 데이터 읽고 쓰기
'''

#import tensorflow as tf
#import tensorflow.compat.v1 as tf
from object_detection_utils import dataset_utils
from object_detection_utils import label_map_util
import xml.etree.ElementTree as ET
import torchvision.transforms.functional as FT
from lxml import etree
import json
import torch
import random
import logging
import os
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import tensorflow as tf


#from object_detection.utils import dataset_util
#from object_detection.utils import label_map_util
#flags = tf.compat.v1.flags
#flags.DEFINE_string('label_map_path','')
#flags.DEFINE_string('data_dir','VOCdevkit')
#flags.DEFINE_string('output_path','pascal_train.record','')
#flags.DEFINE_string('year','VOC2012','')
#flags.DEFINE_string('set','train','')
#flags.DEFINE_string('data_dir','/media/jake/mark-4tb3/input/datasets/pascal/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/','')
#flags.DEFINE_string('annotations_dir','/media/jake/mark-4tb3/input/datasets/pascal/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/Annotations/','')
#FLAGS = flags.FLAGS

SETS = ['train', 'val', 'trainval', 'test']
YEARS = ['VOC2007', 'VOC2012', 'merged']

#TODO transform to config
years = YEARS
data_dir = '/media/jake/mark-4tb3/input/datasets/pascal/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/'
annotations_dir = '/media/jake/mark-4tb3/input/datasets/pascal/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/Annotations/'
label_map_path = './data/pascal_label_map.pbtxt'
img_path = '/media/jake/mark-4tb3/input/datasets/pascal/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/'
#output_path = ''

# Label map
voc_labels = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
label_map = {k: v + 1 for v, k in enumerate(voc_labels)}
ifprint = False
def parse_annotation(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    boxes = list()
    labels = list()
    difficulties = list()

    filename = img_path + root.find('filename').text.lower()

    for object in root.iter('object'):

        difficult = int(object.find('difficult').text == '1')

        label = object.find('name').text.lower().strip()
        if label not in label_map:
            continue

        bbox = object.find('bndbox')
        xmin = int(bbox.find('xmin').text) - 1
        ymin = int(bbox.find('ymin').text) - 1
        xmax = int(bbox.find('xmax').text) - 1
        ymax = int(bbox.find('ymax').text) - 1

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label_map[label])
        difficulties.append(difficult)

    return {'filename': filename,'boxes': boxes, 'labels': labels, 'difficulties': difficulties}

def PascalVOCDataset():

    examples_path = os.path.join(data_dir,'ImageSets','Main','aeroplane_'+ 'train'+'.txt')
    example_list = dataset_utils.read_examples_list(examples_path)
    newSize = [300,300]
    images = []
    labels = []
    boxes = []
    difficulties = []
    new_boxes= []
    for idx, example in tqdm(enumerate(example_list)):

        #if idx % 100 ==0:
            #logging.info('On image %d of %d', idx, len(example_list))

        #annotation
        image_path = os.path.join(annotations_dir, example + '.xml')
        anno = parse_annotation(image_path)
        if ifprint:print(anno)

        image = anno['filename']
        # image read Image.open, cv2
        #image = np.array(Image.open(anno['filename']))
        # #print('img_shape->',img_origin.shape)

        #boxes
        box = anno['boxes']
        #img_origin = cv2.imread(anno['filename'])
        #w,h = img_origin.shape[0],img_origin.shape[1]
        #print('boxes->',box)
        w,h = 300,300
        old_dims = [w,h,w,h]
        #old_dims = tf.expand_dims(old_dims,axis=0,name=None)
        #print('old_dims->',old_dims)
        #Tensor to numpy array
        #new_box = box / old_dims
        new_box = []
        for b in box:
            temp = [x / y for x,y in zip(b,old_dims)]
            new_box.append(temp)


        #box = tf.ragged.constant(box)


        #scale_x = newSize[0] / image.shape[1]
        #scale_y = newSize[1] / image.shape[0]

        #image = cv2.resize(image, (newSize[0], newSize[1]))

        #print('box->',box,type(box))

        #print('new_box->',box)
        label = anno['labels']
        difficulty = anno['difficulties']

        images.append(image)
        boxes.append(box)
        labels.append(label)
        difficulties.append(difficulties)
        new_boxes.append(new_box)
        if ifprint:
            print(np.array(images).shape)
            print(np.array(boxes).shape)
            print(np.array(labels).shape)
            print(np.array(difficulties).shape)
        #break

    return images,boxes,labels,difficulties,new_boxes
