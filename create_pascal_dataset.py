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
import tensorflow.compat.v1 as tf
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

def main():

    examples_path = os.path.join(data_dir,'ImageSets','Main','aeroplane_'+ 'train'+'.txt')
    example_list = dataset_utils.read_examples_list(examples_path)
    for idx, example in enumerate(example_list):

        if idx % 100 ==0:
            logging.info('On image %d of %d', idx, len(example_list))
        path = os.path.join(annotations_dir, example + '.xml')
        anno = parse_annotation(path)
        image = np.array(Image.open(anno['filename']))
        print(anno)
        print(image)

        break
if __name__ == '__main__':
  main()


  '''
#logging.info(path)
#print(path)
#with tf.gfile.GFile(path,'r') as fid:
#    xml_str = fid.read()
#xml = etree.fromstring(xml_str)
#data = dataset_utils.recursive_parse_xml_to_dict(xml)['annotation']

#img = os.path.join(img_path,data['filename'])
#object = data['object'][0]['name']
#bndbox = data['object'][0]['bndbox']

#print(data,img)
#print(object,bndbox)
  
  
  '''