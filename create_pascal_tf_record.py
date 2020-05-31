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
import logging
import os
from object_detection_utils import dataset_utils
from object_detection_utils import label_map_util

from lxml import etree
import tqdm

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

years = YEARS
data_dir = '/media/jake/mark-4tb3/input/datasets/pascal/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/'
annotations_dir = '/media/jake/mark-4tb3/input/datasets/pascal/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/Annotations/'
label_map_path = './data/pascal_label_map.pbtxt'
img_path = '/media/jake/mark-4tb3/input/datasets/pascal/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/'
#output_path = ''
def dic_to_tf_example(data,
                      dataset_directory,
                      label_map_dict,
                      ignore_difficult_instances=False,
                      image_subdirectory='JPEGImages'
                      ):

    img_path = os.path.join(data['folder', image_subdirectory,data['filename']])
    print('img_path->',img_path)


def main():
    #logging.info(tf.__version__)
    #writer = tf.python_io.TFRecordWriter(output_path)

    examples_path = os.path.join(data_dir,'ImageSets','Main','aeroplane_'+ 'train'+'.txt')
    example_list = dataset_utils.read_examples_list(examples_path)
    #label_map_dict = label_map_util.get_label_map_dict(label_map_path)
    for idx, example in tqdm(enumerate(example_list)):

        #if idx % 100 ==0:
            #logging.info('On image %d of %d', idx, len(example_list))
        path = os.path.join(annotations_dir, example + '.xml')
        logging.info(path)
        print(path)
        with tf.gfile.GFile(path,'r') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = dataset_utils.recursive_parse_xml_to_dict(xml)['annotation']
        #dic_to_tf_example(data,data_dir,label_map_dict,ignore_difficult_instances=False)

        img = os.path.join(img_path,data['filename'])
        object = data['object'][0]['name']
        bndbox = data['object'][0]['bndbox']

        print(img)
        print(object,bndbox)

        break
if __name__ == '__main__':
  main()