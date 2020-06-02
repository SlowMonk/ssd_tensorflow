import tensorflow as tf
import PIL
import matplotlib.pyplot as plt
from Image import *
import numpy as np
from PIL import Image

def create_prior_boxes():
    fmap_dims = {
        'conv4_3' : 38,
        'conv7' : 19,
        'conv8_2' : 10,
        'conv9_2' : 5,
        'conv10_2' : 3,
        'conv11_2' : 1}

    obj_scales = {
        'conv4_3' : 0.1,
        'conv7' : 0.2,
        'conv8_2' : 0.375,
        'conv9_2' : 0.55,
        'conv10_2' : 0.725,
        'conv11_2' : 0.9}

    aspect_ratios = {
        'conv4_3' : [1., 2.,0.5],
        'conv7' : [1.,2.,3.,0.5,0.333],
        'conv8_2' : [1., 2.,3.,0.5,0.333],
        'conv9_2' : [1., 2.,3.,0.5, .333],
        'conv10_2' : [1.,2.,0.5],
        'conv11_2': [1.,2.,0.5]
    }

    fmaps = list(fmap_dims.keys())
    prior_boxes = []

    for k, fmap in enumerate(fmaps):
        for i in range(fmap_dims[fmap]):
            for j in range(fmap_dims[fmap]):
                cx = (j + 0.5) / fmap_dims[fmap]
                cy = (i + 0.5) / fmap_dims[fmap]

                for ratio in aspect_ratios[fmap]:
                    prior_boxes.append([cx, cy, obj_scales[fmap] * tf.sqrt(ratio), obj_scales[fmap] / tf.sqrt(ratio)])

                    # For an aspect ratio of 1, use an additional prior whose scale is the geometric mean of the
                    # scale of the current feature map and the scale of the next feature map
                    if ratio == 1.:
                        try:
                            additional_scale = tf.sqrt(obj_scales[fmap] * obj_scales[fmaps[k + 1]])
                        # For the last feature map, there is no "next" feature map
                        except IndexError:
                            additional_scale = 1.
                        prior_boxes.append([cx, cy, additional_scale, additional_scale])

    #prior_boxes = torch.FloatTensor(prior_boxes).to(device)  # (8732, 4)
    #print('prior_boxes ->', prior_boxes)
    #prior_boxes.clamp_(0, 1)  # (8732, 4)
    #print('prior_boxes_B->', prior_boxes)
    return prior_boxes

# TODO 8732개 박스 찍기
def main():

    # PIL image
    prior_boxes=create_prior_boxes()
    print(prior_boxes[0])
    print(prior_boxes[30])
    print(len(prior_boxes))
    path = '/media/jake/mark-4tb3/input/datasets/pascal/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/'
    img_path = path + '2007_000039.jpg'
    print(img_path)
    original_image = PIL.Image.open(img_path, mode='r')
    original_image = original_image.convert('RGB')
    original_image = original_image.resize((300, 300), Image.ANTIALIAS)
    print('original_image->',np.array(original_image).shape)


    # tensorlfow image
    img = resize_image(img_path)
    print('img->',type(img))


    plt.imshow(original_image)
    plt.show()


    #detect(original_image, min_score=0.2, max_overlap=0.5, top_k=200).show()
#main()