import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense,Flatten,Conv2D
from tensorflow.keras.activations import relu
from tensorflow.keras import datasets, layers,optimizers,models
from ssd_utils import *
#vgg16

class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.conv1 = Conv2D(32, 3, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(10, activation='softmax')

  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)

class  VGGBase(Model):
    def __init__(self):
        super(VGGBase,self).__init__()
        self.padding_1 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))  # put this before your conv layer
        self.conv1_1 = tf.keras.layers.Conv2D(3, kernel_size=3,padding='same',strides=1, activation='relu')
        self.conv1_2 = tf.keras.layers.Conv2D(64, kernel_size=3, padding='same',strides=1,activation='relu')
        self.pool1 = tf.keras.layers.MaxPool2D(2,2)

        self.conv2_1  =  tf.keras.layers.Conv2D(128, kernel_size=3, padding='same',strides= 1,activation='relu')
        self.conv2_2 = tf.keras.layers.Conv2D(128, kernel_size=3,padding='same',strides= 1,activation='relu')
        self.pool2 = tf.keras.layers.MaxPool2D(2,2)

        self.conv3_1 =  tf.keras.layers.Conv2D(256, kernel_size=3, padding='same',strides= 1,activation='relu')
        self.conv3_2 =  tf.keras.layers.Conv2D(256, kernel_size=3, padding='same',strides= 1,activation='relu')
        self.conv3_3 =  tf.keras.layers.Conv2D(256, kernel_size=3, padding='same',strides= 1,activation='relu')
        self.pool3 = tf.keras.layers.MaxPool2D(2,2)

        self.conv4_1 = tf.keras.layers.Conv2D(512, kernel_size=3, padding='same', strides=1,activation='relu')
        self.conv4_2 = tf.keras.layers.Conv2D(512, kernel_size=3, padding='same', strides=1,activation='relu')
        self.conv4_3 = tf.keras.layers.Conv2D(512, kernel_size=3, padding='same', strides=1,activation='relu')
        self.pool4 = tf.keras.layers.MaxPool2D(2, 2)

        self.conv5_1 = tf.keras.layers.Conv2D(512, kernel_size=3, padding='same', strides=1,activation='relu')
        self.conv5_2 = tf.keras.layers.Conv2D(512, kernel_size=3, padding='same', strides=1,activation='relu')
        self.conv5_3 = tf.keras.layers.Conv2D(512, kernel_size=3, padding='same', strides=1,activation='relu')
        self.pool5 = tf.keras.layers.MaxPool2D(2, 2)

        self.padding6 = tf.keras.layers.ZeroPadding2D(padding=(6, 6))  # put this before your conv layer
        self.conv6 = tf.keras.layers.Conv2D(1024, kernel_size=3, padding='same',dilation_rate=6,activation='relu') # atrous convolution
        self.conv7 = tf.keras.layers.Conv2D(1024, kernel_size=1,activation='relu')
        #self.load_weights()
    def call(self,x):
        x = self.padding_1(x)
        x = self.conv1_1(x)# (N, 64, 300, 300)
        x = self.conv1_2(x)# (N, 64, 300, 300)
        x = self.pool1(x) # (N, 64, 150, 150)

        x = self.conv2_1(x) # (N, 128, 150, 150)
        x = self.conv2_2(x) # (N, 128, 150, 150)
        x = self.pool2(x)# (N, 128, 75, 75)

        x = self.conv3_1(x) # (N, 256, 75, 75)
        x = self.conv3_2(x)# (N, 256, 75, 75)
        x = self.conv3_3(x)# (N, 256, 75, 75)
        x = self.pool3(x) #(N, 256, 38, 38), it would have been 37 if not for ceil_mode = True

        x = self.conv4_1(x)# (N, 512, 38, 38)
        x = self.conv4_2(x)# (N, 512, 38, 38)
        x = self.conv4_3(x)# (N, 512, 38, 38)
        conv4_3_feats = x# (N, 512, 38, 38)
        x = self.pool4(x)# (N, 512, 19, 19)

        x = self.conv5_1(x) # (N, 512, 19, 19)
        x = self.conv5_2(x) # (N, 512, 19, 19)
        x = self.conv5_3(x) # (N, 512, 19, 19)
        x = self.pool5(x) # (N, 512, 19, 19), pool5 does not reduce dimensions

        x = self.padding6(x)
        x = self.conv6(x) # (N, 1024, 19, 19)
        x = self.conv7(x) # (N, 1024, 19, 19)
        conv7_feats = x

        return conv4_3_feats, conv7_feats

class AixiliaryConvolutions(Model):
    def __init__(self):
        super(AixiliaryConvolutions,self).__init__()
        self.conv8_1 = tf.keras.layers.Conv2D(1024,kernel_size=3,padding = 'same',strides = 1, activation='relu')
        self.padding_8_1 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))  # put this before your conv layer
        self.conv8_2 = tf.keras.layers.Conv2D(256,kernel_size=3,padding='same',strides = 1, activation='relu')

        self.conv9_1 = tf.keras.layers.Conv2D(512,kernel_size= 1, padding='same')
        self.padding_9_1 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))  # put this before your conv layer
        self.conv9_2 = tf.keras.layers.Conv2D(128,kernel_size=3,padding='same')

        self.conv10_1 = Conv2D(256, kernel_size=1, padding='same')
        self.conv10_2 = Conv2D(128, kernel_size=3, padding='same')

        self.conv11_1 = Conv2D(256,1, padding='same')
        self.conv11_2 = Conv2D(128,kernel_size=3,padding='same')

        self.init_conv2d()

    def init_conv2d(self):
        pass

    def call(self,conv7_feats):
        out = self.conv8_1(conv7_feats)
        out = self.conv8_2(out)
        conv8_2_feats = out

        out = self.conv9_1(out)
        out = self.conv9_2(out)
        conv9_2_feats = out

        out = self.conv10_1(out)
        out = self.conv10_2(out)
        conv10_2_feats = out

        out = self.conv11_1(out)
        conv11_2_feats =  self.conv11_2(out)

        return conv8_2_feats, conv9_2_feats, conv10_2_feats,conv11_2_feats

class PredictionConvolutions(Model):
    def __init__(self,n_classes):
        super(PredictionConvolutions,self).__init__()
        self.n_classes = n_classes
        self.locs
        self.classes_scores
    def init_conv2d(self):
        pass
    def call(self):
        return self.locs, self.classes_scores

class SSD(Model):
    def __init__(self,n_classes):
        super(SSD,self).__init__()
        self.n_classes = n_classes
        self.base = VGGBase()
        self.aux_convs = AixiliaryConvolutions()

        self.pred_convs = PredictionConvolutions(n_classes)
        self.priors_cxcy = create_prior_boxes()

    def call(self,image):
        conv4_3_feats, conv7_feats = self.base(image)  # (N, 512, 38, 38), (N, 1024, 19, 19)

        # Run auxiliary convolutions (higher level feature map generators)
        conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats = \
            self.aux_convs(conv7_feats)

        return conv4_3_feats,conv7_feats

    def detect_object(self,predicted_locs, predicted_scores, min_score,max_overlap, top_k):
        pass

class MultiBoxLoss(Model):
    def __init__(self):
        super(MultiBoxLoss,self).__init__()
        self.conf_loss
        self.alpha
        self.loc_loss
    pass