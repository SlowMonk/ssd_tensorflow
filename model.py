import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense,Flatten,Conv2D
from tensorflow.keras.activations import relu
from tensorflow.keras import datasets, layers,optimizers,models
from ssd_utils import *
#vgg16

isprint= False
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
        self.conv1_1 = tf.keras.layers.Conv2D(3, kernel_size=3,strides=1, activation='relu')
        self.conv1_2 = tf.keras.layers.Conv2D(64, kernel_size=3,strides=1,activation='relu')
        self.pool1 = tf.keras.layers.MaxPool2D(2,2)

        self.conv2_1  =  tf.keras.layers.Conv2D(128, kernel_size=3,strides= 1,activation='relu')
        self.conv2_2 = tf.keras.layers.Conv2D(128, kernel_size=3,strides= 1,activation='relu')
        self.pool2 = tf.keras.layers.MaxPool2D(2,2)

        self.conv3_1 =  tf.keras.layers.Conv2D(256, kernel_size=3,strides= 1,activation='relu')
        self.conv3_2 =  tf.keras.layers.Conv2D(256, kernel_size=3,strides= 1,activation='relu')
        self.conv3_3 =  tf.keras.layers.Conv2D(256, kernel_size=3,strides= 1,activation='relu')#(N, 256, 75, 75)
        self.pool3 = tf.keras.layers.MaxPool2D(2,2,padding='same')#(N, 256, 38, 38)

        self.conv4_1 = tf.keras.layers.Conv2D(512, kernel_size=3, strides=1,activation='relu')
        self.conv4_2 = tf.keras.layers.Conv2D(512, kernel_size=3, strides=1,activation='relu')
        self.conv4_3 = tf.keras.layers.Conv2D(512, kernel_size=3, strides=1,activation='relu')# (N, 512, 38, 38)
        #conv4_3_feats = x# (N, 512, 38, 38)
        self.pool4 = tf.keras.layers.MaxPool2D(2, 2)

        self.conv5_1 = tf.keras.layers.Conv2D(512, kernel_size=3, strides=1,activation='relu')
        self.conv5_2 = tf.keras.layers.Conv2D(512, kernel_size=3, strides=1,activation='relu')
        self.conv5_3 = tf.keras.layers.Conv2D(512, kernel_size=3, strides=1,activation='relu')
        self.pool5 = tf.keras.layers.MaxPool2D(3, 1)

        self.padding6 = tf.keras.layers.ZeroPadding2D(padding=(6, 6))  # put this before your conv layer
        self.conv6 = tf.keras.layers.Conv2D(1024, kernel_size=3,dilation_rate=6,activation='relu') # atrous convolution
        self.conv7 = tf.keras.layers.Conv2D(1024, kernel_size=1,activation='relu')
        #self.load_weights()
    def call(self,x):
        x = self.padding_1(x)
        x = self.conv1_1(x)# (N, 64, 300, 300)
        x = self.padding_1(x)
        x = self.conv1_2(x)# (N, 64, 300, 300)
        x = self.pool1(x) # (N, 64, 150, 150)

        x = self.padding_1(x)
        x = self.conv2_1(x) # (N, 128, 150, 150)
        x = self.padding_1(x)
        x = self.conv2_2(x) # (N, 128, 150, 150)
        x = self.pool2(x)# (N, 128, 75, 75)

        x = self.padding_1(x)
        x = self.conv3_1(x) # (N, 256, 75, 75)
        x = self.padding_1(x)
        x = self.conv3_2(x)# (N, 256, 75, 75)
        x = self.padding_1(x)
        x = self.conv3_3(x)# (N, 256, 75, 75)# conv3_3 padding='same'
        x = self.pool3(x) #(N, 256, 38, 38), it would have been 37 if not for ceil_mode = True

        conv1_1 = x
        x = self.padding_1(x)
        x = self.conv4_1(x)# (N, 512, 38, 38)
        x = self.padding_1(x)
        x = self.conv4_2(x)# (N, 512, 38, 38)
        x = self.padding_1(x)
        x = self.conv4_3(x)# (N, 512, 38, 38)
        conv4_3_feats = x# (N, 512, 38, 38)
        x = self.pool4(x)# (N, 512, 19, 19)

        x = self.padding_1(x)
        x = self.conv5_1(x) # (N, 512, 19, 19)
        x = self.padding_1(x)
        x = self.conv5_2(x) # (N, 512, 19, 19)
        x = self.padding_1(x)
        x = self.conv5_3(x) # (N, 512, 19, 19)
        x = self.padding_1(x)
        x = self.pool5(x) # (N, 512, 19, 19), pool5 does not reduce dimensions

        x = self.padding6(x)
        x = self.conv6(x) # (N, 1024, 19, 19)
        x = self.conv7(x) # (N, 1024, 19, 19)
        conv7_feats = x

        return conv4_3_feats, conv7_feats , conv1_1

class AixiliaryConvolutions(Model):
    def __init__(self):
        super(AixiliaryConvolutions,self).__init__()
        self.conv8_1 = tf.keras.layers.Conv2D(256,kernel_size=1,strides = 1, activation='relu')
        self.padding_8_1 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))  # put this before your conv layer
        self.conv8_2 = tf.keras.layers.Conv2D(512,kernel_size=3,strides = 2, activation='relu')

        self.conv9_1 = tf.keras.layers.Conv2D(128,kernel_size= 1)
        self.padding_9_1 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))  # put this before your conv layer
        self.conv9_2 = tf.keras.layers.Conv2D(256,kernel_size=3,strides=2,activation='relu')

        self.conv10_1 = Conv2D(128, kernel_size=1,activation='relu')
        self.conv10_2 = Conv2D(256, kernel_size=3,activation='relu')

        self.conv11_1 = Conv2D(128,kernel_size=1,activation='relu')
        self.conv11_2 = Conv2D(256,kernel_size=3,activation='relu')

        self.init_conv2d()

    def init_conv2d(self):
        pass

    def call(self,conv7_feats):

        out = self.conv8_1(conv7_feats)
        out = self.padding_8_1(out)
        out = self.conv8_2(out)
        conv8_2_feats = out

        out = self.conv9_1(out)
        out = self.padding_9_1(out)
        out = self.conv9_2(out)
        conv9_2_feats = out

        out = self.conv10_1(out)
        out = self.conv10_2(out)
        conv10_2_feats = out

        out = self.conv11_1(out)
        out = self.conv11_2(out)
        conv11_2_feats =  out

        return conv8_2_feats, conv9_2_feats, conv10_2_feats,conv11_2_feats

class PredictionConvolutions(Model):
    def __init__(self,n_classes):
        super(PredictionConvolutions,self).__init__()
        self.n_classes = n_classes
        self.locs = 0
        self.classes_scores = 0

        # Number of prior-boxes we are considering per position in each feature map
        self.n_boxes = {'conv4_3': 4,
                   'conv7': 6,
                   'conv8_2': 6,
                   'conv9_2': 6,
                   'conv10_2': 4,
                   'conv11_2': 4}
        self.padding_1 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))  # put this before your conv layer
        # Localization prediction convolution
        self.loc_conv4_3 = Conv2D(self.n_boxes['conv4_3'] * 4, kernel_size=3)# 38 x 38 x 512
        self.loc_conv7 = Conv2D(self.n_boxes['conv7'] * 4, kernel_size=3)# 19 x 19 x 1024
        self.loc_conv8_2 = Conv2D(self.n_boxes['conv8_2'] * 4, kernel_size=3)# 10x10x256 -> 10 x 10 x 512
        self.loc_conv9_2 = Conv2D(self.n_boxes['conv9_2'] * 4,kernel_size=3)# 5 x 5 x 256
        self.loc_conv10_2 = Conv2D(self.n_boxes['conv10_2'] * 4,  kernel_size=3)# 3 x 3 x 256
        self.loc_conv11_2 = Conv2D( self.n_boxes['conv11_2'] * 4, kernel_size=3)# 1 x 1 x 256

        # Class prediction convolutions (predict classes in localization boxes)
        self.cl_conv4_3 = Conv2D(self.n_boxes['conv4_3'] * n_classes,  kernel_size=3)  # 38 x 38 x 512
        self.cl_conv7 = Conv2D(self.n_boxes['conv7'] * n_classes,  kernel_size=3)  # 19 x 19 x 1024
        self.cl_conv8_2 = Conv2D( self.n_boxes['conv8_2'] * n_classes, kernel_size=3)  # 10 x 10 x 512
        self.cl_conv9_2 = Conv2D(self.n_boxes['conv9_2'] * n_classes, kernel_size=3)  # 5 x 5 x 256
        self.cl_conv10_2 = Conv2D(self.n_boxes['conv10_2'] * n_classes,kernel_size=3)  # 3 x 3 x 256
        self.cl_conv11_2 = Conv2D(self.n_boxes['conv11_2'] * n_classes,  kernel_size=3)  # 1 x 1 x 256
    def init_conv2d(self):
        pass
        # Initialize Convolutions parameters
        #self.init_conv2d()
    def call(self,conv4_3_feats,conv7_feats,conv8_2_feats,conv9_2_feats,conv10_2_feats,conv11_2_feats,conv1_1):
        """
             Forward propagation.

             :param conv4_3_feats: conv4_3 feature map, a tensor of dimensions (N, 512, 38, 38)
             :param conv7_feats: conv7 feature map, a tensor of dimensions (N, 1024, 19, 19)
             :param conv8_2_feats: conv8_2 feature map, a tensor of dimensions (N, 512, 10, 10)
             :param conv9_2_feats: conv9_2 feature map, a tensor of dimensions (N, 256, 5, 5)
             :param conv10_2_feats: conv10_2 feature map, a tensor of dimensions (N, 256, 3, 3)
             :param conv11_2_feats: conv11_2 feature map, a tensor of dimensions (N, 256, 1, 1)
             :return: 8732 locations and class scores (i.e. w.r.t each prior box) for each image

              c_conv4_3 = c_conv4_3.permute(0,2,3,1).contiguous()
             """
        #batch_size= np.array(conv4_3_feats[0]).shape[-1]
        batch_size = np.array(conv4_3_feats).shape[0]
        # conv4_3_feats, conv7_feats
        # conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats

        # L convolution
        l_conv4_3 = self.padding_1(conv4_3_feats)
        l_conv4_3 = self.loc_conv4_3(l_conv4_3) # (N,16,38,38)
        l_conv4_3 = tf.reshape(l_conv4_3,[batch_size,-1,4]) # (N, 5776, 4), there are a total 5776 boxes on this feature map

        l_conv7 = self.padding_1(conv7_feats)# (N, 24, 19, 19)
        l_conv7 = self.loc_conv7(l_conv7)
        l_conv7 = tf.reshape(l_conv7,[batch_size,-1,4])# (N, 2166, 4), there are a total 2116 boxes on this feature map

        l_conv8_2 = self.padding_1(conv8_2_feats)
        l_conv8_2 = self.loc_conv8_2(l_conv8_2)
        l_conv8_2 = tf.reshape(l_conv8_2,[batch_size,-1,4])# (N, 600, 4) there are a total 600 boxes on this feature map

        l_conv9_2 = self.padding_1(conv9_2_feats)
        l_conv9_2 = self.loc_conv9_2(l_conv9_2)
        l_conv9_2 = tf.reshape(l_conv9_2,[batch_size,-1,4])# (N, 150, 4) there are a total 150 boxes on this feature map

        l_conv10_2 = self.padding_1(conv10_2_feats)
        l_conv10_2 = self.loc_conv10_2(l_conv10_2)
        l_conv10_2 = tf.reshape(l_conv10_2,[batch_size,-1,4])# (N, 36, 4) there are a total 36 boxes on this feature map

        l_conv11_2 = self.padding_1(conv11_2_feats)
        l_conv11_2 = self.loc_conv11_2(l_conv11_2)
        l_conv11_2 = tf.reshape(l_conv11_2,[batch_size,-1,4])  # (N, 4, 4) there are a total 4 boxes on this feature map

        # c convolution

        #print('conv8_2_feats->',conv8_2_feats.shape)
        c_conv4_3 = self.padding_1(conv4_3_feats)
        c_conv4_3 = self.cl_conv4_3(c_conv4_3)
        c_conv4_3 = tf.reshape(c_conv4_3,(batch_size,-1,self.n_classes))

        c_conv7 = self.padding_1(conv7_feats)  # (N, 24, 19, 19)
        c_conv7 = self.cl_conv7(c_conv7)
        c_conv7 = tf.reshape(c_conv7,
                             [batch_size, -1, self.n_classes])  # (N, 2166, 4), there are a total 2116 boxes on this feature map

        c_conv8_2 = self.padding_1(conv8_2_feats)
        c_conv8_2 = self.cl_conv8_2(c_conv8_2)
        c_conv8_2 = tf.reshape(c_conv8_2,
                               [batch_size, -1, self.n_classes])  # (N, 600, 4) there are a total 600 boxes on this feature map

        c_conv9_2 = self.padding_1(conv9_2_feats)
        c_conv9_2 = self.cl_conv9_2(c_conv9_2)
        c_conv9_2 = tf.reshape(c_conv9_2,
                               [batch_size, -1, self.n_classes])  # (N, 150, 4) there are a total 150 boxes on this feature map

        c_conv10_2 = self.padding_1(conv10_2_feats)
        c_conv10_2 = self.cl_conv10_2(c_conv10_2)
        c_conv10_2 = tf.reshape(c_conv10_2,
                                [batch_size, -1, self.n_classes])  # (N, 36, 4) there are a total 36 boxes on this feature map

        c_conv11_2 = self.padding_1(conv11_2_feats)
        c_conv11_2 = self.cl_conv11_2(c_conv11_2)
        c_conv11_2 = tf.reshape(c_conv11_2,
                                [batch_size, -1, self.n_classes])  # (N, 4, 4) there are a total 4 boxes on this feature map


        self.locs= tf.concat([l_conv4_3,l_conv7,l_conv8_2,l_conv9_2,l_conv10_2,l_conv11_2],axis=1)
        self.classes_scores = tf.concat([c_conv4_3,c_conv7,c_conv8_2,c_conv9_2,c_conv10_2,c_conv11_2],axis=1)

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
        conv4_3_feats, conv7_feats ,conv1_1 = self.base(image)  # (N, 512, 38, 38), (N, 1024, 19, 19)

        # Run auxiliary convolutions (higher level feature map generators)
        conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats = \
            self.aux_convs(conv7_feats)

        locs, classes_socres = self.pred_convs(conv4_3_feats,conv7_feats,conv8_2_feats,conv9_2_feats,conv10_2_feats,conv11_2_feats,conv1_1)
        return locs, classes_socres

    def detect_object(self,predicted_locs, predicted_scores, min_score,max_overlap, top_k):
        pass

class MultiBoxLoss(Model):
    def __init__(self):
        super(MultiBoxLoss,self).__init__()
        self.conf_loss
        self.alpha
        self.loc_loss
    pass