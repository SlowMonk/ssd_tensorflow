import tensorflow as tf

def resize_image(image,bbox):

    img = tf.io.read_file(image)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32)
    newSize = (300, 300)

    img = tf.image.resize(img,newSize)

    return img,bbox