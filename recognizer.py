import tensorflow as tf
import numpy as np
from tensorflow.python.platform import gfile

BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
BOTTLENECK_TENSOR_SIZE = 2048
N_CLASSES = 17


class RecognizerA:
    def __init__(self, model_file="./inceptionV3/tensorflow_inception_graph.pb"):
        self.sess = tf.Session()
        f = gfile.GFile(model_file, 'rb')
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        self.sess.graph.as_default()
        self.bottleneck_tensor, self.jpeg_data_tensor = tf.import_graph_def(graph_def, name='',
                                                                            return_elements=[BOTTLENECK_TENSOR_NAME,
                                                                                             JPEG_DATA_TENSOR_NAME])
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def recognize(self, file_name):
        image_raw_data = gfile.GFile(file_name, 'rb').read()
        image_value = self.sess.run(self.bottleneck_tensor, {self.jpeg_data_tensor: image_raw_data})
        image_value = np.squeeze(image_value)
        return [image_value]


class RecognizerB:
    def __init__(self, model_file="./train_dir/model.pb"):
        self.processed_data = np.load("flower_processed_data.npy")
        self.flower_name_list = self.processed_data[6]
        self.sess = tf.Session()
        f = gfile.GFile(model_file, 'rb')
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        self.sess.graph.as_default()
        self.bottleneck_input, self.final_tensor = tf.import_graph_def(graph_def, name='',
                                                                       return_elements=["BottleneckInputPlaceholder:0",
                                                                                        "output/prob:0"])
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.prediction = tf.argmax(self.final_tensor, 1)

    def recognize(self, data):
        prediction_value = self.sess.run(self.prediction, feed_dict={self.bottleneck_input: data})
        return self.flower_name_list[prediction_value[0]]


class Recognizer:
    def __init__(self):
        self.ra = RecognizerA()
        self.rb = RecognizerB()

    def recognize(self, file_name):
        return self.rb.recognize(self.ra.recognize(file_name))

