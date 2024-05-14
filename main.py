import os
import numpy as np
from functools import partial
import PIL.Image
import tensorflow as tf
import urllib.request
import zipfile

def main():
    ## Downloading Googles Pre-trained NN ##

    url = 'https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip'
    data_dir = '../data'
    model_name = os.path.split(url)[-1]
    local_zip_file = os.path.join(data_dir, model_name)
    if not os.path.exists(local_zip_file):
        # If it does not exist, then download from here
        model_url = urllib.request.urlopen(url)
        with open(local_zip_file, 'wb') as output:
            output.write(model_url.read())

        # Then Extract
        with zipfile.ZipFile(local_zip_file, 'r') as zip_ref:
            zip_ref.extractall(data_dir)




        model_fn = 'tensorflow_inception_graph.pb'

        ## Creating the session and Loading the Model using TF

        graph = tf.Graph()
        sess = tf.InteractiveSession(graph=graph)

        with tf.gfile.FatGFile(os.path.join(data_dir, model_fn), 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        t_input = tf.placeholder(np.float32, name= 'Input') # defining input tensor
        imagenet_mean = 117.0
        t_preprocessed = tf.expand_dims(t_input-imagenet_mean, 0)
        tf.import_graph_def(graph_def, {'input': t_preprocessed})

        layers = [op.name for op in graph.get_operations() if op.type =='Conv2D' and 'import/' in op.name]
        feature_nums = [int(graph._get_tensor_by_name(name+':0').get_shape()[-1]) for name in layers]

        # Print to see the num of layers and total num of feature channels

        print('Number of Layers:', len(layers))
        print('Total number of feature channels:', sum(feature_nums))
