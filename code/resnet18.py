from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from layer_blocks import *
from utils import *

class ResNet(object):
    """Class used to create a ResNet-18 model"""

    def __init__(self, x, num_classes, do_prob=0.5, is_training=False, mode = 'rgb'):

        self.x = x
        self.num_classes = num_classes
        self.do_prob = do_prob
        self.is_training = is_training
        self.mode = mode

        self._create_model()
        
    def _create_model(self):
        
        # Initial conv & pool
        self.conv1 = conv2d_fixed_padding(self.x, filter_size=[7,7], num_filters=64, strides=2, name='conv1', add_bias=False)
        self.bn1 = batch_norm(self.conv1, num_filters=64, name='bn_conv1')
        self.relu1 = tf.nn.relu(self.bn1, name='relu_conv1')
        self.pool1 = tf.nn.max_pool(self.relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], 
                                                 name='pool1', padding='SAME')
        
        # Residual blocks (x4)
        self.res2, self.inter_res2 = residual_block(self.pool1, num_filters=64, strides=1, is_training=self.is_training, name='2')
        self.res3, self.inter_res3 = residual_block(self.res2, num_filters=128, strides=2, is_training=self.is_training, name='3')
        self.res4, self.inter_res4 = residual_block(self.res3, num_filters=256, strides=2, is_training=self.is_training, name='4')
        self.res5, self.inter_res5 = residual_block(self.res4, num_filters=512, strides=2, is_training=self.is_training, name='5')
        
        # Final pool & classifier
        self.pool2 = tf.nn.avg_pool(self.res5, ksize=[1, 7, 7, 1], strides=[1, 1, 1, 1], 
                                                 name='pool2', padding='VALID')
        self.pool2_flat = tf.reshape(self.pool2, [-1, flat_shape(self.pool2)])
        self.fc = fc(self.pool2_flat, num_in=512, num_out=self.num_classes, activation = False, name='fc1000_wrgb')
        self.softmax = tf.nn.softmax(self.fc, name='softmax')
        
        ## Conv1x1
        #with tf.variable_scope('conv1x1'):
        #    self.conv1x1_res3 = conv2d_fixed_padding(self.res3, filter_size=[1,1], num_filters=1, strides=1,
        #                                         name='conv1x1_res3', add_bias=False)
        #    self.relu_conv1x1_res3 = tf.nn.relu(self.conv1x1_res3, name='relu_conv1x1_res3')
        #    self.conv1x1_res4 = conv2d_fixed_padding(self.res4, filter_size=[1, 1], num_filters=4, strides=1,
        #                                         name='conv1x1_res4', add_bias=False)
        #    self.relu_conv1x1_res4 = tf.nn.relu(self.conv1x1_res4, name='relu_conv1x1_res4')
        #    self.conv1x1_res5 = conv2d_fixed_padding(self.res5, filter_size=[1, 1], num_filters=16, strides=1,
        #                                         name='conv1x1_res5', add_bias=False)
        #    self.relu_conv1x1_res5 = tf.nn.relu(self.conv1x1_res5, name='relu_conv1x1_res5')

        
    def load_params(self, sess, params_dir, trainable=True, skip_layers=None):
        """ Load pre-trained params """

        params = np.load(params_dir).item()

        if trainable:
            s = 'trainable'
        else:
            s = 'non trainable'
        print("\nCopying ({}) parameters of layer...".format(s))
        
        

        # Loop over the stored layers
        for layer in params:

            # Not in the layers to re-train from scratch
            with tf.variable_scope(self.mode +"/"+ layer, reuse=True):
            #with tf.variable_scope(layer, reuse=True):    
                # Distinguish between batch norm layer and others
                if layer[:2]=='bn':
                    # Load mean
                    var = tf.get_variable('mean', trainable=trainable)
                    sess.run(var.assign(params[layer]['mean']))
                    
                    # Load variance
                    var = tf.get_variable('variance', trainable=trainable)
                    sess.run(var.assign(params[layer]['variance']))
                    
                    # Load scale
                    var = tf.get_variable('scale', trainable=trainable)
                    sess.run(var.assign(params[layer]['scale']))
                    
                    # Load offset
                    var = tf.get_variable('offset', trainable=trainable)
                    sess.run(var.assign(params[layer]['offset']))
                    
                else:
                    # Load weights
                    var = tf.get_variable('weights', trainable=trainable)
                    sess.run(var.assign(params[layer]['weights']))
                    try:
                        var = tf.get_variable('biases', trainable=trainable)
                        sess.run(var.assign(params[layer]['biases']))
                    except:
                        print("{} (bias term = False)".format(layer))
                    
        return params
