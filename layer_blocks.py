import numpy as np
import tensorflow as tf

def conv2d(x, filter_size, num_filters, strides, name, padding='SAME', groups=1):
    """2D convolution + non-linearity"""

    input_channels = int(x.get_shape()[-1])

    convolve = lambda i, j: tf.nn.conv2d(i, j, strides=[1, strides, strides, 1], padding=padding, name=name)

    with tf.variable_scope(name) as scope:
        W = tf.get_variable('weights', shape=[filter_size[0], filter_size[1], input_channels/groups, num_filters])
        b = tf.get_variable('biases', shape=[num_filters])

    if groups == 1:
        conv = convolve(x, W)

    # In the cases of multiple groups, split inputs & weights and
    else:
        # Split input and weights and convolve them separately
        x_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
        W_groups = tf.split(axis=3, num_or_size_splits=groups, value=W)
        output_groups = [convolve(i, k) for i, k in zip(x_groups, W_groups)]

        # Concat the convolved output together again
        conv = tf.concat(axis=3, values=output_groups)
        
    return tf.nn.relu(conv + b, name=scope.name)


def conv2d_fixed_padding(x, filter_size, num_filters, strides, name, add_bias=True):
    """Strided 2-D convolution with explicit padding."""
     # The padding is consistent and is based only on `kernel_size`, not on the
     # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
    
    def fixed_padding():
        pad = ( filter_size[0] - 1 ) // 2
        padded_x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]]) # [batch, height, width, channel]
        return padded_x
        
    #if strides > 1:
    x = fixed_padding()
        
    input_channels = int(x.get_shape()[-1])
        
    with tf.variable_scope(name) as scope:
        W = tf.get_variable('weights', shape=[filter_size[0], filter_size[1], input_channels, num_filters])
        if add_bias:
            b = tf.get_variable('biases', shape=[num_filters])

    conv = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID', name=name)
    
    return tf.identity(conv + b if add_bias else conv, name=scope.name)

'''
def batch_normalization(x, n_out, phase_train, name):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope(name) as scope:
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                     name='offset', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                      name='scale', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.99)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5)
    return normed
'''

def batch_norm(x, num_filters, name):
    with tf.variable_scope(name) as scope:
        m = tf.get_variable('mean', shape=[num_filters])
        v = tf.get_variable('variance', shape=[num_filters])
        s = tf.get_variable('scale', shape=[num_filters])
        o = tf.get_variable('offset', shape=[num_filters])
        
    return tf.nn.batch_normalization(x, mean=m, variance=v, scale=s, offset=o, variance_epsilon=1e-5, name=name)

def batch_normalization(x, is_training, name):
    return tf.layers.batch_normalization(x,axis=-1,momentum=0.99, epsilon=1e-3, center=True,scale=True, training=is_training, name=name)

def residual_block(x, num_filters, strides, is_training, name):
    
    ## Block A
    # branch2a
    conv_a2a = conv2d_fixed_padding(x, filter_size=[3,3], num_filters=num_filters, strides=strides, name='res'+name+'a_branch2a', add_bias=False)
    bn_a2a = batch_norm(conv_a2a, num_filters=num_filters, name='bn'+name+'a_branch2a')
    #bn_a2a = batch_normalization(conv_a2a, is_training=is_training, name='bn'+name+'a_branch2a')
    relu_a2a = tf.nn.relu(bn_a2a, name='res'+name+'a_branch2a_relu')
    
    # branch2b
    conv_a2b = conv2d_fixed_padding(relu_a2a, filter_size=[3,3], num_filters=num_filters, strides=1, name='res'+name+'a_branch2b', add_bias=False)
    bn_a2b = batch_norm(conv_a2b, num_filters=num_filters, name='bn'+name+'a_branch2b')
    #bn_a2b = batch_normalization(conv_a2b, is_training=is_training, name='bn'+name+'a_branch2b')

    # branch1
    conv_a1 = conv2d_fixed_padding(x, filter_size=[1,1], num_filters=num_filters, strides=strides, name='res'+name+'a_branch1', add_bias=False)
    bn_a1 = batch_norm(conv_a1, num_filters=num_filters, name='bn'+name+'a_branch1')
    #bn_a1 = batch_normalization(conv_a1, is_training=is_training, name='bn'+name+'a_branch1')

    # add_nl1
    relu_add1 = tf.nn.relu(bn_a1 + bn_a2b, name='res'+name+'a')
    
    ##Block B
    # branch2a
    conv_b2a = conv2d_fixed_padding(relu_add1, filter_size=[3,3], num_filters=num_filters, strides=1, name='res'+name+'b_branch2a', add_bias=False)
    bn_b2a = batch_norm(conv_b2a, num_filters=num_filters, name='bn'+name+'b_branch2a')
    #bn_b2a = batch_normalization(conv_b2a, is_training=is_training, name='bn'+name+'b_branch2a')
    relu_b2a = tf.nn.relu(bn_b2a, name='res'+name+'b_branch2a_relu')
    
    # branch2b
    conv_b2b = conv2d_fixed_padding(relu_b2a, filter_size=[3,3], num_filters=num_filters, strides=1, name='res'+name+'b_branch2b', add_bias=False)
    bn_b2b = batch_norm(conv_b2b, num_filters=num_filters, name='bn'+name+'b_branch2b')
    #bn_b2b = batch_normalization(conv_b2b, is_training=is_training, name='bn'+name+'b_branch2b')

    # add_nl2
    relu_b2b = tf.nn.relu(relu_add1 + bn_b2b, name='res'+name+'b')
    
    return relu_b2b, relu_add1


## Residual blocks for ResNet-50

def bottleneck_block_up(x, num_filters, strides, is_training, name , character):

    ## Block A
    # branch2a
    conv_a2a = conv2d_fixed_padding(x, filter_size=[1,1], num_filters=num_filters, strides=strides, name='res' + name + character +'_branch2a', add_bias=False)
    bn_a2a = batch_normalization(conv_a2a,is_training=is_training, name='bn' + name + character +'_branch2a')
    relu_a2a = tf.nn.relu(bn_a2a, name='res' + name + character +'_branch2a_relu')
    
    # branch2b
    conv_a2b = conv2d_fixed_padding(relu_a2a, filter_size=[3,3], num_filters=num_filters, strides=1, name='res' + name + character +'_branch2b', add_bias=False)
    bn_a2b = batch_normalization(conv_a2b,is_training=is_training,  name='bn' + name + character +'_branch2b')
    relu_a2b = tf.nn.relu(bn_a2b, name='res' + name + character +'_branch2b_relu')
    
    # branch2c
    conv_a2c = conv2d_fixed_padding(relu_a2b, filter_size=[1,1], num_filters= num_filters * 4, strides=1, name='res' + name + character +'_branch2c', add_bias=False)
    bn_a2c = batch_normalization(conv_a2c,is_training=is_training,  name='bn' + name + character +'_branch2c')
    


    # branch1
    conv_a1 = conv2d_fixed_padding(x, filter_size=[1,1], num_filters = num_filters * 4, strides=strides, name='res' + name + character +'_branch1', add_bias=False)
    bn_a1 = batch_normalization(conv_a1,is_training=is_training,  name='bn' + name + character +'_branch1')
    
    # add_nl1
    #relu_add1 = tf.nn.relu(bn_a1 + bn_a2c, name='res'+name+'a')
    return tf.nn.relu(bn_a1 + bn_a2c , name = 'res' + name + character  )


def bottleneck_block_bottom(x, num_filters, strides,is_training, name , character):

    ## Block A
    # branch2a
    conv_a2a = conv2d_fixed_padding(x, filter_size=[1,1], num_filters=num_filters, strides=1, name='res' + name + character +'_branch2a', add_bias=False)
    bn_a2a = batch_normalization(conv_a2a, is_training=is_training, name='bn' + name + character +'_branch2a')
    relu_a2a = tf.nn.relu(bn_a2a, name='res'+name+character +'_branch2a_relu')
    
    # branch2b
    conv_a2b = conv2d_fixed_padding(relu_a2a, filter_size=[3,3], num_filters=num_filters, strides=1, name='res' + name + character +'_branch2b', add_bias=False)
    bn_a2b = batch_normalization(conv_a2b,is_training=is_training, name='bn' + name + character +'_branch2b')
    relu_a2b = tf.nn.relu(bn_a2b, name='res' + name + character +'_branch2b_relu')
    
    # branch2c
    conv_a2c = conv2d_fixed_padding(relu_a2b, filter_size=[1,1], num_filters=num_filters * 4, strides=1, name='res' + name + character +'_branch2c', add_bias=False)
    bn_a2c = batch_normalization(conv_a2c, is_training=is_training, name='bn' + name + character +'_branch2c')
 
    # add_nl1
    #relu_add1 = tf.nn.relu(bn_a1 + bn_a2c, name='res'+name+'a')
    return tf.nn.relu(x + bn_a2c , name = 'res' + name + character )

def bottleneck_block(x, num_filters, strides,is_training, name):
    if(name == "2"):
        print ("2")
        x_a = bottleneck_block_up(x,num_filters=num_filters,strides=strides,name=name,character='a', is_training=is_training)
        x_b = bottleneck_block_bottom(x_a,num_filters=num_filters,strides=strides,name=name,character='b', is_training=is_training)
        x_c = bottleneck_block_bottom(x_b,num_filters=num_filters,strides=strides,name=name,character='c', is_training=is_training)

        out = [x_a,x_b,x_c]


    if(name == "3"):
        print ("3")
        x_a = bottleneck_block_up(x,num_filters=num_filters,strides=strides,name=name,character='a', is_training=is_training)
        x_b = bottleneck_block_bottom(x_a,num_filters=num_filters,strides=strides,name=name,character='b', is_training=is_training)
        x_c = bottleneck_block_bottom(x_b,num_filters=num_filters,strides=strides,name=name,character='c', is_training=is_training)
        x_d = bottleneck_block_bottom(x_c,num_filters=num_filters,strides=strides,name=name,character='d', is_training=is_training)

        out = [x_a,x_b,x_c,x_d] 

    if(name == "4"):
        print ("4")
        x_a = bottleneck_block_up(x,num_filters=num_filters,strides=strides,name=name,character='a',is_training=is_training)
        x_b = bottleneck_block_bottom(x_a,num_filters=num_filters,strides=strides,name=name,character='b',is_training=is_training)
        x_c = bottleneck_block_bottom(x_b,num_filters=num_filters,strides=strides,name=name,character='c',is_training=is_training)
        x_d = bottleneck_block_bottom(x_c,num_filters=num_filters,strides=strides,name=name,character='d',is_training=is_training)
        x_e = bottleneck_block_bottom(x_d,num_filters=num_filters,strides=strides,name=name,character='e',is_training=is_training)
        x_f = bottleneck_block_bottom(x_e,num_filters=num_filters,strides=strides,name=name,character='f',is_training=is_training)

        out = [x_a,x_b,x_c,x_d,x_e,x_f] 

    if(name == "5"):
        print ("5")
        x_a = bottleneck_block_up(x,num_filters=num_filters,strides=strides,name=name,character='a',is_training=is_training)
        x_b = bottleneck_block_bottom(x_a,num_filters=num_filters,strides=strides,name=name,character='b',is_training=is_training)
        x_c = bottleneck_block_bottom(x_b,num_filters=num_filters,strides=strides,name=name,character='c',is_training=is_training)

        out = [x_a,x_b,x_c] 

    return out



def fc(x, num_in, num_out, name, activation = True, do_prob = 1.0):
    """Fully connected layer (+ non-linearity + dropout)"""

    with tf.variable_scope(name) as scope:

        # Create tf variables for the weights and biases
        W = tf.get_variable('weights', shape=[num_in, num_out], trainable=True)
        b = tf.get_variable('biases', [num_out], trainable=True)

        # Matrix multiply weights and inputs and add bias
        h = tf.nn.xw_plus_b(x, W, b, name=scope.name)
        relu = tf.nn.relu(h)


        if activation == True:
            # Apply ReLu non linearity
            return tf.nn.dropout(relu, keep_prob=do_prob)
        else:
            return h

def dilated_conv2d(x, num_filters, filter_size, dilation_rate, strides, name, kernel_constraint=None, padding='SAME'):
    dconv = tf.layers.conv2d(x, num_filters, filter_size, dilation_rate=dilation_rate,
                                    kernel_constraint=kernel_constraint, name=name, use_bias=False)
    return dconv[:,0:dconv.shape[1]:strides[0],0:dconv.shape[2]:strides[1],:]

def separable_conv2d(x, depthwise_filter, pointwise_filter, strides, name):

    with tf.variable_scope(name) as scope:
        W_depthwise = tf.get_variable('weights_depthwise', shape=depthwise_filter)
        W_pointwise = tf.get_variable('weights_pointwise', shape=pointwise_filter)

    return tf.nn.separable_conv2d(x, W_depthwise, W_pointwise, strides, padding='SAME')




## Different versions of transformation block## Different versions of transformation blockss

def transformation_block(x, num_filters, kernel_constraint, training_phase, name):
    
    kernel_size = int(x.get_shape()[2])/7
    depth_size = int(x.get_shape()[3])

    relu_conv_spatial = tf.layers.average_pooling2d(x, [7, 7], strides=[7, 7], name='pool_'+name+'_spatial')

    #conv_spatial = tf.layers.conv2d(x, num_filters, [kernel_size, kernel_size], strides=[kernel_size, kernel_size],
    #                                kernel_constraint=kernel_constraint, name=name+'_spatial', use_bias=False) 
    #conv_spatial = tf.layers.conv2d(pool_spatial, depth_size, [kernel_size, kernel_size], strides=[kernel_size, kernel_size],
    #                                kernel_constraint=kernel_constraint, name=name+'_spatial', use_bias=False)
    #bn_conv_spatial = tf.layers.batch_normalization(conv_spatial, training=training_phase, name='bn_'+name+'_spatial')
    #relu_conv_spatial = tf.nn.relu(bn_conv_spatial, name='relu_'+name+'_spatial')

    conv_depth = tf.layers.conv2d(relu_conv_spatial, num_filters, [1,1], strides=[1,1],
                                  kernel_constraint=kernel_constraint, name=name+'_depth', use_bias=False)
    bn_conv_depth = tf.layers.batch_normalization(conv_depth, training=training_phase, name='bn_'+name+'_depth')
    #bn_conv_depth = conv_depth
    relu_conv_depth = tf.nn.relu(bn_conv_depth, name='relu_'+name+'_depth')

    relu_conv_depth = tf.layers.max_pooling2d(relu_conv_depth, [kernel_size, kernel_size], strides=[kernel_size, kernel_size], name='max_'+name+'_depth')

    return relu_conv_depth

def transformation_block_v1(x, num_filters, kernel_constraint, training_phase, name):

    kernel_size = int(x.get_shape()[2])/7
    depth_size = int(x.get_shape()[3])

    conv_spatial = tf.layers.conv2d(x, num_filters, [7, 7], strides=[7, 7],
                                    kernel_constraint=kernel_constraint, name=name+'_spatial', use_bias=False) 
    #conv_spatial = tf.layers.conv2d(pool_spatial, depth_size, [kernel_size, kernel_size], strides=[kernel_size, kernel_size],
    #                                kernel_constraint=kernel_constraint, name=name+'_spatial', use_bias=False)
    bn_conv_spatial = tf.layers.batch_normalization(conv_spatial, training=training_phase, name='bn_'+name+'_spatial')
    #bn_conv_spatial = conv_spatial
    relu_conv_spatial = tf.nn.relu(bn_conv_spatial, name='relu_'+name+'_spatial')

    conv_depth = tf.layers.conv2d(relu_conv_spatial, num_filters, [1,1], strides=[1,1],
                                  kernel_constraint=kernel_constraint, name=name+'_depth', use_bias=False)
    bn_conv_depth = tf.layers.batch_normalization(conv_depth, training=training_phase, name='bn_'+name+'_depth')
    #bn_conv_depth = conv_depth
    relu_conv_depth = tf.nn.relu(bn_conv_depth, name='relu_'+name+'_depth')

    relu_conv_depth = tf.layers.max_pooling2d(relu_conv_depth, [kernel_size, kernel_size], strides=[kernel_size, kernel_size], name='max_'+name+'_depth')

    return relu_conv_depth

def transformation_block_v1_bis(x, num_filters, kernel_constraint, training_phase, name):
    '''
    Transformation layer w/ separable convolutions:
        num_conv = 1
        residual = NO
    '''

    kernel_size = int(x.get_shape()[2])/7

    sep_conv = separable_conv2d(x, [7,7,int(x.get_shape()[-1]),1], [1,1,int(x.get_shape()[-1]),num_filters], [1,7,7,1], name='sep_'+name)
    bn_sep_conv = tf.layers.batch_normalization(sep_conv, training=training_phase, name='bn_sep_conv_'+name)
    relu_sep_conv = tf.nn.relu(bn_sep_conv, name='relu_sep_conv'+name)

    out = tf.layers.max_pooling2d(relu_sep_conv, [kernel_size, kernel_size], strides=[kernel_size,kernel_size], name='avg_pool__'+name)

    return out


def transformation_block_v2(x, num_filters, kernel_constraint, training_phase, name):

    kernel_size = int(x.get_shape()[2])/7
    depth_size = int(x.get_shape()[3])

    conv_spatial = dilated_conv2d(x, num_filters, [3,3], dilation_rate=[3,3], strides=[7, 7],
                                  kernel_constraint=kernel_constraint, name=name+'_spatial')
    bn_conv_spatial = tf.layers.batch_normalization(conv_spatial, training=training_phase, name='bn_'+name+'_spatial')
    relu_conv_spatial = tf.nn.relu(bn_conv_spatial, name='relu_'+name+'_spatial')

    conv_depth = tf.layers.conv2d(relu_conv_spatial, num_filters, [1,1], strides=[1,1],
                                  kernel_constraint=kernel_constraint, name=name+'_depth', use_bias=False)
    bn_conv_depth = tf.layers.batch_normalization(conv_depth, training=training_phase, name='bn_'+name+'_depth')
    #bn_conv_depth = conv_depth
    relu_conv_depth = tf.nn.relu(bn_conv_depth, name='relu_'+name+'_depth')

    relu_conv_depth = tf.layers.max_pooling2d(relu_conv_depth, [kernel_size, kernel_size], strides=[kernel_size, kernel_size], name='max_'+name+'_depth')

    return relu_conv_depth

def transformation_block_v3(x, num_filters, kernel_constraint, training_phase, name):
    '''
    Transformation layer w/ separable convolutions:
        num_conv = 1
        residual = NO
    '''
    
    striding_factor = int(x.get_shape()[2])/7
    max_pool_stride = striding_factor/2
    if max_pool_stride<=1:
        conv_stride = striding_factor
        x_pool = x
    else:
        conv_stride = striding_factor/max_pool_stride
        x_pool = tf.layers.max_pooling2d(x, [max_pool_stride+1,max_pool_stride+1], strides=[max_pool_stride,max_pool_stride], name='pool_'+name)
    print(x.get_shape(), num_filters)   
    print(striding_factor, max_pool_stride, conv_stride) 
    sep_conv = separable_conv2d(x_pool, [3,3,int(x_pool.get_shape()[-1]),1], [1,1,int(x_pool.get_shape()[-1]),num_filters], [1,conv_stride,conv_stride,1], name='sep_'+name)
    bn_sep_conv = tf.layers.batch_normalization(sep_conv, training=training_phase, name='bn_sep_conv_'+name)
    relu_sep_conv = tf.nn.relu(bn_sep_conv, name='relu_sep_conv'+name)

    out = tf.layers.average_pooling2d(relu_sep_conv, [7,7], strides=[1,1], name='avg_pool__'+name)

    return out

def transformation_block_v4(x, num_filters, constraint, training_phase, name):
    '''
    Transformation layer w/ separable convolutions:
        num_conv = 1
        residual = YES
    '''
    
    striding_factor = int(x.get_shape()[2])/7
    max_pool_stride = striding_factor/2
    if max_pool_stride<=1:
        conv_stride = striding_factor
        x_pool = x
    else:
        conv_stride = striding_factor/max_pool_stride
        x_pool = tf.layers.max_pooling2d(x, [max_pool_stride+1,max_pool_stride+1], strides=[max_pool_stride,max_pool_stride], name='pool_'+name)
    
    sep_conv = separable_conv2d(x_pool, [3,3,int(x_pool.get_shape()[-1]),1], [1,1,int(x_pool.get_shape()[-1]),num_filters], [1,conv_stride,conv_stride,1], name='sep_'+name)
    bn_sep_conv = tf.layers.batch_normalization(sep_conv, training=training_phase, name='bn_sep_conv_'+name)
    relu_sep_conv = tf.nn.relu(bn_sep_conv, name='relu_sep_conv'+name)

    res_conv = tf.layers.conv2d(x, num_filters, [1,1], strides=[striding_factor,striding_factor],
                                name=name+'_depth', use_bias=False)
    bn_res_conv = tf.layers.batch_normalization(res_conv, training=training_phase, name='bn_res_conv_'+name)
    relu_res_conv = tf.nn.relu(bn_res_conv, name='relu_res_conv'+name)

    return relu_sep_conv + relu_res_conv

def transformation_block_v5(x, num_filters, constraint, training_phase, name):
    '''
    Transformation layer w/ separable convolutions:
        num_conv = 2
        residual = NO
    '''
    
    striding_factor = int(x.get_shape()[2])/7
    max_pool_stride = striding_factor/2
    if max_pool_stride<=1:
        conv_stride = striding_factor
        x_pool = x
    else:
        conv_stride = striding_factor/max_pool_stride
        x_pool = tf.layers.max_pooling2d(x, [max_pool_stride+1,max_pool_stride+1], strides=[max_pool_stride,max_pool_stride], name='pool_'+name)
    
    sep_conv1 = separable_conv2d(x_pool, [3,3,int(x_pool.get_shape()[-1]),1], [1,1,int(x_pool.get_shape()[-1]),num_filters], [1,conv_stride,conv_stride,1], name='sep_conv1_'+name)
    bn_sep_conv1 = tf.layers.batch_normalization(sep_conv1, training=training_phase, name='bn_sep_conv1_'+name)
    relu_sep_conv1 = tf.nn.relu(bn_sep_conv1, name='relu_sep_conv1_'+name)

    sep_conv2 = separable_conv2d(relu_sep_conv1, [3,3,int(relu_sep_conv1.get_shape()[-1]),1], [1,1,int(relu_sep_conv1.get_shape()[-1]),num_filters], [1,1,1,1], name='sep_conv2_'+name)
    bn_sep_conv2 = tf.layers.batch_normalization(sep_conv2, training=training_phase, name='bn_sep_conv2_'+name)
    relu_sep_conv2 = tf.nn.relu(bn_sep_conv2, name='relu_sep_conv2_'+name)

    return relu_sep_conv2

def transformation_block_v6(x, num_filters, constraint, training_phase, name):
    '''
    Transformation layer w/ separable convolutions:
        num_conv = 2
        residual = YES
    '''

    striding_factor = int(x.get_shape()[2])/7
    max_pool_stride = striding_factor/2
    if max_pool_stride<=1:
        conv_stride = striding_factor
        x_pool = x
    else:
        conv_stride = striding_factor/max_pool_stride
        x_pool = tf.layers.max_pooling2d(x, [max_pool_stride+1,max_pool_stride+1], strides=[max_pool_stride,max_pool_stride], name='pool_'+name)

    sep_conv1 = separable_conv2d(x_pool, [3,3,int(x_pool.get_shape()[-1]),1], [1,1,int(x_pool.get_shape()[-1]),num_filters], [1,conv_stride,conv_stride,1], name='sep_conv1_'+name)
    bn_sep_conv1 = tf.layers.batch_normalization(sep_conv1, training=training_phase, name='bn_sep_conv1_'+name)
    relu_sep_conv1 = tf.nn.relu(bn_sep_conv1, name='relu_sep_conv1_'+name)

    sep_conv2 = separable_conv2d(relu_sep_conv1, [3,3,int(relu_sep_conv1.get_shape()[-1]),1], [1,1,int(relu_sep_conv1.get_shape()[-1]),num_filters], [1,1,1,1], name='sep_conv2_'+name)
    bn_sep_conv2 = tf.layers.batch_normalization(sep_conv2, training=training_phase, name='bn_sep_conv2_'+name)
    relu_sep_conv2 = tf.nn.relu(bn_sep_conv2, name='relu_sep_conv2_'+name)

    res_conv = tf.layers.conv2d(x, num_filters, [1,1], strides=[striding_factor,striding_factor],
                                name=name+'_depth', use_bias=False)
    bn_res_conv = tf.layers.batch_normalization(res_conv, training=training_phase, name='bn_res_conv_'+name)
    relu_res_conv = tf.nn.relu(bn_res_conv, name='relu_res_conv'+name)

    return relu_sep_conv2 + relu_res_conv


