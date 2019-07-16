import os
import sys
import shutil
from datetime import datetime
import progressbar
from image_data_handler_joint_multimodal import ImageDataHandler
from resnet18_conv1x1 import ResNet
from layer_blocks import *
from tensorflow.data import Iterator
from utils import flat_shape
import cPickle as pickle

from keras.objectives import categorical_crossentropy
from keras.optimizers import RMSprop
from keras.layers import LSTM, GRU, Dense, Dropout
from keras.metrics import categorical_accuracy
from keras import backend as K

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

from tensorflow.python.client import device_lib
print device_lib.list_local_devices()

""" Configuration setting """

tf.set_random_seed(7)

# Data-related params
dataset_dir_rgb = '/mnt/datasets/wrgbd_eval_dataset/wrgbd_rgb++/'
params_dir_rgb = '/mnt/params/models/resnet18_wrgbd/wrgbd_rgb++_split1.npy'

dataset_dir_depth = '/mnt/datasets/wrgbd_eval_dataset/wrgbd_surfnorm++/'
params_dir_depth = '/mnt/params/models/resnet18_wrgbd/wrgbd_surfnorm++_split1.npy'

train_file = '/mnt/datasets/wrgbd_eval_dataset/split_files_and_labels/sync_tr_split_1.txt'
val_file = '/mnt/datasets/wrgbd_eval_dataset/split_files_and_labels/sync_val_split_1.txt'

# Log params
log_dir = '../log/test/'
log = ["epoch", "train_loss", "val_loss", "val_acc"]
tensorboard_log = '/tmp/tensorflow/'

# Solver params
learning_rate = [[0.0001]]
num_epochs = 30
batch_size = [[128]]
num_neurons = [[100]]
l2_factor = [[0.0]]
maximum_norm = [[4],[np.inf]]
dropout_rate = [[0.4]]

depth_transf = [[1024]]
transf_block = transformation_block_v1

# Checkpoint dir
checkpoint_dir = "/tmp/my_caffenet/"
if not os.path.isdir(checkpoint_dir): os.mkdir(checkpoint_dir)

# Input/Output
num_classes = 51
img_size = [224, 224]
num_channels = 3

def generate_perp_loss(rgb_nodes, depth_nodes, reg_lambda):

    loss = 0
    loss_list = []
    for rgb, depth, l in zip(rgb_nodes, depth_nodes, reg_lambda):
        aT_b = tf.matmul(rgb, depth, transpose_a=True, transpose_b=False)
        frob_norm = tf.norm(aT_b)
        feature_dim = int(rgb.shape[1])
        squared_frob = tf.square(tf.divide(frob_norm, feature_dim))
        loss_list.append(squared_frob)
        loss += l*squared_frob

    return loss, loss_list


def log_file(history_callback, params):

    log_name = log_dir + 'log_res2'
    for p in params:
        log_name += ('_' + str(p))
    with open(log_name, 'w+') as f:
        num_entries = len(history_callback[log[0]])
        for i in np.arange(num_entries):
            line = log[0] + ' = ' + str(history_callback[log[0]][i]) + ' , ' + \
                   log[1] + ' = ' + str(history_callback[log[1]][i]) + ' , ' + \
                   log[2] + ' = ' + str(history_callback[log[2]][i]) + ' , ' + \
                   log[3] + ' = ' + str(history_callback[log[3]][i]) + '\n'

            f.write(line)

    print('Log file saved.\n')

def count_params(trainable_variables):
    global_w = 0
    for var in trainable_variables:
        shape = var.shape
        local_w = 1
        for i in range(len(shape)):
            local_w *= int(shape[i])
        global_w += local_w
    return global_w


# Create a all combination of hyper-parameters
set_params = [lr+nn+bs+aa+mn+do+dt for lr in learning_rate for nn in num_neurons for bs in batch_size for aa in l2_factor for mn in maximum_norm for do in dropout_rate for dt in depth_transf]

for hp in set_params:

    lr = hp[0]
    nn = hp[1]
    bs = hp[2]
    aa = hp[3]
    mn = hp[4]
    do = hp[5]
    dt = hp[6]

    """ Data management """

    # Place data loading and preprocessing on the cpu
    #with tf.device('/cpu:0'):
    dataset_dir = [dataset_dir_rgb, dataset_dir_depth]

    tr_data = ImageDataHandler(
        train_file,
        data_dir=dataset_dir,
        img_size=img_size,
        batch_size=bs,
        num_classes=num_classes,
        shuffle=True,
        random_crops=False)

    val_data = ImageDataHandler(
        val_file,
        data_dir=dataset_dir,
        img_size=img_size,
        batch_size=bs,
        num_classes=num_classes,
        shuffle=False,
        random_crops=False)

        # create a re-initializable iterator given the dataset structure
        # no need for two different to deal with training and val data,
       # just two initializers
    iterator = Iterator.from_structure(tr_data.data.output_types,
                                       tr_data.data.output_shapes)
    next_batch = iterator.get_next() # op

    # Ops for initializing the two different iterators
    training_init_op = iterator.make_initializer(tr_data.data)
    validation_init_op = iterator.make_initializer(val_data.data)

    # Get the number of training/validation steps per epoch
    tr_batches_per_epoch = int(np.floor(tr_data.data_size/bs))
    val_batches_per_epoch = int(np.floor(val_data.data_size/bs))

    init_op_rgb = {'training': training_init_op, 'validation': validation_init_op}
    batches_per_epoch = {'training': tr_batches_per_epoch, 'validation': val_batches_per_epoch}

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
  
    # Log vars
    log_epoch = []
    log_train_loss = []
    log_val_loss = []
    log_val_acc = []


    # Start Tensorflow session
    #with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, device_count={'GPU':1})) as sess:
    with tf.Session() as sess: 
        """ Network definition """        

        ## RGB branch

        # TF placeholder for graph input and output
        x_rgb = tf.placeholder(tf.float32, [None, img_size[0], img_size[1], 3])
        x_depth = tf.placeholder(tf.float32, [None, img_size[0], img_size[1], 3])
        y = tf.placeholder(tf.float32, [None, num_classes])

        x_rgb = tf.reshape(x_rgb, [-1, img_size[0], img_size[1], 3])
        x_depth = tf.reshape(x_depth, [-1, img_size[0], img_size[1], 3])
        y = tf.reshape(y, [-1, num_classes])

        keep_prob = tf.placeholder(tf.float32)
        training_phase = tf.placeholder(tf.bool)

        # Max norm
        ckn = np.inf
        rkc = mn#np.inf
        rrc = mn#np.inf
        dkc = mn#np.inf
        conv_kernel_constraint = tf.keras.constraints.MaxNorm(ckn, axis=[0,1,2])
        rnn_kernel_constraint = tf.keras.constraints.MaxNorm(rkc, axis=[0,1])
        rnn_recurrent_constraint = tf.keras.constraints.MaxNorm(rrc, axis=1)
        dense_kernel_constraint = tf.keras.constraints.MaxNorm(dkc, axis=0)

        # Initialize models
        with tf.variable_scope('rgb', reuse=None):
            model_rgb = ResNet(x_rgb, num_classes, mode='rgb')
        with tf.variable_scope('depth', reuse=None):
            model_depth = ResNet(x_depth, num_classes, mode='depth')


        # Extract features
        #res1_rgb = model_rgb.relu1
        inter_res2_rgb = model_rgb.inter_res2
        res2_rgb = model_rgb.res2
        inter_res3_rgb = model_rgb.inter_res3
        res3_rgb = model_rgb.res3
        inter_res4_rgb = model_rgb.inter_res4
        res4_rgb = model_rgb.res4
        inter_res5_rgb = model_rgb.inter_res5
        res5_rgb = model_rgb.res5

        #res1_depth = model_depth.relu1
        inter_res2_depth = model_depth.inter_res2
        res2_depth = model_depth.res2
        inter_res3_depth = model_depth.inter_res3
        res3_depth = model_depth.res3
        inter_res4_depth = model_depth.inter_res4
        res4_depth = model_depth.res4
        inter_res5_depth = model_depth.inter_res5
        res5_depth = model_depth.res5


        sm_rgb = model_rgb.softmax
        sm_depth = model_depth.softmax

        # Conv1x1
        with tf.variable_scope('conv1x1'):
            
            #depth_transf = 64
            
            #relu_conv1x1_res1_rgb = transformation_block(res1_rgb, 1024, conv_kernel_constraint, training_phase, 'redux_rgb_res1')
            relu_conv1x1_inter_res2_rgb = transf_block(inter_res2_rgb, dt, conv_kernel_constraint, training_phase, 'redux_rgb_inter_res2')
            relu_conv1x1_res2_rgb = transf_block(res2_rgb, dt, conv_kernel_constraint, training_phase, 'redux_rgb_res2') 
            relu_conv1x1_inter_res3_rgb = transf_block(inter_res3_rgb, dt, conv_kernel_constraint, training_phase, 'redux_rgb_inter_res3')
            relu_conv1x1_res3_rgb = transf_block(res3_rgb, dt, conv_kernel_constraint, training_phase, 'redux_rgb_res3')
            relu_conv1x1_inter_res4_rgb = transf_block(inter_res4_rgb, dt, conv_kernel_constraint, training_phase, 'redux_rgb_inter_res4')
            relu_conv1x1_res4_rgb = transf_block(res4_rgb, dt, conv_kernel_constraint, training_phase, 'redux_rgb_res4')
            relu_conv1x1_inter_res5_rgb = transf_block(inter_res5_rgb, dt, conv_kernel_constraint, training_phase, 'redux_rgb_inter_res5')
            relu_conv1x1_res5_rgb = transf_block(res5_rgb, dt, conv_kernel_constraint, training_phase, 'redux_rgb_res5')
            
            #relu_conv1x1_res1_depth = transformation_block(res1_depth, 1024, conv_kernel_constraint, training_phase, 'redux_depth_res1')
            relu_conv1x1_inter_res2_depth = transf_block(inter_res2_depth, dt, conv_kernel_constraint, training_phase, 'redux_depth_inter_res2')
            relu_conv1x1_res2_depth = transf_block(res2_depth, dt, conv_kernel_constraint, training_phase, 'redux_depth_res2')
            relu_conv1x1_inter_res3_depth = transf_block(inter_res3_depth, dt, conv_kernel_constraint, training_phase, 'redux_depth_inter_res3')
            relu_conv1x1_res3_depth = transf_block(res3_depth, dt, conv_kernel_constraint, training_phase, 'redux_depth_res3')
            relu_conv1x1_inter_res4_depth = transf_block(inter_res4_depth, dt, conv_kernel_constraint, training_phase, 'redux_depth_inter_res4')
            relu_conv1x1_res4_depth = transf_block(res4_depth, dt, conv_kernel_constraint, training_phase, 'redux_depth_res4')
            relu_conv1x1_inter_res5_depth = transf_block(inter_res5_depth, dt, conv_kernel_constraint, training_phase, 'redux_depth_inter_res5')
            relu_conv1x1_res5_depth = transf_block(res5_depth, dt, conv_kernel_constraint, training_phase, 'redux_depth_res5')
 
        #relu_conv1x1_res1_rgb = tf.reshape(relu_conv1x1_res1_rgb, [-1, flat_shape(relu_conv1x1_res1_rgb)])
        relu_conv1x1_inter_res2_rgb = tf.reshape(relu_conv1x1_inter_res2_rgb, [-1, flat_shape(relu_conv1x1_inter_res2_rgb)])
        relu_conv1x1_res2_rgb = tf.reshape(relu_conv1x1_res2_rgb, [-1, flat_shape(relu_conv1x1_res2_rgb)])
        relu_conv1x1_inter_res3_rgb = tf.reshape(relu_conv1x1_inter_res3_rgb, [-1, flat_shape(relu_conv1x1_inter_res3_rgb)])
        relu_conv1x1_res3_rgb = tf.reshape(relu_conv1x1_res3_rgb, [-1, flat_shape(relu_conv1x1_res3_rgb)])
        relu_conv1x1_inter_res4_rgb = tf.reshape(relu_conv1x1_inter_res4_rgb, [-1, flat_shape(relu_conv1x1_inter_res4_rgb)])
        relu_conv1x1_res4_rgb = tf.reshape(relu_conv1x1_res4_rgb, [-1, flat_shape(relu_conv1x1_res4_rgb)])
        relu_conv1x1_inter_res5_rgb = tf.reshape(relu_conv1x1_inter_res5_rgb, [-1, flat_shape(relu_conv1x1_inter_res5_rgb)])
        relu_conv1x1_res5_rgb = tf.reshape(relu_conv1x1_res5_rgb, [-1, flat_shape(relu_conv1x1_res5_rgb)])

        #relu_conv1x1_res1_depth = tf.reshape(relu_conv1x1_res1_depth, [-1, flat_shape(relu_conv1x1_res1_depth)])
        relu_conv1x1_inter_res2_depth = tf.reshape(relu_conv1x1_inter_res2_depth, [-1, flat_shape(relu_conv1x1_inter_res2_depth)])
        relu_conv1x1_res2_depth = tf.reshape(relu_conv1x1_res2_depth, [-1, flat_shape(relu_conv1x1_res2_depth)])
        relu_conv1x1_inter_res3_depth = tf.reshape(relu_conv1x1_inter_res3_depth, [-1, flat_shape(relu_conv1x1_inter_res3_depth)])
        relu_conv1x1_res3_depth = tf.reshape(relu_conv1x1_res3_depth, [-1, flat_shape(relu_conv1x1_res3_depth)])
        relu_conv1x1_inter_res4_depth = tf.reshape(relu_conv1x1_inter_res4_depth, [-1, flat_shape(relu_conv1x1_inter_res4_depth)])
        relu_conv1x1_res4_depth = tf.reshape(relu_conv1x1_res4_depth, [-1, flat_shape(relu_conv1x1_res4_depth)])
        relu_conv1x1_inter_res5_depth = tf.reshape(relu_conv1x1_inter_res5_depth, [-1, flat_shape(relu_conv1x1_inter_res5_depth)])
        relu_conv1x1_res5_depth = tf.reshape(relu_conv1x1_res5_depth, [-1, flat_shape(relu_conv1x1_res5_depth)])

        # RGB and depth pipelines' merge point
        #relu_conv1x1_res1 = tf.concat([relu_conv1x1_res1_rgb, relu_conv1x1_res1_depth], axis=1)
        relu_conv1x1_inter_res2 = tf.concat([relu_conv1x1_inter_res2_rgb, relu_conv1x1_inter_res2_depth], axis=1)
        relu_conv1x1_res2 = tf.concat([relu_conv1x1_res2_rgb, relu_conv1x1_res2_depth], axis=1)
        relu_conv1x1_inter_res3 = tf.concat([relu_conv1x1_inter_res3_rgb, relu_conv1x1_inter_res3_depth], axis=1)
        relu_conv1x1_res3 = tf.concat([relu_conv1x1_res3_rgb, relu_conv1x1_res3_depth], axis=1)
        relu_conv1x1_inter_res4 = tf.concat([relu_conv1x1_inter_res4_rgb, relu_conv1x1_inter_res4_depth], axis=1)
        relu_conv1x1_res4 = tf.concat([relu_conv1x1_res4_rgb, relu_conv1x1_res4_depth], axis=1)
        relu_conv1x1_inter_res5 = tf.concat([relu_conv1x1_inter_res5_rgb, relu_conv1x1_inter_res5_depth], axis=1)
        relu_conv1x1_res5 = tf.concat([relu_conv1x1_res5_rgb, relu_conv1x1_res5_depth], axis=1)

        rnn_input = tf.stack([relu_conv1x1_inter_res2, relu_conv1x1_res2, relu_conv1x1_inter_res3, relu_conv1x1_res3, relu_conv1x1_inter_res4, relu_conv1x1_res4, relu_conv1x1_inter_res5, relu_conv1x1_res5], axis=1)
        #rnn_input = tf.stack([relu_conv1x1_inter_res4, relu_conv1x1_res4, relu_conv1x1_inter_res5, relu_conv1x1_res5], axis=1)
 
        # Recurrent net
        with tf.variable_scope("rnn"):
            rnn_h = GRU(nn, activation='tanh', dropout=do, recurrent_dropout=do, name="rnn_h", 
                        kernel_constraint=rnn_kernel_constraint, recurrent_constraint=rnn_recurrent_constraint)(rnn_input)
            preds = Dense(num_classes, activation='softmax')(rnn_h)

        # Include keras-related metadata in the session
        K.set_session(sess)

        # Define trainable variables
        trainable_variables_rnn = [v for v in tf.trainable_variables(scope="rnn")]
        trainable_variables_conv1x1 = [v for v in tf.trainable_variables(scope="conv1x1")]
        #trainable_variables_rgb = [v for v in tf.trainable_variables(scope="rgb")]
        #trainable_variables_depth = [v for v in tf.trainable_variables(scope="depth")]

        # Define the training and validation opsi
        global_step = tf.Variable(0, trainable=False)
        increment_global_step = tf.assign(global_step, global_step+1)
        lr_boundaries = [int(num_epochs*tr_batches_per_epoch*0.5)]#, int(num_epochs*tr_batches_per_epoch*0.6)]
        lr_values = [lr, lr/10]#, lr/100]
        decayed_lr = tf.train.piecewise_constant(global_step, lr_boundaries, lr_values)
        
        #k = 0.2
        #decayed_lr = tf.train.inverse_time_decay(lr, global_step, 1000, k)

        lr_mult_conv1x1 = 1 

        # L2-regularization
        alpha_rnn = aa
        alpha_conv1x1 = aa
        l2_rnn = tf.add_n([tf.nn.l2_loss(tv_rnn) for tv_rnn in trainable_variables_rnn
                                                    if 'bias' not in tv_rnn.name])*alpha_rnn
        l2_conv1x1 = tf.add_n([tf.nn.l2_loss(tv_conv1x1) for tv_conv1x1 in trainable_variables_conv1x1
                                                    if 'bias' not in tv_conv1x1.name])*alpha_conv1x1
        

        # F2-norm
        rgb_nodes = [relu_conv1x1_inter_res2_rgb, relu_conv1x1_res2_rgb, relu_conv1x1_inter_res3_rgb, relu_conv1x1_res3_rgb, relu_conv1x1_inter_res4_rgb, relu_conv1x1_res4_rgb, relu_conv1x1_inter_res5_rgb, relu_conv1x1_res5_rgb]
        #rgb_nodes = [relu_conv1x1_inter_res4_rgb, relu_conv1x1_res4_rgb, relu_conv1x1_inter_res5_rgb, relu_conv1x1_res5_rgb]
        depth_nodes = [relu_conv1x1_inter_res2_depth, relu_conv1x1_res2_depth, relu_conv1x1_inter_res3_depth, relu_conv1x1_res3_depth, relu_conv1x1_inter_res4_depth, relu_conv1x1_res4_depth, relu_conv1x1_inter_res5_depth, relu_conv1x1_res5_depth]
        #depth_nodes = [relu_conv1x1_inter_res4_depth, relu_conv1x1_res4_depth, relu_conv1x1_inter_res5_depth, relu_conv1x1_res5_depth]
        reg = tf.Variable(0.0, trainable=False)
        #sigm_reg = 0.0001*tf.sigmoid(((12*tf.cast(global_step,tf.float32))/(num_epochs*tr_batches_per_epoch))-6, name='reg_sigmoid')
        sigm_reg = 0.00001
        recompute_reg = tf.assign(reg, sigm_reg)
        reg_lambda = [10*recompute_reg, 10*recompute_reg, recompute_reg, recompute_reg, 0.01*recompute_reg, 0.01*recompute_reg, 0.0*recompute_reg, 0.0*recompute_reg] 
       
        # Loss 
        loss_perp, loss_perp_list = generate_perp_loss(rgb_nodes, depth_nodes, reg_lambda) 
        loss_l2 = l2_rnn + l2_conv1x1
        loss_cls = tf.reduce_mean(categorical_crossentropy(y, preds))
    
        loss = loss_cls + loss_perp #+ loss_l2
        train_step_rnn = tf.keras.optimizers.RMSprop(lr=decayed_lr).get_updates(loss=loss, params=trainable_variables_rnn)
        train_step_conv1x1 = tf.keras.optimizers.RMSprop(lr=decayed_lr*lr_mult_conv1x1).get_updates(loss=loss, params=trainable_variables_conv1x1)
        #train_step_rgb = tf.train.RMSPropOptimizer(decayed_lr*0.01, momentum=0.9, decay = 0.0).minimize(loss, var_list=trainable_variables_rgb)
        #train_step_depth = tf.train.RMSPropOptimizer(decayed_lr*0.01, momentum=0.9, decay = 0.0).minimize(loss, var_list=trainable_variables_depth) 
        #train_step_rnn = tf.train.RMSPropOptimizer(decayed_lr).minimize(loss, var_list=trainable_variables_rnn)
        #train_step_conv1x1 = tf.train.RMSPropOptimizer(decayed_lr*lr_mult_conv1x1, momentum=0.9, decay = 0.0).minimize(loss, var_list=trainable_variables_conv1x1)
        #train_step_conv1x1 = tf.train.RMSPropOptimizer(decayed_lr*lr_mult_conv1x1).minimize(loss, var_list=trainable_variables_conv1x1)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)                                                                           
        with tf.control_dependencies(update_ops):
            train_step = tf.group(train_step_rnn, train_step_conv1x1, increment_global_step, recompute_reg)
            #train_step = tf.group(train_step_rnn, train_step_conv1x1, train_step_rgb, train_step_depth, increment_global_step, recompute_reg)
        accuracy = tf.reduce_mean(categorical_accuracy(y, preds))   

        accuracy_rgb = tf.reduce_mean(categorical_accuracy(y, sm_rgb))
        accuracy_depth = tf.reduce_mean(categorical_accuracy(y, sm_depth))
      

        # Create summaries for Tensorboard
        tf.summary.scalar("loss_cls", loss_cls)
        tf.summary.scalar("loss_perp", loss_perp)
        tf.summary.scalar("loss_perp_res3", loss_perp_list[0])
        tf.summary.scalar("loss_perp_res4", loss_perp_list[1])
        tf.summary.scalar("loss_perp_res5", loss_perp_list[2])
        tf.summary.scalar("loss_l2", loss_l2)
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("accuracy", accuracy)
        tf.summary.scalar("learning rate", decayed_lr)
        tf.summary.scalar("lambda", reg)
        summary_op = tf.summary.merge_all()

        name = str(lr) + '_' + str(bs) + '_' +  str(nn)
        train_writer = tf.summary.FileWriter(tensorboard_log + name + '/train/', graph = sess.graph)
        val_writer = tf.summary.FileWriter(tensorboard_log + name + '/val/')

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        #print(sess.run(trainable_variables_rnn[0]))


#        # Max norm
#        for idx_w, w in enumerate(trainable_variables_rnn):
#            trainable_variables_rnn[idx_w] = tf.assign(w, tf.clip_by_value(w, -0.0, 0.0))
#        for idx_w, w in enumerate(trainable_variables_conv1x1):
#            trainable_variables_conv1x1[idx_w] = tf.assign(w, tf.clip_by_value(w, -0.0, 0.0))

        #print(sess.run(trainable_variables_rnn[0]))

        # Load the pretrained weights into the non-trainable layer
        model_rgb.load_params(sess, params_dir_rgb, trainable=False)
        model_depth.load_params(sess, params_dir_depth, trainable=False)

        print("\nHyper-parameters: lr={}, #neurons={}, bs={}, l2={}, max_norm={}, dropout_rate={}".format(lr,nn,bs,aa,mn,do))     
        print("Number of trainable parameters = {}".format(count_params(trainable_variables_rnn)+count_params(trainable_variables_conv1x1)))    

        print("\n{} Generate features from training set".format(datetime.now()))
         
        tb_train_count=0        
        tb_val_count = 0

         
        # Loop over number of epochs
        num_samples = 0
        # Training set
        
        sess.run(training_init_op)

        # Progress bar setting
        bar = progressbar.ProgressBar(maxval=tr_batches_per_epoch, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        train_loss = 0
        for i in range(tr_batches_per_epoch):
            bar.update(i+1)
            tb_train_count+=1
            #print("batch {} / {}".format(i, train_batches_per_epoch))
            rgb_batch, depth_batch, label_batch = sess.run(next_batch)

            num_samples += np.shape(rgb_batch)[0]
            feed_dict = {x_rgb: rgb_batch, x_depth: depth_batch, y: label_batch, keep_prob: (1-do), training_phase: True, K.learning_phase(): 1}
            batch_loss, summary = sess.run([loss, summary_op], feed_dict=feed_dict)
            train_loss += batch_loss
            train_writer.add_summary(summary, tb_train_count)

        bar.finish()

        train_loss /= tr_batches_per_epoch #num_samples
        print("training loss = {}\n".format(train_loss))

        val_acc = 0
        val_loss = 0
        num_samples = 0  

        val_acc_rgb = 0
        val_acc_depth = 0

        sess.run(validation_init_op)
        for i in range(val_batches_per_epoch):

            tb_val_count+=1
            #print("batch {} / {}".format(i, val_batches_per_epoch))
            rgb_batch, depth_batch, label_batch = sess.run(next_batch)
            num_samples += np.shape(rgb_batch)[0]

            feed_dict = {x_rgb: rgb_batch, x_depth: depth_batch, y: label_batch, keep_prob: 1.0, training_phase: False, K.learning_phase(): 0}
            #batch_loss, batch_acc, summary = sess.run([loss, accuracy, summary_op], feed_dict=feed_dict)
            batch_loss, batch_acc, batch_acc_rgb, batch_acc_depth, summary = sess.run([loss, accuracy, accuracy_rgb, accuracy_depth, summary_op], feed_dict=feed_dict)


            val_acc_rgb +=batch_acc_rgb
            val_acc_depth +=batch_acc_depth

            val_loss+=batch_loss
            val_acc+=batch_acc
            val_writer.add_summary(summary, tb_val_count*(tr_batches_per_epoch/val_batches_per_epoch))

        val_loss /= val_batches_per_epoch #num_samples
        val_acc /= val_batches_per_epoch #num_samples

        val_acc_rgb /= val_batches_per_epoch
        val_acc_depth /= val_batches_per_epoch

        print("\n{} Validation loss : {}, Validation Accuracy = {:.4f}, RGB Accuracy = {:.4f}, Depth Accuracy = {:.4f}".format(datetime.now(), val_loss, val_acc, val_acc_rgb, val_acc_depth))
        


        # Loop over number of epochs
        for epoch in range(num_epochs):
            num_samples = 0
                 
            # Training set
            print("\nEpoch: {}/{}".format(epoch + 1, num_epochs))
            sess.run(training_init_op)

            # Progress bar setting
            bar = progressbar.ProgressBar(maxval=tr_batches_per_epoch, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
            bar.start()
            train_loss = 0
            for i in range(tr_batches_per_epoch):
                bar.update(i+1)
                tb_train_count+=1
                #print("batch {} / {}".format(i, train_batches_per_epoch))
                rgb_batch, depth_batch, label_batch = sess.run(next_batch)

                num_samples += np.shape(rgb_batch)[0]

                feed_dict = {x_rgb: rgb_batch, x_depth: depth_batch, y: label_batch, keep_prob: (1-do), training_phase: True, K.learning_phase(): 1}
                batch_loss, _, summary = sess.run([loss, train_step, summary_op], feed_dict=feed_dict)
                train_loss += batch_loss
     
                train_writer.add_summary(summary, tb_train_count)

            bar.finish()

            train_loss /= tr_batches_per_epoch #num_samples
            print("training loss = {}, {}\n".format(train_loss, tr_batches_per_epoch))

    
            if (epoch+1)%1 == 0:
                sess.run(validation_init_op)
                num_samples = 0
                val_loss = 0
                val_acc = 0

                for i in range(val_batches_per_epoch):

                    tb_val_count+=1
                    #print("batch {} / {}".format(i, val_batches_per_epoch))
                    rgb_batch, depth_batch, label_batch = sess.run(next_batch)

                    num_samples += np.shape(rgb_batch)[0]

                    feed_dict = {x_rgb: rgb_batch, x_depth: depth_batch, y: label_batch, keep_prob: 1.0, training_phase: False, K.learning_phase(): 0}
                    batch_loss, batch_acc, summary = sess.run([loss, accuracy, summary_op], feed_dict=feed_dict)

                    val_loss+=batch_loss
                    val_acc+=batch_acc
                    val_writer.add_summary(summary, tb_val_count*(tr_batches_per_epoch/val_batches_per_epoch))                

                val_loss /= val_batches_per_epoch #num_samples
                val_acc /= val_batches_per_epoch #num_samples

                print("\n{} Validation loss : {}, Validation Accuracy = {:.4f}".format(datetime.now(), val_loss, val_acc))

                log_epoch.append(epoch)
                log_train_loss.append(train_loss)
                log_val_loss.append(val_loss)
                log_val_acc.append(val_acc)

            # Early stopping for ill-posed params combination
            if ((epoch == 0) and (val_acc < 0.2)) or ((epoch == 9) and (val_acc < 0.5)) or np.isnan(train_loss):
                print("Training stopped due to poor results or divergence: validation loss = {}".format(val_acc))
                break
                

        history_callback = {log[0]:log_epoch, log[1]:log_train_loss, log[2]:log_val_loss, log[3]:log_val_acc}
        log_file(history_callback, hp)

    tf.reset_default_graph()
    shutil.rmtree(tensorboard_log)
