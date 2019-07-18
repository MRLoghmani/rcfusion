""" Containes a helper class for image input pipelines in tensorflow """

import tensorflow as tf
import numpy as np
from random import randint, choice

from tensorflow.data import Dataset
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor

class ImageDataHandler(object):
    """ Class to handle the input pipeline for image data """

    def __init__(self, txt_file, data_dir, params_dir, img_size, batch_size, num_classes, num_channels=3, shuffle=False,
                 random_crops = False, buffer_size=1000):

        self.num_classes = num_classes
        self.img_size = img_size
        self.img_mean_rgb = np.load(params_dir + '/ocid_mean_rgb++.npy')
        self.img_mean_depth = np.load(params_dir + '/ocid_mean_surfnorm++.npy')
        self.num_channels = num_channels
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.shuffle = shuffle
        self.random_crops = random_crops
        self.img_paths_rgb, self.img_paths_depth, self.labels, self.data_size = self._read_txt_file(txt_file, data_dir)
        self.data = self._create_dataset()


    def _read_txt_file(self, txt_file, data_dir):
        """ Returns list of image path and corresponding labels as tensors from txt_file """

        img_paths = []
        labels = []
        with open(txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                items = line.split(' ')
                img_paths.append(items[0])
                labels.append(int(items[1]))

        # WARNING: this initial shuffling is important
        # see link: https://kratzert.github.io/2017/06/15/example-of-tensorflows-new-input-pipeline.html
        data_size = len(labels)
        img_paths, labels = self._shuffle_lists(img_paths, labels, data_size)
        
        img_paths_rgb = []
        img_paths_depth = []
        for path in img_paths:
            img_paths_rgb.append(data_dir[0] + path)# + "rgbcrop.png")
            img_paths_depth.append(data_dir[1] + path)# + "depthcrop.png")

        img_paths_rgb = convert_to_tensor(img_paths_rgb, dtype=dtypes.string)
        img_paths_depth = convert_to_tensor(img_paths_depth, dtype=dtypes.string)
        labels = convert_to_tensor(labels, dtype=dtypes.int32)

        return img_paths_rgb, img_paths_depth, labels, data_size


    def _create_dataset(self):
        """ Generate dataset (in batches) """

        ## Dataset from tensor slices:
        # img_paths and labels are the tensors
        # each element 'i' of data will have two components: img_paths[i] and labels[i]
        data = Dataset.from_tensor_slices((self.img_paths_rgb, self.img_paths_depth, self.labels))

        ## Dataset.map
        # Dataset.map is the same as built-in python map but with parallelism options
        # create a new dataset by applying the function (_prepare_input) to each element of data
        data = data.map(self._prepare_input, num_parallel_calls=8).prefetch(100*self.batch_size)
        #data = data.map(self._prepare_input, num_threads=8, output_buffer_size=100*self.batch_size)

        if self.shuffle:
            data = data.shuffle(self.buffer_size)

        # Create a new dataset where each elements contains 'batch_size' elements of the previous dataset
        data = data.batch(self.batch_size)

        return data



    def _prepare_input(self, filename_rgb, filename_depth, label):
        """ Run common pre-processing steps to prepare ONE image """

        # Convert label number into one-hot-encoding
        one_hot = tf.one_hot(label, self.num_classes)

        # Load and pre-process the image
        img_string_rgb = tf.read_file(filename_rgb)
        img_string_depth = tf.read_file(filename_depth)
        img_decoded_rgb = tf.image.decode_png(img_string_rgb, channels=self.num_channels)
        img_decoded_depth = tf.image.decode_png(img_string_depth, channels=self.num_channels)
        img_resized_rgb = tf.image.resize_images(img_decoded_rgb, [256,256])
        img_resized_depth = tf.image.resize_images(img_decoded_depth, [256,256])
        #mean_img = tf.image.resize_images(IMG_MEAN, self.img_size)
        img_bgr_rgb = img_resized_rgb[:, :, ::-1] # RGB -> BGR
        img_bgr_depth = img_resized_depth[:, :, ::-1] # RGB -> BGR
        img_centered_rgb = tf.subtract(tf.cast(img_bgr_rgb, dtype=tf.float32), self.img_mean_rgb[:, :, ::-1]) #IMG_MEAN_RGB[:, :, ::-1]
        img_centered_depth = tf.subtract(tf.cast(img_bgr_depth, dtype=tf.float32), self.img_mean_depth[:, :, ::-1]) #IMG_MEAN_DEPTH[:, :, ::-1]
 
        img_resized_rgb = tf.image.resize_images(img_centered_rgb, self.img_size)
        img_resized_depth = tf.image.resize_images(img_centered_depth, self.img_size)
        
        """ 
        Data augmentation comes here.
        """
        rot_param = choice([0,1,2,3])
        vert_flip = choice([True, False])
        horiz_flip = choice([True, False])
        delta_brightness = choice([-1,+1])*randint(0, 25)

        img_resized_rgb = tf.image.rot90(img_resized_rgb, k=rot_param)
        img_resized_depth = tf.image.rot90(img_resized_depth, k=rot_param)

        #if vert_flip:
        #    img_resized_rgb = tf.image.flip_left_right(img_resized_rgb)
        #    img_resized_depth = tf.image.flip_left_right(img_resized_depth)

        #if horiz_flip:
        #    img_resized_rgb = tf.image.flip_up_down(img_resized_rgb)
        #    img_resized_depth = tf.image.flip_up_down(img_resized_depth)
        
        img_resized_rgb = tf.image.adjust_brightness(img_resized_rgb, delta_brightness)
        img_resized_depth = tf.image.adjust_brightness(img_resized_depth, delta_brightness)

        '''
        if (self.random_crops): # random crop
            offset = [randint(0, 256 - self.img_size[0]), randint(0, 256 - self.img_size[1])]
            #img_resized_rgb = tf.random_crop(img_centered_rgb, [self.img_size[0],self.img_size[1],self.num_channels])
            #img_resized_depth = tf.random_crop(img_centered_depth, [self.img_size[0],self.img_size[1],self.num_channels])
        else: # central crop
            offset = [int(round(float(256 - self.img_size[0])/2)), int(round(float(256 - self.img_size[1])/2))]

        print(offset)
            
        img_resized_rgb = tf.image.crop_to_bounding_box(img_centered_rgb,
                                                        offset[0], offset[1],
                                                        self.img_size[0], self.img_size[1])
        img_resized_depth = tf.image.crop_to_bounding_box(img_centered_depth,
                                                          offset[0], offset[1],
                                                          self.img_size[0], self.img_size[1])

        '''

        return img_resized_rgb, img_resized_depth, one_hot


    def _shuffle_lists(self, img_paths, labels, data_size):
        """ Dataset shuffling - joint shuffling of the list of image paths and labels."""

        tmp_img_paths = img_paths
        tmp_labels = labels
        permutation = np.random.permutation(data_size)
        img_paths = []
        labels = []
        for i in permutation:
            img_paths.append(tmp_img_paths[i])
            labels.append(tmp_labels[i])

        return img_paths, labels
