"""
Augmenters that apply mirroring/flipping operations to images.

Do not import directly from this file, as the categorization is not final.
Use instead
    `from imgaug import augmenters as iaa`
and then e.g. ::

    seq = iaa.Sequential([
        iaa.Fliplr((0.0, 1.0)),
        iaa.Flipud((0.0, 1.0))
    ])

List of augmenters:
    * Fliplr
    * Flipud
"""
from __future__ import print_function, division, absolute_import
from .. import imgaug as ia
# TODO replace these imports with iap.XYZ
from ..parameters import StochasticParameter, Deterministic, Binomial, Choice, DiscreteUniform, Normal, Uniform, FromLowerResolution
from .. import parameters as iap
from abc import ABCMeta, abstractmethod
import random
import numpy as np
import copy as copy_module
import re
import math
from scipy import misc, ndimage
from skimage import transform as tf, segmentation, measure
import itertools
import cv2
import six
import six.moves as sm
import types
import warnings

from .meta import Augmenter

class Fliplr(Augmenter):
    """
    Flip/mirror input images horizontally.

    Parameters
    ----------
    p : int or float or StochasticParameter, optional(default=0)
        Probability of each image to get flipped.

    name : string, optional(default=None)
        See `Augmenter.__init__()`

    deterministic : bool, optional(default=False)
        See `Augmenter.__init__()`

    random_state : int or np.random.RandomState or None, optional(default=None)
        See `Augmenter.__init__()`

    Examples
    --------
    >>> aug = iaa.Fliplr(0.5)

    would horizontally flip/mirror 50 percent of all input images.


    >>> aug = iaa.Fliplr(1.0)

    would horizontally flip/mirror all input images.

    """

    def __init__(self, p=0, name=None, deterministic=False, random_state=None):
        super(Fliplr, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        if ia.is_single_number(p):
            self.p = Binomial(p)
        elif isinstance(p, StochasticParameter):
            self.p = p
        else:
            raise Exception("Expected p to be int or float or StochasticParameter, got %s." % (type(p),))

    def _augment_images(self, images, random_state, parents, hooks):
        nb_images = len(images)
        samples = self.p.draw_samples((nb_images,), random_state=random_state)
        for i in sm.xrange(nb_images):
            if samples[i] == 1:
                images[i] = np.fliplr(images[i])
        return images

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        nb_images = len(keypoints_on_images)
        samples = self.p.draw_samples((nb_images,), random_state=random_state)
        for i, keypoints_on_image in enumerate(keypoints_on_images):
            if samples[i] == 1:
                width = keypoints_on_image.shape[1]
                for keypoint in keypoints_on_image.keypoints:
                    keypoint.x = (width - 1) - keypoint.x
        return keypoints_on_images

    def get_parameters(self):
        return [self.p]


class Flipud(Augmenter):
    """
    Flip/mirror input images vertically.

    Parameters
    ----------
    p : int or float or StochasticParameter, optional(default=0)
        Probability of each image to get flipped.

    name : string, optional(default=None)
        See `Augmenter.__init__()`

    deterministic : bool, optional(default=False)
        See `Augmenter.__init__()`

    random_state : int or np.random.RandomState or None, optional(default=None)
        See `Augmenter.__init__()`

    Examples
    --------
    >>> aug = iaa.Flipud(0.5)

    would vertically flip/mirror 50 percent of all input images.

    >>> aug = iaa.Flipud(1.0)

    would vertically flip/mirror all input images.

    """

    def __init__(self, p=0, name=None, deterministic=False, random_state=None):
        super(Flipud, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        if ia.is_single_number(p):
            self.p = Binomial(p)
        elif isinstance(p, StochasticParameter):
            self.p = p
        else:
            raise Exception("Expected p to be int or float or StochasticParameter, got %s." % (type(p),))

    def _augment_images(self, images, random_state, parents, hooks):
        nb_images = len(images)
        samples = self.p.draw_samples((nb_images,), random_state=random_state)
        for i in sm.xrange(nb_images):
            if samples[i] == 1:
                images[i] = np.flipud(images[i])
        return images

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        nb_images = len(keypoints_on_images)
        samples = self.p.draw_samples((nb_images,), random_state=random_state)
        for i, keypoints_on_image in enumerate(keypoints_on_images):
            if samples[i] == 1:
                height = keypoints_on_image.shape[0]
                for keypoint in keypoints_on_image.keypoints:
                    keypoint.y = (height - 1) - keypoint.y
        return keypoints_on_images

    def get_parameters(self):
        return [self.p]

'''
class Rettangolo(Augmenter):
    def _augment_images(self, image,x,y,p_x,p_y):
    # x e y saranno in percentuale (valori da 0--1) ma il limite deve essere 0.5
    # p_x e p_x punto da cui parte il rettangolo (influenzeranno il limite del rettangolo)
    # di x e y

    x = x * 100
    y = y * 100

    #print ("le dim sono " , image.shape) # 256 224 
    lato_x = image.shape[1] #224 
    lato_y = image.shape[0] #256
    #print ("le ascisse sono " ,lato_x)  
    #print ("le ordinate sono ", lato_y)

    x_lunghezza = int ( round( ( x * lato_x ) / 100.0 ) ) 
    y_lunghezza = int ( round( ( y * lato_y ) / 100.0 ) )
    print x_lunghezza
    if ( (p_x + x_lunghezza) >= lato_x ): 
        x_lunghezza = x_lunghezza - (( p_x + x_lunghezza ) - lato_x )
    
    if ( (p_y + y_lunghezza) >= lato_y ): 
        y_lunghezza = y_lunghezza - (( p_y + y_lunghezza ) - lato_y )
    print x_lunghezza

    media = np.mean(image)
    print ("media" , media)


    for i in range(0,x_lunghezza):
        for j in range(0,y_lunghezza):
            #print image_mod.shape
            image[p_y+j][p_x+i][0] = np.mean(image[:][:][0])
            image[p_y+j][p_x+i][1] = np.mean(image[:][:][1])
            image[p_y+j][p_x+i][2] = np.mean(image[:][:][2])
    return image
'''


class Rettangolo(Augmenter):

    def __init__(self, x,y,p_x,p_y, name=None, deterministic=False, random_state=None):
        super(Rettangolo, self).__init__(name=name, deterministic=deterministic, random_state=random_state)
        self.x = x 
        self.y = y
        self.p_x = p_x
        self.p_y = p_y
        

    def _augment_images(self, images, random_state, parents, hooks):
        # x e y saranno in percentuale (valori da 0--1) ma il limite deve essere 0.5
    # p_x e p_x punto da cui parte il rettangolo (influenzeranno il limite del rettangolo)
    # di x e y
        x = self.x  
        y = self.y 
        p_x = self.p_x 
        p_y = self.p_y 

        nb_images = len(images)
        #samples = self.p.draw_samples((nb_images,), random_state=random_state)
        for i in sm.xrange(nb_images):
            image = images[i]
        

            x = x * 100
            y = y * 100

    #print ("le dim sono " , image.shape) # 256 224 
            lato_x = image.shape[1] #224 
            lato_y = image.shape[0] #256
    #print ("le ascisse sono " ,lato_x)  
    #print ("le ordinate sono ", lato_y)

            x_lunghezza = int ( round( ( x * lato_x ) / 100.0 ) ) 
            y_lunghezza = int ( round( ( y * lato_y ) / 100.0 ) )
        
            #print (x_lunghezza)

            if ( (p_x + x_lunghezza) >= lato_x ): 
                x_lunghezza = x_lunghezza - (( p_x + x_lunghezza ) - lato_x )
    
            if ( (p_y + y_lunghezza) >= lato_y ): 
                y_lunghezza = y_lunghezza - (( p_y + y_lunghezza ) - lato_y )
            #print ("",x_lunghezza)

            media = np.mean(image)
            #print ("media" , media)
            media_0 = np.mean(image[:][:][0])
            media_1 = np.mean(image[:][:][1])
            media_2 = np.mean(image[:][:][2])

            for ii in range(0,x_lunghezza):
                for j in range(0,y_lunghezza):
                    #print image_mod.shape
                    image[p_y+j][p_x+ii][0] = media_0
                    image[p_y+j][p_x+ii][1] = media_1
                    image[p_y+j][p_x+ii][2] = media_2
        return images

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        nb_images = len(keypoints_on_images)
        samples = self.p.draw_samples((nb_images,), random_state=random_state)
        for i, keypoints_on_image in enumerate(keypoints_on_images):
            if samples[i] == 1:
                width = keypoints_on_image.shape[1]
                for keypoint in keypoints_on_image.keypoints:
                    keypoint.x = (width - 1) - keypoint.x
        return keypoints_on_images

    def get_parameters(self):
        return [self.p]