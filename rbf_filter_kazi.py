# -*- coding: utf-8 -*-
"""
Created on Mon Dec 7 19:14:46 2020

@author: Kazi T Shahid and Ioannis D Schizas

This code is used for the script "autoencoder_main.py"

This is the RBF Filter Layer, where the input is a hyperspectral datacube of a mixed pixel and its surrounding neighborhood, and the output is the weighted averaged vector for that neighborhood.


Inputs:
datacube = cube of hyperspectral pixels depicting a neighborhood. 
Dimensions of "datacube" are n*n*B, where n is an odd number, and B represents number of spectral bands

Weights:
betas_weights = weights for averaging. Dimensions are batch_size*n*n, which means this will be repeated B times throughout the datacube

If you wish to use this code, please cite the URL given above for the dataset, and also the URL where this code was downloaded from:
https://github.com/KaziTShahid/Nonlinear-Hyperspectral-Unmixing-Spatial-Filters-Autoencoder

"""


import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np
import numpy.matlib  as matlib


def rbf_neighborhood(datacube):

    datacube_reshape = tf.reshape(datacube,(datacube.shape[0],datacube.shape[1]*datacube.shape[2],datacube.shape[3]))
    centers = datacube[:, int((datacube.shape[1]-1)/2), int((datacube.shape[2]-1)/2) , :]
    centers_repmat = tf.repeat(centers[:, tf.newaxis, :], datacube_reshape.shape[1], axis=1)
    
    dists = tf.math.pow(tf.math.reduce_euclidean_norm(datacube_reshape-centers_repmat,axis=2),2)
    
    betas = (1/(tf.math.reduce_mean(dists)))**1
    betas = matlib.repmat(betas,dists.shape[0],dists.shape[1])
    
    return np.squeeze(betas)


class rbf_filter(Layer):
    
    def __init__(self,inp):
        
        self.inp = inp       
        self.initial_betas_shape = (inp.shape[0],inp.shape[1]*inp.shape[2])
        
        super(rbf_filter, self).__init__()
        
    def build(self, input_shape):
        
        self.betas_weights = self.add_weight("betas", shape = self.initial_betas_shape,initializer=tf.constant_initializer(rbf_neighborhood(self.inp)))        
        
        super().build(input_shape)

    
    def call(self, x):
        
        self.datacube_reshape = tf.reshape(x,(x.shape[0],x.shape[1]*x.shape[2],x.shape[3]))
        
        self.centers = x[:, int((x.shape[1]-1)/2), int((x.shape[2]-1)/2) , :]
        self.centers_repmat = tf.repeat(self.centers[:, tf.newaxis, :], self.datacube_reshape.shape[1], axis=1)
    
        self.dists = tf.math.pow(tf.math.reduce_euclidean_norm(self.datacube_reshape-self.centers_repmat,axis=2),2)
        
        
        self.rbf_dists = tf.math.exp(-1 * self.betas_weights * self.dists)   
        
        self.proportions = self.rbf_dists/ tf.keras.backend.repeat_elements(tf.math.reduce_sum(self.rbf_dists,axis=1,keepdims=True), self.datacube_reshape.shape[1],axis=1)
        self.proportions = tf.repeat(self.proportions[:,:,tf.newaxis], self.datacube_reshape.shape[2], axis=2)
        
        return tf.math.reduce_sum(self.datacube_reshape*self.proportions,axis=1)
        
