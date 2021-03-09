# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 22:41:38 2021

@author: Kazi T Shahid and Ioannis D. Schizas

This code is used for the paper ""

This will generate an autoencoder for unsupervised nonlinear hyperspectral unmixing, utilizing spatial information with filters and will also estimate the number of endmembers

If you wish to run the code as-is, download the "Pavia University" dataset from the URL below:
http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes

If you wish to add more datasets, add their names to the list called 'dataset_choices' in the subsection 'hyperparameters'
Also, specify the classes to choose to create the datasets in the subsection 'building data'
If you wish to use this code, please cite the URL given above for the dataset, and also the URL where this code was downloaded from:
https://github.com/KaziTShahid/Nonlinear-Hyperspectral-Unmixing-Spatial-Filters-Autoencoder

"""


def lin_mixing(ref_pixels,all_percentages,num_classes): # for the linear mixing model, "ref_pixels" are the endmembers, "all_percentages" are the abundances, "num_classes" would be number of endmembers
    
    all_pixels = []    
        
    for i in range(len(all_percentages)):
        
        abundances = all_percentages[i]
        current_pixel = np.zeros([1,len(ref_pixels[0])])
        
        for j in range(num_classes):        
            current_pixel += abundances[j]*ref_pixels[j]
        all_pixels.append(np.transpose(current_pixel))
        
    return all_pixels


def bilin_mixing(ref_pixels,all_percentages,num_classes,TAKE_REPEATING_PRODUCTS,gamma,upto_how_many_degrees):  # for the Fan and Bilinear mixing models
    
    classes = list(range(num_classes))
    pairs = []
    for i in range(2,upto_how_many_degrees+1):    
    
        if (TAKE_REPEATING_PRODUCTS==1): # TAKE_REPEATING_PRODUCTS = 0 means we are using the Fan model, TAKE_REPEATING_PRODUCTS = 1 implies the Bilinear model
             pairs += list(itertools.product(classes,repeat = i))
        else: pairs += list(itertools.combinations(classes, i))
    
    all_pixels = []    
        
    for i in range(len(all_percentages)):
        
        abundances = all_percentages[i]
        current_pixel = np.zeros([1,len(ref_pixels[0])])
                
        for j in range(num_classes):        
            current_pixel += abundances[j]*ref_pixels[j]
        for j in range(len(pairs)):
            
            pair = pairs[j]
            current_pixel += gamma * (abundances[pair[0]]*ref_pixels[pair[0]]) * (abundances[pair[1]]*ref_pixels[pair[1]])
        
        all_pixels.append(np.transpose(current_pixel))
    
    return all_pixels


def ppnm_mixing(ref_pixels,all_percentages,num_classes,b_s): # for the PPNM (Polynomial Post-Nonlinear Mixing) model
    
    all_pixels = []    
        
    for i in range(len(all_percentages)):
        
        abundances = all_percentages[i]
        current_pixel = np.zeros([1,len(ref_pixels[0])])
        
        for j in range(num_classes):        
            current_pixel += abundances[j]*ref_pixels[j]
        current_pixel += np.multiply(current_pixel,current_pixel) * b_s[i]
        all_pixels.append(np.transpose(current_pixel))
        
    return all_pixels


def rbf_neighborhood(section): 
    
    # in a neighborhood of pixels around a center, "section" is the vectorized form
    # of that neighborhood, so a neighborhood of n*n*B (B=# of spectral bands) is converted into "section" of 
    # dimension n^2*B, with the center of the neighborhood in the middle
    # this function makes a weighted average of "section", with higher weights placed
    # on pixels which are more similar to the center pixel, by measuing the RBF distance
    
    moving_average_window = section.shape[0]    
    
    dists = np.zeros(moving_average_window)
    for i in range(moving_average_window):
        dists[i] = (np.linalg.norm(section[int((section.shape[0]-1)/2),:]-section[i,:]))**2
        
    betas = (1/(np.mean(dists)))**2
    rbf_dists = np.zeros(moving_average_window)
    for i in range(moving_average_window):
        rbf_dists[i] = np.exp ( -1 * betas * (np.linalg.norm(section[int((section.shape[0]-1)/2),:]-section[i,:]))**2 )
    proportions = rbf_dists/ sum(rbf_dists)
    
    return proportions,betas,rbf_dists


def making_field_with_padding(all_pixels,crop_width,crop_height,num_classes,pixels_per_class,conv_window):
    
    # creates a synthetic field of crops based on the generated mixed pixels, with padding
    # the field size is crop_height*(crop_width*num_classes)*B (B=# of spectral bands), and then
    # padding is added on top of it, where the padding for each section of crop_height*crop_width
    # is an average of the pixels in that section
    
    
    
    all_pixels_field = np.zeros([crop_height,crop_width*num_classes,all_pixels.shape[1]])
    all_pixels_field_sec_avg = np.zeros([num_classes,all_pixels.shape[1]])
    
    for i in range(num_classes):
        all_pixels_field[:,i*crop_width:(i+1)*crop_width,:] = np.reshape(all_pixels[i*pixels_per_class:(i+1)*pixels_per_class,:],(crop_height,crop_width,all_pixels.shape[1]))
        all_pixels_field_sec_avg[i,:] = np.mean(all_pixels_field[:,i*crop_width:(i+1)*crop_width,:],axis=(0,1))
                 
    all_pixels_field = np.lib.pad(all_pixels_field, ((int((conv_window-1)/2),int((conv_window-1)/2)),(int((conv_window-1)/2),int((conv_window-1)/2)),(0,0)), 'constant', constant_values=(0))
    
    start_pos = 0
    for i in range(num_classes):
        
        if i == 0:
            zeros_locs1 = np.array(np.where(all_pixels_field[:,0:(int((conv_window-1)/2)),:]==0))
            zeros_locs2 = np.array(np.where(all_pixels_field[0:(int((conv_window-1)/2)),start_pos:start_pos+crop_width+(int((conv_window-1)/2)),:]==0))
            zeros_locs3 = np.array(np.where(all_pixels_field[all_pixels_field.shape[0]-(int((conv_window-1)/2)):all_pixels_field.shape[0],start_pos:start_pos+crop_width+(int((conv_window-1)/2)),:]==0))
            zeros_locs3[0,:] += all_pixels_field.shape[0]-(int((conv_window-1)/2))
            zeros_locs = np.concatenate((zeros_locs1,zeros_locs2,zeros_locs3),axis=1)
            all_pixels_field[zeros_locs[0,:],zeros_locs[1,:],:] = all_pixels_field_sec_avg[i,:]
            start_pos += crop_width+(int((conv_window-1)/2))
        elif i>0 and i<(num_classes-1):
            zeros_locs1 = np.array(np.where(all_pixels_field[0:(int((conv_window-1)/2)),start_pos:start_pos+crop_width,:]==0))
            zeros_locs2 = np.array(np.where(all_pixels_field[all_pixels_field.shape[0]-(int((conv_window-1)/2)):all_pixels_field.shape[0],start_pos:start_pos+crop_width,:]==0))
            zeros_locs2[0,:] += all_pixels_field.shape[0]-(int((conv_window-1)/2))
            zeros_locs1[1,:] += start_pos
            zeros_locs2[1,:] += start_pos
            zeros_locs = np.concatenate((zeros_locs1,zeros_locs2),axis=1)
            all_pixels_field[zeros_locs[0,:],zeros_locs[1,:],:] = all_pixels_field_sec_avg[i,:]
            start_pos += crop_width
        elif i == (num_classes-1):                  
            zeros_locs1 = np.array(np.where(all_pixels_field[:,all_pixels_field.shape[1]-(int((conv_window-1)/2)):all_pixels_field.shape[1],:]==0))
            zeros_locs2 = np.array(np.where(all_pixels_field[0:(int((conv_window-1)/2)),start_pos:start_pos+crop_width+(int((conv_window-1)/2)),:]==0))
            zeros_locs3 = np.array(np.where(all_pixels_field[all_pixels_field.shape[0]-(int((conv_window-1)/2)):all_pixels_field.shape[0],start_pos:start_pos+crop_width+(int((conv_window-1)/2)),:]==0))
            zeros_locs2[0,:] += all_pixels_field.shape[0]-(int((conv_window-1)/2))
            zeros_locs1[1,:] += start_pos+crop_width
            zeros_locs2[1,:] += start_pos
            zeros_locs3[1,:] += start_pos
            zeros_locs = np.concatenate((zeros_locs1,zeros_locs2,zeros_locs3),axis=1)
            all_pixels_field[zeros_locs[0,:],zeros_locs[1,:],:] = all_pixels_field_sec_avg[i,:]
            start_pos += crop_width+(int((conv_window-1)/2))               
                        
    return all_pixels_field


def cov_matrix_calculation(all_pixels,C,scaling_factor,num_of_eigs,betas_kmeans):
    
    cov_matrix = np.exp(-1 * betas_kmeans * scaling_factor * np.sum((np.transpose(C-np.transpose(all_pixels)) )**2,axis=1))    
    vals, vecs = eigsh(cov_matrix, k=num_of_eigs,return_eigenvectors=True)
    vals = vals[::-1]
    vecs = vecs[:,::-1]
    rec_errors = np.zeros(len(vals))
    rec_errors_diffs = np.zeros(len(vals)-1)
    rec_errors_diffs_diffs = np.zeros(len(vals)-2)
    
    all_rec_cov_matrix = np.zeros([len(vals),cov_matrix.shape[0],cov_matrix.shape[0]])
    
    for i in range(len(vals)):                    
        rec_cov_matrix = np.zeros(cov_matrix.shape)
        for j in range(i+1):
            temp_vec = np.expand_dims(vecs[:,j],-1)
            rec_cov_matrix += vals[j]*np.matmul(temp_vec,np.transpose(temp_vec))
            
        all_rec_cov_matrix[i,:,:] = rec_cov_matrix
        
        rec_errors[i] = np.linalg.norm(cov_matrix-rec_cov_matrix,'fro') 
        if i>0:
            rec_errors_diffs[i-1] = rec_errors[i-1]-rec_errors[i]
        if i>1:
            rec_errors_diffs_diffs[i-2] = rec_errors_diffs[i-2]-rec_errors_diffs[i-1]
    
    est_num_classes = np.where(rec_errors_diffs_diffs==max(rec_errors_diffs_diffs))[0][0]+2
    
    return cov_matrix,est_num_classes,rec_errors,rec_errors_diffs,rec_errors_diffs_diffs



def rmse_measure(actual_matrix,estimated_matrix):
    
    # measures RMSE (Root Mean Square Error) for measuring endmember accuracy
    
    actual_dim = actual_matrix.shape[1]
    estimated_dim = estimated_matrix.shape[1]    
    smallest_rmse = np.zeros(actual_dim)
    rejects = []
    
    for i in range(actual_dim):
        
        if i>0 and i % estimated_dim == 0:
            rejects = []
        
        rmse_values = np.sum((np.transpose(np.transpose(estimated_matrix) - actual_matrix[:,i]))**2,axis=0)
        smallest_rmse_loc = np.argsort(rmse_values)
        for j in smallest_rmse_loc:
            if j not in rejects:
                smallest_rmse[i] = rmse_values[smallest_rmse_loc[j]]
                rejects.append(j)
                break
        
    return sum(smallest_rmse)
        

def sad_measure(actual_matrix,estimated_matrix):
    
    # measures SAD (Spectral Angle Divergence) for measuring endmember accuracy
    
    actual_dim = actual_matrix.shape[0]
    estimated_dim = estimated_matrix.shape[0]    
    smallest_sad = np.zeros(actual_dim)
    rejects = []
    
    for i in range(actual_dim):
        
        if i>0 and i % estimated_dim == 0:
            rejects = []
        
        sad_values = np.arccos(np.matmul(estimated_matrix,actual_matrix[i,:]) / ( np.linalg.norm(actual_matrix[i,:]) * np.linalg.norm(estimated_matrix,axis=1) ))
        smallest_sad_loc = np.argsort(sad_values)
        for j in smallest_sad_loc:
            if j not in rejects:
                smallest_sad[i] = sad_values[smallest_sad_loc[j]]
                rejects.append(j)
                break
        
    return sum(smallest_sad)
        
        
import math
import scipy.io
import random
import tensorflow as tf
import logging
logging.getLogger('tensorflow').disabled = True

###### This code is handy in limiting your GPU memory, if the current dataset is too large for your GPU to handle

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     # Currently, memory growth needs to be the same across GPUs
#     for gpu in gpus:
#       # tf.config.experimental.set_memory_growth(gpu, True)
#       tf.config.experimental.set_virtual_device_configuration(gpus[0],
#             [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Memory growth must be set before GPUs have been initialized
#     print(e)

import gc
gc.collect()

import time
import numpy as np
import numpy.matlib as matlib
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.sparse.linalg import svds, eigsh
from VCA import vca
import itertools
# import pickle

from scipy.signal import savgol_filter

from sklearn.cluster import KMeans
from rbf_kazi import RBFLayer
from rbf_filter_kazi import rbf_filter
from nonlin_layer_kazi import NONLIN_Layer
from ppnm_layer_kazi import PPNM_Layer


#%% Hyperparameters

iterations = 3 #how many times the synthetic data will be generated again
timer = 0 # calculating how much total time the network takes for all iterations, across all datasets

upto_how_many_degrees = 2 #upto how many degree cross-product terms to consider
if upto_how_many_degrees < 2:
    raise Exception('upto_how_many_degrees has to be equal to at least 2')


b_s_lb = -0.3 # lower bound for scaling factor with ppnm method
b_s_ub = 0.3 # upper bound for scaling factor with ppnm method

optimizer = tf.optimizers.Adam(learning_rate=0.0001)
num_epochs = 100

SNR_values = [0,5,10,15,20]
dead_pixel_percentages = [0,10,20] # what percentage of pixel entries will be "dead", meaning their values will be 0

crop_width = 20
crop_height = 20
pixels_per_class = crop_width*crop_height # how many mixed pixels will have majority abundance of each class

gamma = 1 # scaling factor for bilinear model (setting gamma=0 is the same as making Linear Mixing Model)
main_material_percentage_max = 90 #upper bound for majority abundance of one class
main_material_percentage_min = 80 #lower bound for majority abundance of one class

actual_num_classes = 4
total_pixels = actual_num_classes*pixels_per_class
                    
num_of_eigs = 10
conv_window = 5


false_condition_counter = 0 # counts all the times when the wrong number of endmembers were estimated
overall_counter = 0 # counts every iteration

patch_size = 5
if np.mod(patch_size,2)!=1:
    raise Exception('choose patch_size that is an odd number')

if main_material_percentage_max <= main_material_percentage_min:
    raise Exception('choose max that is higher than the min')


mixing_models = ['fan', 'bilin','ppnm']
dataset_choices = ['PaviaU']
        
rmse_values = np.zeros([len(dead_pixel_percentages),len(mixing_models),len(dataset_choices),len(SNR_values),iterations])    
all_sad_rec = np.zeros([len(dead_pixel_percentages),len(mixing_models),len(dataset_choices),len(SNR_values),iterations])    # SAD values of endmembers of all cases
all_sad_vca_before = np.zeros([len(dead_pixel_percentages),len(mixing_models),len(dataset_choices),len(SNR_values),iterations]) # SAD values of endmembers of all cases for VCA before applying the averaging filter
all_sad_vca_after = np.zeros([len(dead_pixel_percentages),len(mixing_models),len(dataset_choices),len(SNR_values),iterations])  # SAD values of endmembers of all cases for VCA after applying the averaging filter  
all_est_abundances = [] # save all abundance estimations in a list
false_condition_flags = np.zeros([len(dead_pixel_percentages),len(mixing_models),len(dataset_choices),len(SNR_values),iterations]) # keeps track of the times where the wrong number of endmembers were estimated
every_est_num_classes = np.zeros([len(dead_pixel_percentages),len(mixing_models),len(dataset_choices),len(SNR_values),iterations]) # keeps track of all the endmembers' numbers' estimations
divergence_flags = np.zeros([len(dead_pixel_percentages),len(mixing_models),len(dataset_choices),len(SNR_values),iterations])    # keeps track of the times where the neural network diverged (if it ever happens)
all_est_endmembers = [] # save all endmember estimations in a list

for dead_pixel_percentage_index in range(len(dead_pixel_percentages)):     
    
    temp_est_abundances1 = []
    temp_est_endmembers1 = []
        
    for mix_model_index in range(len(mixing_models)):
            
        mixing_model = mixing_models[mix_model_index]
        unmixing_layer = mixing_model
        
        if mixing_model == 'fan':
            TAKE_REPEATING_PRODUCTS = 0
        elif mixing_model == 'bilin': 
            TAKE_REPEATING_PRODUCTS = 1
        
        #%% choosing dataset
        
        temp_est_abundances2 = []
        temp_est_endmembers2 = []
    
        for dataset_index in range(len(dataset_choices)):
                
            dataset_choice = dataset_choices[dataset_index]
    
    
            #%% building data
            
            ground_truth = scipy.io.loadmat(dataset_choice+'_gt.mat')
            data = scipy.io.loadmat(dataset_choice+'.mat')
            
            if dataset_choice == 'PaviaU':
                dataset_gt = ground_truth['paviaU_gt']
                dataset = data['paviaU']
                
                data_to_choose = [1,4,5,9]  #chooses which classes to take reference pixels from
                
            #%% automatically determined parameters        
    
            dims = dataset.shape                    
            
            if (actual_num_classes != len(data_to_choose)) or (total_pixels != pixels_per_class*actual_num_classes):
                raise Exception('num_classes or total_pixels not consistent')
            
            if upto_how_many_degrees > actual_num_classes:
                raise Exception('upto_how_many_degrees cannot exceed value of num_classes')
            
            kmeans_dists = np.zeros([total_pixels,len(SNR_values)])
            inter_dists = np.zeros([len(SNR_values),actual_num_classes,actual_num_classes])
            
            dataset = dataset/np.max(dataset) #normalizing data to unit peak
            
            temp_est_abundances3 = []
            temp_est_endmembers3 = []
            
            for all_index in range(iterations):
                
                print('['+str(dead_pixel_percentage_index+1)+' '+str(mix_model_index+1)+' '+str(dataset_index+1)+' '+str(all_index+1)+' '+'] out of ['+str(len(dead_pixel_percentages))+' '+str(len(mixing_models))+' '+str(len(dataset_choices))+' '+str(iterations)+' '+']')
            
                #%% finding reference pixels
            
                
                if dataset_choice != 'Cuprite': 
                
                    all_pixel_locations = []
                    ref_pixels = []
                    for i in range(actual_num_classes):
                        current_class = data_to_choose[i]
                        current_class_locations = np.where(dataset_gt==current_class)
                        x = current_class_locations[0]
                        y = current_class_locations[1]
                        random.seed(all_index*i)
                        loc = random.randint(0,len(x)-1)
                        ref_pixels.append(dataset[x[loc],y[loc],:])
        
                else: 
                    ref_pixels = []
                    for i in range(actual_num_classes):
                        ref_pixels.append(dataset[data_to_choose[i]-1,:])
                        
                        
                #%% generating mixed pixels
    
                all_percentages = []
                
                tally = 0
                for i in range(actual_num_classes):
                    for j in range(pixels_per_class):
                        
                        tally += 1
                        random.seed(all_index+tally)
                        main_percentage = random.randint(main_material_percentage_min,main_material_percentage_max)
                        remaining_percentages = np.zeros([actual_num_classes-1,1])
                
                        for k in range(actual_num_classes-1):
                            random.seed(all_index+tally+k)
                            remaining_percentages[k,0] = random.random()
                        remaining_percentages = remaining_percentages/sum(remaining_percentages)*(100-main_percentage)
                        current_percentages = np.zeros([actual_num_classes,1])
                        current_percentages[i] = main_percentage
                        current_percentages[list(set(list(range(actual_num_classes)))-set([i]))] = remaining_percentages
                        
                        current_percentages /= 100
                        all_percentages.append(np.squeeze(current_percentages))
                        
                
                    
                np.random.seed(all_index*1000) # can be replaced with iteration number afterwards
                b_s = np.random.uniform(b_s_lb, b_s_ub, len(all_percentages))
                
                #%% choice of mixture model        
    
                if mixing_model=='lin':
                    all_pixels = np.squeeze(lin_mixing(ref_pixels,all_percentages,actual_num_classes))
                elif mixing_model=='fan':
                    all_pixels = np.squeeze(bilin_mixing(ref_pixels,all_percentages,actual_num_classes,TAKE_REPEATING_PRODUCTS,gamma,upto_how_many_degrees))
                elif mixing_model=='bilin':
                    all_pixels = np.squeeze(bilin_mixing(ref_pixels,all_percentages,actual_num_classes,TAKE_REPEATING_PRODUCTS,gamma,upto_how_many_degrees))
                elif mixing_model=='ppnm':
                    all_pixels = np.squeeze(ppnm_mixing(ref_pixels,all_percentages,actual_num_classes,b_s))
                    
                all_percentages_array = np.array(all_percentages)                
    
                orig_data = all_pixels
                orig_ref_pixels = ref_pixels
                
                temp_est_abundances4 = []
                temp_est_endmembers4 = []
            
                for SNR_index in range(len(SNR_values)):
                    
                    noise_dB = SNR_values[SNR_index]
                    
                    all_pixels = orig_data
                    ref_pixels = orig_ref_pixels
                    
                    k = 1 / (10** (noise_dB/10) )
                    random.seed(all_index*SNR_index)
                    all_pixels = all_pixels + np.random.normal(scale=k*np.max(all_pixels), size=[total_pixels,ref_pixels[0].shape[0]])
                    
                    
                    #%% introducing dead pixels
                    
                    dead_pixel_percentage = dead_pixel_percentages[dead_pixel_percentage_index]                    
                    
                    total_entries = all_pixels.shape[0]*all_pixels.shape[1]       
                    random.seed(iterations)
                    dead_pixel_locations = random.sample(range(total_entries), int(total_entries*dead_pixel_percentage*0.01))
                    all_pixels = np.ndarray.flatten(all_pixels)
                    all_pixels[dead_pixel_locations] = 0
                    all_pixels = np.reshape(all_pixels,[total_pixels,dims[2]])
                    all_pixels_new = np.zeros(all_pixels.shape)
                    
                    
                    #%% vca for just after dead pixels, before filtering
                    
                    vca_values = vca(np.transpose(all_pixels),actual_num_classes,verbose = True,snr_input = 0)
                    vca_order = vca_values[1]
                    correct_order_vca = np.argsort(vca_order)
                    vca_endmembers = np.transpose(vca_values[0])        
                    vca_endmembers_before = vca_endmembers[correct_order_vca,:]                    
                    
                    
                    #%% conv filter for dead pixels   
                    
                    # creates a synthetic field 
                    
                    conv_field = making_field_with_padding(all_pixels,crop_width,crop_height,actual_num_classes,pixels_per_class,conv_window)
                    
                    ind = 0
                    for i in range(int((conv_window-1)/2),crop_width*actual_num_classes+( int((conv_window-1)/2) )):
                        for j in range(int((conv_window-1)/2),crop_height+( int((conv_window-1)/2) )):
                            
                            section = conv_field[j-int((conv_window-1)/2):j+conv_window-int((conv_window-1)/2),i-int((conv_window-1)/2):i+conv_window-int((conv_window-1)/2),:]
                            section = np.reshape(section,(conv_window**2,section.shape[2]))
                            window,conv_betas,conv_rbf_dists = rbf_neighborhood(section)
                            window = np.transpose(matlib.repmat(window,all_pixels.shape[1],1))
                            
                            all_pixels_new[ind,:] = np.sum(section*window,axis=0)
                            ind += 1
                            
                    all_pixels = all_pixels_new       
                            
                    
                    #%% svd on data     
                    
                    temp = np.transpose(all_pixels)
                    
                    U, s, Vh = linalg.svd(temp)
                    U, s, Vh = svds(temp,k=6)
                    Lowmixed = np.matmul(np.transpose(U),temp)
                    temp = np.matmul(U,Lowmixed)                         
                    all_pixels = np.transpose(temp)
                    
                    #%% making synthetic field    
                    
                    # this creates a crop field, "all_pixels_field" with the synthetically generated pixels, "all_pixels"
                    # the principle is the same as the previous subsection "conv filter for dead pixels"
                    
                    all_pixels_field = np.zeros([crop_height,crop_width*actual_num_classes,all_pixels.shape[1]])
                    all_pixels_field_sec_avg = np.zeros([actual_num_classes,all_pixels.shape[1]])
                    
                    for i in range(actual_num_classes):
                        all_pixels_field[:,i*crop_width:(i+1)*crop_width,:] = np.reshape(all_pixels[i*pixels_per_class:(i+1)*pixels_per_class,:],(crop_height,crop_width,all_pixels.shape[1]))
                        all_pixels_field_sec_avg[i,:] = np.mean(all_pixels_field[:,i*crop_width:(i+1)*crop_width,:],axis=(0,1))
                        
                    #%% datacube with padding
                    
                    # the crop field is then split into datacubes, so a field with n*m*B pixels (excluding padding)
                    # is used to create a dataset containing (n*m) datacubes of dimension x*x*B dimension each,
                    # so the total dimension is of (n*m)*x*x*B, where x*x is the neighborhood size
                    
                    
                    all_pixels_field = np.lib.pad(all_pixels_field, ((int((patch_size-1)/2),int((patch_size-1)/2)),(int((patch_size-1)/2),int((patch_size-1)/2)),(0,0)), 'constant', constant_values=(0))
                    
                    start_pos = 0
                    for i in range(actual_num_classes):
                        
                        if i==0 or i==(actual_num_classes-1):
                            zeros_locs = np.where(all_pixels_field[:,start_pos:start_pos+crop_width+(int((patch_size-1)/2)),:]==0)
                            all_pixels_field[zeros_locs[0],zeros_locs[1]+start_pos,:] = all_pixels_field_sec_avg[i,:]
                            start_pos += crop_width+(int((patch_size-1)/2))
                        elif i>0 and i<(actual_num_classes-1):                        
                            zeros_locs = np.where(all_pixels_field[:,start_pos:start_pos+crop_width,:]==0)
                            all_pixels_field[zeros_locs[0],zeros_locs[1]+start_pos,:] = all_pixels_field_sec_avg[i,:]
                            start_pos += crop_width
                    
                    datacube = np.zeros([crop_height*crop_width*actual_num_classes,patch_size,patch_size,all_pixels.shape[1]])                
                    
                    ind = 0
                    for i in range(int((patch_size-1)/2),crop_width*actual_num_classes+( int((patch_size-1)/2) )):
                        for j in range(int((patch_size-1)/2),crop_height+( int((patch_size-1)/2) )):
                            
                            datacube[ind,:,:,:] = all_pixels_field[j-int((patch_size-1)/2):j+patch_size-int((patch_size-1)/2),i-int((patch_size-1)/2):i+patch_size-int((patch_size-1)/2),:]
                            ind += 1    
                    
                    
                    #%% betas_init
                    
                    C = np.expand_dims(all_pixels, -1)                
                    covmat = np.sum((np.transpose(C-np.transpose(all_pixels)) )**2,axis=1)  # covariance matrix of all_pixels
                    betas_init = (np.mean(np.max(np.tril(covmat,-1),axis=1)))
                    
                    
                    #%% eigenvalue-based endmember number estimation
                    
                    scaling_factors = np.linspace(0.5,20,num=10) # the factors with which I shall scale the betas_init value to get the ideal kernel covariance matrix
                    cov_matrices = np.zeros([total_pixels,total_pixels,len(scaling_factors)])
                    all_est_num_classes = np.zeros(len(scaling_factors))
                    all_rec_errors = np.zeros([len(scaling_factors),num_of_eigs])
                    all_rec_errors_diffs = np.zeros([len(scaling_factors),num_of_eigs-1])
                    all_rec_errors_diffs_diffs = np.zeros([len(scaling_factors),num_of_eigs-2])
    
                    start_time = time.time()
                    for i in range(len(scaling_factors)):
                        cov_matrices[:,:,i], all_est_num_classes[i],all_rec_errors[i,:],all_rec_errors_diffs[i,:],all_rec_errors_diffs_diffs[i,:] = cov_matrix_calculation(all_pixels,C,scaling_factors[i],num_of_eigs,betas_init)
                    print('time for eigenvalue based endmember calculation for loop: ',time.time()-start_time)
                    
                    ###### finding num_classes through err_diff variance drop ######
                    
                    percentage_threshold = 5 / 100
                    variances = np.var(all_rec_errors_diffs,axis=0)
                    variances_diffs = abs(variances[:-1]-variances[1:])
                    
                    if len(set(np.where(variances<=percentage_threshold*max(variances))[0]) & set(np.where(variances_diffs<=percentage_threshold*max(variances_diffs))[0]) & set(list(range(1,num_of_eigs)))) != 0:
                        est_num_classes = min(set(np.where(variances<=percentage_threshold*max(variances))[0]) & set(np.where(variances_diffs<=percentage_threshold*max(variances_diffs))[0]) & set(list(range(1,num_of_eigs))))+1
                    else: est_num_classes = 2
                    
                    every_est_num_classes[dead_pixel_percentage_index,mix_model_index,dataset_index,SNR_index,all_index] = est_num_classes
                    
                    if (est_num_classes != actual_num_classes):
                        false_condition_flags[dead_pixel_percentage_index,mix_model_index,dataset_index,SNR_index,all_index] = 1
                        false_condition_counter += 1
    
                    #%% vca & kmeans
                    
                    vca_values = vca(np.transpose(all_pixels),actual_num_classes,verbose = True,snr_input = 0)
                    vca_order = vca_values[1]
                    correct_order_vca = np.argsort(vca_order)
                    vca_endmembers = np.transpose(vca_values[0])        
                    vca_endmembers_after = vca_endmembers[correct_order_vca,:]
                    
                    kmeans = KMeans(n_clusters=est_num_classes, random_state=0).fit(np.squeeze(all_pixels))
                    kmeans_centers = kmeans.cluster_centers_
                    kmeans_centers_backup = kmeans_centers
                    kmeans_labels = kmeans.labels_
                    
                    kmeans_centers_ordered = np.zeros(kmeans_centers.shape)
                    
                    ind = 0
                    
                    for i in range(est_num_classes):
                        section_avg = np.mean(all_pixels[ind:ind+int(total_pixels/est_num_classes),:],axis=0)
                        dists = np.sum((section_avg - kmeans_centers)**2,axis=1)
                        ok_loc = np.where(dists==min(dists))[0][0]
                        kmeans_centers_ordered[i,:] = kmeans_centers[ok_loc,:]
                        kmeans_centers = np.delete(kmeans_centers, ok_loc, 0)
                        ind += int(total_pixels/est_num_classes)
                        
                    #%% initializing betas
        
                    input_array = all_pixels
                    dists = np.zeros(total_pixels)
                    for i in range(total_pixels):
                        temp_dists = np.zeros(est_num_classes)
                        for j in range(est_num_classes):
                            temp_dists[j] = np.linalg.norm(input_array[i,:]-kmeans_centers_ordered[j,:])
                        dists[i] = np.min(temp_dists)
                    betas_kmeans = ((1/np.mean(dists)) ** 2 )                    
                    
                    #%% defining the autoencoder
                    
                    
                    class Nonlinear_Unmixing_AutoEndcoder(tf.keras.Model):
                            
                        def __init__(self):
                            super(Nonlinear_Unmixing_AutoEndcoder, self).__init__()
                            
                            self.flatten_layer = tf.keras.layers.Flatten()
                            
                            self.rbf_filter = rbf_filter(datacube)
                            
                            self.rbflayer = RBFLayer(est_num_classes, betas_kmeans, centers = kmeans_centers_ordered)
                            
                            self.nonlin_layer = NONLIN_Layer(TAKE_REPEATING_PRODUCTS,upto_how_many_degrees,initial_endmembers = kmeans_centers_ordered)
                            
                            self.ppnm_layer = PPNM_Layer(initial_endmembers = kmeans_centers_ordered)
                            
                        def call(self, inp):
                                
                            x1 = inp
                            
                            x1 = self.rbf_filter(x1)                            
                            
                            inp_reshape = inp[:,int((patch_size-1)/2),int((patch_size-1)/2),:]     # taking the center pixels 
                            
                            x1 = self.rbflayer(x1)
                            x1 = tf.divide(x1, tf.math.reduce_sum(x1,axis=1)[:, np.newaxis]) # normalizing to unit sum
                            
                            classes = list(range(est_num_classes))
                            
                            pairs=[]
                            for i in range(2,upto_how_many_degrees+1):    
                            
                                if (TAKE_REPEATING_PRODUCTS==1):
                                      pairs += list(itertools.product(classes,repeat = i))
                                else: pairs += list(itertools.combinations(classes, i))  
                            
                            x = tf.pad(x1, ((0,0),(0,len(pairs))), 'constant', constant_values=(0)) #padding
                            
                            mask = []
                            for i in range(len(pairs)):
        
                                pair = pairs[i]
                                abundance_cross_product = tf.ones(total_pixels)
                                for j in range(len(pair)): 
                                    abundance_cross_product = tf.math.multiply(abundance_cross_product, x[:,pair[j]])
                                    
                                mask.append(abundance_cross_product)
                            
                            mask_tf = tf.convert_to_tensor(mask)
                            mask_tf = tf.pad(mask_tf, ((est_num_classes,0),(0,0)), 'constant', constant_values=(0)) 
                            new_x = x + tf.transpose(mask_tf) #creating nonlinear abundance vector
                            
                            
                            if (unmixing_layer == 'bilin') or (unmixing_layer == 'fan'):
                                x = self.nonlin_layer(new_x)
                            elif (unmixing_layer == 'ppnm'):
                                x = self.ppnm_layer(x1)                            
                            
                            return x, inp_reshape, x1                        
                    
                    def loss(x, x_bar):
                        return tf.losses.mean_squared_error(x, x_bar)
                    def grad(model, inputs):
                        with tf.GradientTape() as tape:
                            reconstruction, inputs_reshaped, x1 = model(inputs)
                            loss_value = loss(inputs_reshaped, reconstruction)
                        return loss_value, tape.gradient(loss_value, model.trainable_variables), inputs_reshaped, reconstruction, x1
                    
                    divergence_counter = 0
                    flag = 0 # if divergence occurs, flag will stay 0 and the network will re-run repeatedly until divergence is avoided
                    
                    while (flag==0):
                        
                        #%% defining parameters
                        
                        model = Nonlinear_Unmixing_AutoEndcoder()
                        global_step = tf.Variable(0)
                        reconstructed_pixels = []
                        all_estimated_endmembers = []
                        norm_errors = []
                        norm_errors_endmembers = []
                        norm_errors_endmembers_smoothed = []
                        est_abundances = []
                        all_estimated_endmembers_rec = []
                        all_estimated_endmembers_centers = []
                        all_weights = []
                        
                        #%% solving done with batch_size a constant                        
                        
                        batch_size = total_pixels
                        
                        if np.mod(total_pixels,batch_size) != 0:
                            raise Exception('choose batch size that can split "total_pixels"')
                            
                        start_time = time.time()
                         
                        for epoch in range(num_epochs):
                            
                            if (np.mod(epoch,20)==0):
                                print("Epoch: ", epoch)
                                
                            for x in range(0, total_pixels, batch_size):                                
                                
                                x_inp = datacube[x : x + batch_size,:,:,:]
                                
                                loss_value, grads, inputs_reshaped, reconstruction, out_bottleneck = grad(model, x_inp)
                        
                                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                                
                                out_rec = np.squeeze(reconstruction)
                                reconstructed_pixels.append(out_rec)
                                weights = model.get_weights()
                                all_weights.append(weights)
                                estimated_endmembers_rec = weights[len(weights)-2]                                
                                
                                if epoch == num_epochs-1:
                                    est_abundances.append(np.squeeze(out_bottleneck))
                        
                                    norm_errors = []
                                    norm_errors_endmembers = []
                                    norm_errors_endmembers_smoothed = []
                        
                        print('elapsed time: ' +str(time.time()-start_time) )  
                        timer += time.time()-start_time
                        est_abundances = est_abundances[0]
                        
                        if np.isnan(np.sum(est_abundances)) or np.isnan(np.sum(estimated_endmembers_rec)):
                            print('abundance ',np.sum(est_abundances))
                            print('endmembers ',np.sum(estimated_endmembers_rec))
                            divergence_counter += 1
                            if divergence_counter == 10: # if divergence occurs 10 times in a row, a very erroneous estimate is created and the loop closes
                                flag = 1
                                est_abundances = np.ones(est_abundances.shape)/est_num_classes
                                estimated_endmembers_rec = np.ones(estimated_endmembers_rec.shape)
                                divergence_flags[dead_pixel_percentage_index,mix_model_index,dataset_index,SNR_index,all_index] = 1
                            
                        else:
                            flag=1
    

                    #%% calculating rmse (abundance)
                            
                    if est_num_classes == actual_num_classes:
                        rmse = (np.sum((est_abundances - all_percentages_array)**2) / (total_pixels*actual_num_classes) )**0.5
                    elif est_num_classes != actual_num_classes: 
                        rmse = (rmse_measure(all_percentages_array,est_abundances) / (total_pixels*actual_num_classes) )**0.5
                        
                    rmse_values[dead_pixel_percentage_index,mix_model_index,dataset_index,SNR_index,all_index] = rmse
                    
                    #%% calculating sads (endmembers)
                    
                    yhat = np.zeros(estimated_endmembers_rec.shape)
                    for i in range(est_num_classes):
                        yhat[i,:] = savgol_filter(estimated_endmembers_rec[i,:], 21, 8) # Savitzky-Golay filter to help with denoising endmember estimations
                        
                    if est_num_classes == actual_num_classes:
                        sads = sum(np.arccos(np.sum(np.array(ref_pixels)*yhat,axis=1) / (np.linalg.norm(ref_pixels,axis=1)*np.linalg.norm(yhat,axis=1))))/actual_num_classes
                    elif est_num_classes != actual_num_classes: 
                        sads = sad_measure(np.array(ref_pixels),yhat) / actual_num_classes                        
                    
                    all_sad_rec[dead_pixel_percentage_index,mix_model_index,dataset_index,SNR_index,all_index] = sads
                    
                    sad = 0
                    for i in range(actual_num_classes):
                        sad += math.acos( sum(np.multiply(ref_pixels[i],vca_endmembers_before[i,:])) / ( np.linalg.norm(ref_pixels[i]) * np.linalg.norm(vca_endmembers_before[i,:]) ) )
                    
                    sads = sad/actual_num_classes
                    
                    all_sad_vca_before[dead_pixel_percentage_index,mix_model_index,dataset_index,SNR_index,all_index] = sads
                    
                    
                    sad = 0
                    for i in range(actual_num_classes):
                        sad += math.acos( sum(np.multiply(ref_pixels[i],vca_endmembers_after[i,:])) / ( np.linalg.norm(ref_pixels[i]) * np.linalg.norm(vca_endmembers_after[i,:]) ) )
                    
                    sads = sad/actual_num_classes
                    
                    all_sad_vca_after[dead_pixel_percentage_index,mix_model_index,dataset_index,SNR_index,all_index] = sads
                    
                    
                    temp_est_abundances4.append(est_abundances)
                    temp_est_endmembers4.append(yhat)
                    
                    overall_counter += 1
                    
                temp_est_abundances3.append(temp_est_abundances4) 
                temp_est_endmembers3.append(temp_est_endmembers4) 
            
            temp_est_abundances2.append(temp_est_abundances3) 
            temp_est_endmembers2.append(temp_est_endmembers3) 
            
        temp_est_abundances1.append(temp_est_abundances2) 
        temp_est_endmembers1.append(temp_est_endmembers2) 
        
    all_est_abundances.append(temp_est_abundances1) 
    all_est_endmembers.append(temp_est_endmembers1) 


#%% printing results
            
rmse_mean = np.mean(rmse_values,axis=4)
all_sad_rec_mean = np.mean(all_sad_rec,axis=4)
all_sad_vca_before_mean = np.mean(all_sad_vca_before,axis=4)
all_sad_vca_after_mean = np.mean(all_sad_vca_after,axis=4)


for i in range(len(dead_pixel_percentages)):
    for j in range(len(mixing_models)):
        for k in range(len(dataset_choices)):
        
            print('RMSE of ', dataset_choices[k], ' for ', mixing_models[j], ' model for ', dead_pixel_percentages[i], 'percent dead pixels:', rmse_mean[i,j,k,:])
            print('SAD of ', dataset_choices[k], ' for ', mixing_models[j], ' model for ', dead_pixel_percentages[i], 'percent dead pixels:', all_sad_rec_mean[i,j,k,:])
            print('SAD of ', dataset_choices[k], ' for ', mixing_models[j], " model's VCA before filtering for ", dead_pixel_percentages[i], 'percent dead pixels:', all_sad_vca_before_mean[i,j,k,:])
            print('SAD of ', dataset_choices[k], ' for ', mixing_models[j], " model's VCA after filtering for ", dead_pixel_percentages[i], 'percent dead pixels:', all_sad_vca_after_mean[i,j,k,:])
        

print(false_condition_counter, ' out of ',overall_counter, 'iterations wrong # of endmembers guessed (', false_condition_counter/overall_counter*100,'%)' )        
print(np.sum(divergence_flags), ' out of ',overall_counter, 'iterations diverged (', np.sum(divergence_flags)/overall_counter*100,'%)' )        

#%% plotting results, comparing with VCA

for ii in range(len(dead_pixel_percentages)):
    
    dead_pixel_percentage = dead_pixel_percentages[ii]

    ind=0
    fig = plt.figure(figsize=(20,20)) #this works well for large plots with many subplots
    # fig = plt.figure() #this works well for smaller plots with few subplots
    fig.suptitle('plots_' + str(dead_pixel_percentage) + '_percent_dead_pixels_' + str(iterations) + '_iterations')
    for i in range(len(mixing_models)):
        for j in range(len(dataset_choices)):
            
            ind += 1
            ax = plt.subplot(len(mixing_models), len(dataset_choices), ind)
            ax.set_title(dataset_choices[j] + '_' + mixing_models[i])
            ax.plot(SNR_values, np.squeeze(rmse_mean[ii,i,j,:]),marker='D', label="RMSE")        
    
            ax.set_title(dataset_choices[j] + '_' + mixing_models[i])
            ax.plot(SNR_values, np.squeeze(all_sad_rec_mean[ii,i,j,:]),marker='D', label="SAD REC")
    
            ax.set_title(dataset_choices[j] + '_' + mixing_models[i])
            ax.plot(SNR_values, np.squeeze(all_sad_vca_before_mean[ii,i,j,:]),marker='D', label="SAD VCA BEFORE")
            
            ax.set_title(dataset_choices[j] + '_' + mixing_models[i])
            ax.plot(SNR_values, np.squeeze(all_sad_vca_after_mean[ii,i,j,:]),marker='D', label="SAD VCA AFTER")
            
            ax.legend(loc="upper right")
            
    
    figtitle = ('plots_' + str(dead_pixel_percentage) + '_percent_dead_pixels_'+ str(actual_num_classes) + '_classes_' + str(iterations) + '_iterations.pdf')
    # fig.savefig(figtitle)  #uncomment if you wish to save the figure to a file with the title written in the way above

#%% saving results to a file

## I made the title to contain the hyperparameters' values, this makes it easier to tune the values

# title = 'all_results_proposed_' + str(actual_num_classes) + '_classes_' + str(iterations)  + '_iters_'+str(percentage_threshold)+'_percent_thresold_across_dead_pixels.mat'
# with open(title, 'wb') as f:
#     pickle.dump([mixing_models, dataset_choices, rmse_values, all_sad_rec, all_sad_vca_before,all_sad_vca_after, false_condition_flags,dead_pixel_percentages,divergence_flags,scaling_factors,every_est_num_classes], f)
    
# with open(title, 'rb') as f:
#     [mixing_models, dataset_choices, rmse_values, all_sad_rec,  all_sad_vca_before,all_sad_vca_after, false_condition_flags,dead_pixel_percentages,divergence_flags,scaling_factors] = pickle.load(f)