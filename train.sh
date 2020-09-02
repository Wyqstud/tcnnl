#!/bin/bash
########################################################
# step 1
#python main_baseline.py --model_spatial_pool 'avg'
#
#python main_baseline.py --train_sampler 'random'
##
#python main_baseline.py --test_sampler 'dense'
##
#python main_baseline.py --transform_method 'interval'

#################################################################
# step 2
#python main_baseline.py --sampler_method 'random'
#
#python main_baseline.py --sampler_method 'fix'
#
#python main_baseline.py --triplet_distance 'euclidean'
#
#python main_baseline.py --test_distance 'euclidean'
#
#python main_baseline.py --is_cat 'no'

###################################################################
# step 3
#python main_baseline.py --model_spatial_pool 'avg'

python main_baseline.py --model_spatial_pool 'max' --sampler_method 'fix'

python main_baseline.py --model_spatial_pool 'avg' --sampler_method 'fix'

python main_baseline.py --model_spatial_pool 'max' --sampler_method 'random'

python main_baseline.py --model_spatial_pool 'avg' --sampler_method 'random'