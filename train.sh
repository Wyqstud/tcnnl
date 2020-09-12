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

#python main_baseline.py --model_spatial_pool 'max' --sampler_method 'fix'
#
#python main_baseline.py --model_spatial_pool 'avg' --sampler_method 'fix'
#
#python main_baseline.py --model_spatial_pool 'max' --sampler_method 'random'

##################################################################3
# step 4
#python main_baseline.py --arch 'ResNet50' \
#                        --model_spatial_pool 'avg' \
#                        --train_sampler 'Random_choice'
#
#python main_baseline.py --arch 'tem_dense' \
#                        --model_spatial_pool 'avg' \
#                        --train_sampler 'Random_choice'

#python main_baseline.py --arch 'tem_dense' \
##                        --model_spatial_pool 'avg' \
##                        --sampler_method 'random'

#python main_baseline.py --arch 'ResNet50' \
#                        --model_spatial_pool 'avg' \
#                        --model_temporal_pool 'avg' \
#                        --train_sampler 'random' \
#                        --test_sampler 'dense' \
#                        --transform_method 'consecutive'\
#                        --triplet_distance 'euclidean'\
#                        --test_distance 'euclidean'\
#                        --is_cat 'no'
                        
#python main_baseline.py --arch 'tem_dense' \
#                        --model_spatial_pool 'avg' \
#                        --model_temporal_pool 'avg' \
#                        --train_sampler 'random' \
#                        --test_sampler 'dense' \
#                        --transform_method 'consecutive'\
#                        --triplet_distance 'euclidean'\
#                        --test_distance 'euclidean'\
#                        --is_cat 'no'

##################################################################3
# step 1
#python main_baseline.py --arch 'ResNet50'
#
#python main_baseline.py --arch 'STAM'\
#                        --is_mutual_channel_attention 'no'\
#                        --is_mutual_spatial_attention 'no'\
#                        --is_appearance_channel_attention 'no'\
#                        --is_appearance_spatial_attention 'no'\
#                        --layer_num 1

#python main_baseline.py --arch 'STAM'\
#                        --is_mutual_channel_attention 'no'\
#                        --is_mutual_spatial_attention 'no'\
#                        --is_appearance_channel_attention 'no'\
#                        --is_appearance_spatial_attention 'no'\
#                        --layer_num 2
#
#python main_baseline.py --arch 'STAM'\
#                        --is_mutual_channel_attention 'no'\
#                        --is_mutual_spatial_attention 'no'\
#                        --is_appearance_channel_attention 'no'\
#                        --is_appearance_spatial_attention 'no'\
#                        --layer_num 3

#python main_baseline.py --arch 'STAM'\
#                        --is_mutual_channel_attention 'no'\
#
#python main_baseline.py --arch 'STAM'\
#                        --is_mutual_spatial_attention 'no'

#python main_baseline.py --arch 'STAM'\
#                        --is_mutual_spatial_attention 'no'\
#                        --is_mutual_channel_attention 'no'

#python main_baseline.py --arch 'STAM'\
#                        --is_mutual_channel_attention 'no'\
#                        --is_mutual_spatial_attention 'no'\
#                        --is_appearance_channel_attention 'no'\
#                        --is_appearance_spatial_attention 'no'

#python main_baseline.py --arch 'STAM'\
#                        --layer_num '2'\
#                        --is_mutual_channel_attention 'yes'\
#                        --is_mutual_spatial_attention 'yes'\
#                        --is_appearance_channel_attention 'yes'\
#                        --is_appearance_spatial_attention 'yes'\
#                        --is_down_channel 'no'

#python main_baseline.py --arch 'STAM'\
#                        --layer_num '2'\
#                        --is_mutual_spatial_attention 'yes'\
#                        --is_mutual_channel_attention 'no'\
#                        --is_appearance_spatial_attention 'yes'\
#                        --is_appearance_channel_attention 'no'\
#                        --is_down_channel 'no'\
#                        --sampler_method 'random'

python main_baseline.py --arch 'STAM'

#python main_baseline.py --arch 'STAM'\
#                        --layer_num '3'\
#                        --is_mutual_spatial_attention 'yes'\
#                        --is_mutual_channel_attention 'yes'\
#                        --is_appearance_spatial_attention 'yes'\
#                        --is_appearance_channel_attention 'yes'\
#                        --sampler_method 'random'



