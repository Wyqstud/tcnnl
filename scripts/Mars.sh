#!/bin/scripts

#python Train.py --arch 'STAM'\
#                        --dataset 'mars'\
#                        --model_spatial_pool 'avg'\
#                        --model_temporal_pool 'avg'\
#                        --train_sampler 'Random_interval'\
#                        --test_sampler 'Begin_interval'\
#                        --transform_method 'consecutive'\
#                        --sampler_method 'random'\
#                        --triplet_distance 'cosine'\
#                        --test_distance 'cosine'\
#                        --is_cat 'no'\
#                        --feature_method 'cat'\
#                        --is_mutual_channel_attention 'no'\
#                        --is_mutual_spatial_attention 'yes'\
#                        --is_appearance_channel_attention 'no'\
#                        --is_appearance_spatial_attention 'yes'\
#                        --layer_num 3 \
#                        --seq_len 8 \
#                        --is_down_channel 'yes'\
#                        --sampler 'RandomIdentitySampler'\
#                        --fix 'yes'

#python Train.py --arch 'STAM'\
#                        --dataset 'mars'\
#                        --model_spatial_pool 'avg'\
#                        --model_temporal_pool 'avg'\
#                        --train_sampler 'Random_interval'\
#                        --test_sampler 'Begin_interval'\
#                        --transform_method 'consecutive'\
#                        --sampler_method 'random'\
#                        --triplet_distance 'cosine'\
#                        --test_distance 'cosine'\
#                        --is_cat 'yes'\
#                        --feature_method 'cat'\
#                        --is_mutual_channel_attention 'yes'\
#                        --is_mutual_spatial_attention 'yes'\
#                        --is_appearance_channel_attention 'yes'\
#                        --is_appearance_spatial_attention 'yes'\
#                        --layer_num 3 \
#                        --seq_len 8 \
#                        --is_down_channel 'yes'\
#                        --sampler 'RandomIdentitySampler'\
#                        --fix 'yes'

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%test%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

python Test.py  --arch 'ResNet50'\
                --dataset 'mars'\
                --model_spatial_pool 'avg'\
                --model_temporal_pool 'avg'\
                --train_sampler 'Begin_interval'\
                --test_sampler 'Begin_interval'\
                --transform_method 'consecutive'\
                --sampler_method 'random'\
                --triplet_distance 'cosine'\
                --test_distance 'cosine'\
                --is_cat 'yes'\
                --feature_method 'cat'\
                --is_mutual_channel_attention 'yes'\
                --is_mutual_spatial_attention 'yes'\
                --is_appearance_channel_attention 'yes'\
                --is_appearance_spatial_attention 'yes'\
                --layer_num 3 \
                --seq_len 8 \
                --is_down_channel 'yes'\
                --print_gram True\
                --layer_name 'down_channel'\
                --test_path '/home/wyq/exp/ablation experiment/2020-11-26_10-54-01/rank1_0.8929348_checkpoint_ep420.pth'