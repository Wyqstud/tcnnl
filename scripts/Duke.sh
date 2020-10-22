#!/bin/scripts

python main_baseline.py --arch STAM\
                        --model_spatial_pool 'avg'\
                        --model_temporal_pool 'avg'\
                        --train_sampler 'Random_interval'\
                        --test_sampler 'dense'\
                        --transform_method 'consecutive'\
                        --sampler_method 'random'\
                        --triplet_distance 'cosine'\
                        --test_distance 'cosine'\
                        --is_cat 'yes'\
                        --feature_method 'cat'\
                        --is_mutual_channel_attention True\
                        --is_mutual_spatial_attention True\
                        --is_appearance_channel_attention True\
                        --is_appearance_spatial_attention True\
                        --layer_num 3\
                        --seq_len 8