#!/bin/scripts

#python main_baseline.py --arch STAM\
#                        --model_spatial_pool 'avg'\
#                        --model_temporal_pool 'avg'\
#                        --train_sampler 'Random_interval'\
#                        --test_sampler 'Begin_interval'\
#                        --transform_method 'consecutive'\
#                        --sampler_method 'fix'\
#                        --triplet_distance 'cosine'\
#                        --test_distance 'cosine'\
#                        --is_cat 'yes'\
#                        --feature_method 'cat'\
#                        --is_mutual_channel_attention 'yes'\
#                        --is_mutual_spatial_attention 'yes'\
#                        --is_appearance_channel_attention 'yes'\
#                        --is_appearance_spatial_attention 'yes'\
#                        --layer_num 3\
#                        --seq_len 8\
#                        --split_id 0

#python main_baseline.py --split_id 0 \
#                        --test_sampler dense \
#                        --is_cat 'no'
#
#python main_baseline.py --split_id 1\
#                        --test_sampler dense \
#                        --is_cat 'no'
#
python main_baseline.py --split_id 2\
                        --test_sampler dense \
                        --is_cat 'no'
#
python main_baseline.py --split_id 3\
                        --test_sampler dense \
                        --is_cat 'no'
#
#python main_baseline.py --split_id 1\
#                        --test_sampler Begin_interval \
#                        --is_cat 'no'\
#

python main_baseline.py --split_id 4\
                        --test_sampler dense \
                        --is_cat 'no'

python main_baseline.py --split_id 5\
                        --test_sampler dense \
                        --is_cat 'no'

python main_baseline.py --split_id 6\
                        --test_sampler dense \
                        --is_cat 'no'
#
python main_baseline.py --split_id 7\
                        --test_sampler dense \
                        --is_cat 'no'

python main_baseline.py --split_id 8\
                        --test_sampler dense \
                        --is_cat 'no'

python main_baseline.py --split_id 9\
                        --test_sampler dense \
                        --is_cat 'no'

#python main_baseline.py --split_id 0 \
#                        --test_sampler dense \
#                        --is_cat 'no'\
#                        --LabelSmooth 'no'
#
#python main_baseline.py --split_id 1\
#                        --test_sampler dense \
#                        --is_cat 'no'\
#                        --LabelSmooth 'no'
#
#python main_baseline.py --split_id 2\
#                        --test_sampler dense \
#                        --is_cat 'no'\
#                        --LabelSmooth 'no'
#
#python main_baseline.py --split_id 3\
#                        --test_sampler dense \
#                        --is_cat 'no'\
#                        --LabelSmooth 'no'
#
#python main_baseline.py --split_id 4\
#                        --test_sampler dense \
#                        --is_cat 'no'\
#                        --LabelSmooth 'no'
#
#python main_baseline.py --split_id 5\
#                        --test_sampler dense \
#                        --is_cat 'no'\
#                        --LabelSmooth 'no'
#
#python main_baseline.py --split_id 6\
#                        --test_sampler dense \
#                        --is_cat 'no'\
#                        --LabelSmooth 'no'
#
#python main_baseline.py --split_id 7\
#                        --test_sampler dense \
#                        --is_cat 'no'\
#                        --LabelSmooth 'no'
#
#python main_baseline.py --split_id 8\
#                        --test_sampler dense \
#                        --is_cat 'no'\
#                        --LabelSmooth 'no'
#
#python main_baseline.py --split_id 9\
#                        --test_sampler dense \
#                        --is_cat 'no'\
#                        --LabelSmooth 'no'