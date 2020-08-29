#!/bin/bash

#python main_baseline.py --model_spatial_pool 'avg'
#
#python main_baseline.py --train_sampler 'random'
##
#python main_baseline.py --test_sampler 'dense'
##
#python main_baseline.py --transform_method 'interval'

python main_baseline.py --sampler_method 'random'

python main_baseline.py --sampler_method 'fix'

