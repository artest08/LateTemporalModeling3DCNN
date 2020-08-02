#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 15:14:46 2020

@author: esat
"""
import yaml
from slowfast.models.video_model_builder import SlowFast
import argparse
from slowfast.utils.parser import load_config
from slowfast.utils.checkpoint import load_checkpoint

'''
_C.DATA.MEAN = [0.45, 0.45, 0.45]
# List of input frame channel dimensions.

_C.DATA.INPUT_CHANNEL_NUM = [3, 3]

# The std value of the video raw pixels across the R G B channels.
_C.DATA.STD = [0.225, 0.225, 0.225]
'''

yaml_location = '/media/esat/6a7c4273-8106-47bc-b992-6760dfcea9a1/tsnCoffe/two-stream-pytorch/models/SlowFast/configs/Kinetics/SLOWFAST_8x8_R50.yaml'

path_to_checkpoint = '/media/esat/6a7c4273-8106-47bc-b992-6760dfcea9a1/tsnCoffe/two-stream-pytorch/weights/SLOWFAST_8x8_R50.pkl'
parser = argparse.ArgumentParser(description='PyTorch Two-Stream Action Recognition')
#parser.add_argument('--data', metavar='DIR', default='./datasets/ucf101_frames',
#                    help='path to dataset')
parser.add_argument('--cfg_file', metavar='DIR', default=yaml_location,
                    help='path to datset setting files')

parser.add_argument('--opts', metavar='DIR', default=None,
                    help='path to datset setting files')

args = parser.parse_args()

cfg = load_config(args)

model = SlowFast(cfg)

load_checkpoint(path_to_checkpoint,
    model,data_parallel=False, convert_from_caffe2=True)

