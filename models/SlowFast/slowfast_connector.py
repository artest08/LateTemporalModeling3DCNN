#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 16:47:58 2020

@author: esat
"""

import yaml
from .slowfast.models.video_model_builder import SlowFast
import argparse
from .slowfast.utils.parser import load_config
from .slowfast.utils.checkpoint import load_checkpoint
import os
import torch.nn as nn
import torch

'''
_C.DATA.MEAN = [0.45, 0.45, 0.45]
# List of input frame channel dimensions.

_C.DATA.INPUT_CHANNEL_NUM = [3, 3]

# The std value of the video raw pixels across the R G B channels.
_C.DATA.STD = [0.225, 0.225, 0.225]
'''

dir_folder_name = os.path.dirname(os.path.realpath(__file__))
config_folder_name = os.path.join(dir_folder_name, 'configs','Kinetics')



def create_args():
    parser = argparse.ArgumentParser(description='Slowfast')
    #parser.add_argument('--data', metavar='DIR', default='./datasets/ucf101_frames',
    #                    help='path to dataset')
    parser.add_argument('--cfg_file', metavar='DIR', default=None,
                        help='path to datset setting files')
    
    parser.add_argument('--opts', metavar='DIR', default=None,
                        help='path to datset setting files')
    
    args = parser.parse_args()
    del parser
    return args
    
def slowfast_50(model_path):
    yaml_name = 'SLOWFAST_8x8_R50.yaml'
    yaml_file_name = os.path.join(config_folder_name, yaml_name)
    args = create_args()
    args.cfg_file = yaml_file_name
    cfg = load_config(args)
    model = SlowFast(cfg)
    if model_path != '':
        params = torch.load(model_path)
        model.load_state_dict(params['state_dict'])
    return model
    del args

#model = slowfast_50()