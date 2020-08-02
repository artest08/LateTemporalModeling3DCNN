#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 02:07:51 2019

@author: esat
"""

import os, sys
import collections
import numpy as np
import cv2
import math
import random
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from sklearn.metrics import confusion_matrix

datasetFolder="../../datasets"
sys.path.insert(0, "../../")

import models
from VideoSpatialPrediction3D import VideoSpatialPrediction3D

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

dataset_names = sorted(name for name in datasets.__all__)

parser = argparse.ArgumentParser(description='PyTorch Two-Stream Action Recognition RGB Test Case')

parser.add_argument('--dataset', '-d', default='hmdb51',
                    choices=["ucf101", "hmdb51", "window"],
                    help='dataset: ucf101 | hmdb51')
parser.add_argument('--arch', '-a', metavar='ARCH', default='rgb_resneXt3D64f101_bert10_FRAB',
                    choices=model_names)

parser.add_argument('-s', '--split', default=2, type=int, metavar='S',
                    help='which split of data to work on (default: 1)')

parser.add_argument('-w', '--window', default=14, type=int, metavar='V',
                    help='validation file index (default: 3)')

parser.add_argument('-t', '--tsn', dest='tsn', action='store_true',
                    help='TSN Mode')

parser.add_argument('-v', '--val', dest='window_val', action='store_true',
                    help='Window Validation Selection')


multiGPUTest = False
multiGPUTrain = False
ten_crop_enabled = False
num_seg=16
num_seg_3D=1

result_dict = {}

def buildModel(model_path,num_categories):
    params = torch.load(model_path)
    if not '3D' in args.arch:
        model=models.__dict__[args.arch](modelPath='', num_classes=num_categories,length=num_seg)
    else:
        model=models.__dict__[args.arch](modelPath='', num_classes=num_categories,length=num_seg_3D)
    if args.tsn:
        new_dict = {k[7:]: v for k, v in params['state_dict'].items()} 
        model_dict=model.state_dict() 
        model_dict.update(new_dict)
        model.load_state_dict(model_dict)
    elif multiGPUTest:
        model=torch.nn.DataParallel(model)
        new_dict={"module."+k: v for k, v in params['state_dict'].items()} 
        model.load_state_dict(new_dict)
        
    elif multiGPUTrain:
        new_dict = {k[7:]: v for k, v in params['state_dict'].items()} 
        model_dict=model.state_dict() 
        model_dict.update(new_dict)
        model.load_state_dict(model_dict)
    else:
        model.load_state_dict(params['state_dict'])
    model.cuda()
    model.eval()  
    return model


def main():
    global args
    args = parser.parse_args()
    if args.tsn:    
        modelLocation="./checkpoint/"+args.dataset+"_tsn_"+args.arch+"_split"+str(args.split)
    else:
        modelLocation="./checkpoint/"+args.dataset+"_"+args.arch+"_split"+str(args.split)

    model_path = os.path.join('../../',modelLocation,'model_best.pth.tar') 
    
    if args.dataset=='ucf101':
        frameFolderName = "ucf101_frames"
    elif args.dataset=='hmdb51':
        frameFolderName = "hmdb51_frames"
    elif args.dataset=='smtV2':
        frameFolderName = "smtV2_frames"
    elif args.dataset=='window':
        frameFolderName = "window_frames"
    data_dir=os.path.join(datasetFolder,frameFolderName)
    
    if '64f' in args.arch:
        length=64
    elif '32f' in args.arch:
        length=32
    elif '8f' in args.arch:
        length=8
    else:
        length=16

    #args.window_val = True
    if 'rgb' in args.arch:
        extension = 'img_{0:05d}.jpg'
        if args.window_val:
            val_fileName = "window%d.txt" %(args.window)
        else:
            if args.dataset=='window': 
                val_fileName = "test_rgb_split%d.txt" %(args.split)
            else:
                val_fileName = "val_rgb_split%d.txt" %(args.split)
    elif 'pose' in args.arch:
        extension = 'pose1_{0:05d}.jpg'
        if args.window_val:
            val_fileName = "window%d.txt" %(args.window)
        else:
            if args.dataset=='window': 
                val_fileName = "test_pose_split%d.txt" %(args.split)
            else:
                val_fileName = "val_pose_split%d.txt" %(args.split)
    elif 'flow' in args.arch:
        if args.dataset=='window': 
            val_fileName = "test_flow_split%d.txt" %(args.split)
        else:
            val_fileName = "val_flow_split%d.txt" %(args.split)
        if 'ucf101' in args.dataset or 'window' in args.dataset:
            extension = 'flow_{0}_{1:05d}.jpg'
        elif 'hmdb51' in args.dataset:
            extension = 'flow_{0}_{1:05d}'

    val_file=os.path.join(datasetFolder,'settings',args.dataset,val_fileName)
    
    start_frame = 0
    if args.dataset=='ucf101':
        num_categories = 101
    elif args.dataset=='hmdb51':
        num_categories = 51
    elif args.dataset=='smtV2':
        num_categories = 174
    elif args.dataset=='window':
        num_categories = 3

    model_start_time = time.time()
    spatial_net=buildModel(model_path,num_categories)
    model_end_time = time.time()
    model_time = model_end_time - model_start_time
    print("Action recognition model is loaded in %4.4f seconds." % (model_time))

    
    f_val = open(val_file, "r")
    val_list = f_val.readlines()
    print("we got %d test videos" % len(val_list))

    line_id = 1
    match_count = 0
    match_count_top3 = 0

    y_true=[]
    y_pred=[]
    timeList=[]
    #result_list = []
    for line in val_list:
        line_info = line.split(" ")
        clip_path = os.path.join(data_dir,line_info[0])
        duration = int(line_info[1])
        input_video_label = int(line_info[2]) 
        
        start = time.time()

        spatial_prediction = VideoSpatialPrediction3D(
            clip_path,
            spatial_net,
            num_categories,
            args.arch,
            start_frame,
            duration,
            length = length, 
            extension = extension,
            ten_crop = ten_crop_enabled)
            
        
        end = time.time()
        estimatedTime=end-start
        timeList.append(estimatedTime)
        
        pred_index, _, top3 = spatial_prediction
        
        print("Sample %d/%d: GT: %d, Prediction: %d" % (line_id, len(val_list), input_video_label, pred_index))
        print("Estimated Time  %0.4f" % estimatedTime)
        print("------------------")
        if pred_index == input_video_label:
            match_count += 1
            
        if input_video_label in top3:
            match_count_top3 += 1

        line_id += 1
        y_true.append(input_video_label)
        y_pred.append(pred_index)

        
    print(confusion_matrix(y_true,y_pred))

    print("Accuracy with mean calculation is %4.4f" % (float(match_count)/len(val_list)))
    print("top3 accuracy %4.4f" % (float(match_count_top3)/len(val_list)))
    print(modelLocation)
    print("Mean Estimated Time %0.4f" % (np.mean(timeList)))  
    print('multiple clips')
    if ten_crop_enabled:
        print('10 crops')
    else:
        print('single crop')
        
    if args.window_val:
        print("window%d.txt" %(args.window))
    
    resultDict={'y_true':y_true,'y_pred':y_pred}
    
    np.save('results/%s.npy' %(args.dataset+args.arch+"_split"+str(args.split)), resultDict) 

if __name__ == "__main__":
    main()

