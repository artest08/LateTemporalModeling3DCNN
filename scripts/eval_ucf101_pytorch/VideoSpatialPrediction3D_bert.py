#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 14:57:05 2019

@author: esat
"""


import os
import sys
import numpy as np
import math
import cv2
import scipy.io as sio

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

sys.path.insert(0, "../../")
import video_transforms

soft=nn.Softmax(dim=1)
def VideoSpatialPrediction3D_bert(
        vid_name,
        net,
        num_categories,
        architecture_name,
        start_frame=0,
        num_frames=0,
        num_seg=4,
        length = 16,
        extension = 'img_{0:05d}.jpg',
        ten_crop = False
        ):

    if num_frames == 0:
        imglist = os.listdir(vid_name)
        newImageList=[]
        if 'rgb' in architecture_name or 'pose' in architecture_name:
            for item in imglist:
                if 'img' in item:
                   newImageList.append(item) 
        elif 'flow' in architecture_name:
            for item in imglist:
                if 'flow_x' in item:
                   newImageList.append(item) 
        duration = len(newImageList)
    else:
        duration = num_frames
    
    if 'rgb' in architecture_name:
        if 'I3D' in architecture_name:
            
            if not 'resnet' in architecture_name:
                clip_mean = [0.5, 0.5, 0.5] 
                clip_std = [0.5, 0.5, 0.5]
            else:
                clip_mean = [0.45, 0.45, 0.45]
                clip_std = [0.225, 0.225, 0.225] 
            normalize = video_transforms.Normalize(mean=clip_mean,
                                     std=clip_std)
            val_transform = video_transforms.Compose([
                    video_transforms.ToTensor(),
                    normalize,
                ])
            if '112' in architecture_name:
                scale = 0.5
            else:
                scale = 1
        elif 'MFNET3D' in architecture_name:
            clip_mean = [0.48627451, 0.45882353, 0.40784314]
            clip_std = [0.234, 0.234, 0.234] 
            normalize = video_transforms.Normalize(mean=clip_mean,
                                     std=clip_std)
            val_transform = video_transforms.Compose([
                    video_transforms.ToTensor(),
                    normalize])
            if '112' in architecture_name:
                scale = 0.5
            else:
                scale = 1
        elif 'tsm' in architecture_name:
            clip_mean = [0.485, 0.456, 0.406]
            clip_std = [0.229, 0.224, 0.225] 
            normalize = video_transforms.Normalize(mean=clip_mean,
                                     std=clip_std)
            val_transform = video_transforms.Compose([
                    video_transforms.ToTensor(),
                    normalize])
            scale = 1
        elif "r2plus1d" in architecture_name:
            clip_mean = [0.43216, 0.394666, 0.37645]
            clip_std = [0.22803, 0.22145, 0.216989]
            normalize = video_transforms.Normalize(mean=clip_mean,
                                     std=clip_std)
            val_transform = video_transforms.Compose([
                    video_transforms.ToTensor(),
                    normalize])
            scale = 0.5
        elif 'rep_flow' in architecture_name:
            clip_mean = [0.5, 0.5, 0.5] 
            clip_std = [0.5, 0.5, 0.5]
    
            normalize = video_transforms.Normalize(mean=clip_mean,
                                     std=clip_std)
            val_transform = video_transforms.Compose([
                    video_transforms.ToTensor(),
                    normalize,
                ])
            scale = 1
        elif "slowfast" in architecture_name:
            clip_mean = [0.45, 0.45, 0.45]
            clip_std = [0.225, 0.225, 0.225] 
            normalize = video_transforms.Normalize(mean=clip_mean,
                                     std=clip_std)
            val_transform = video_transforms.Compose([
                    video_transforms.ToTensor(),
                    normalize,       
                ])
            scale = 1
        else:
            scale = 0.5
            clip_mean = [114.7748, 107.7354, 99.4750]
            clip_std = [1, 1, 1]
            normalize = video_transforms.Normalize(mean=clip_mean,
                                     std=clip_std)
            val_transform = video_transforms.Compose([
                    video_transforms.ToTensor2(),
                    normalize,
                ])
    elif 'flow' in architecture_name:
        if 'I3D' in architecture_name:
            clip_mean = [0.5] * 2
            clip_std = [0.5] * 2
            normalize = video_transforms.Normalize(mean=clip_mean,
                                             std=clip_std)
            
            val_transform = video_transforms.Compose([
                    video_transforms.ToTensor(),
                    normalize,
                ])
            scale = 1
        else:
            scale = 0.5
            clip_mean = [127.5, 127.5]
            clip_std = [1, 1]
            normalize = video_transforms.Normalize(mean=clip_mean,
                                     std=clip_std)
            val_transform = video_transforms.Compose([
                    video_transforms.ToTensor2(),
                    normalize,
                ])

    # selection
    #step = int(math.floor((duration-1)/(num_samples-1)))
    if '224' in architecture_name:
        scale = 1
    if '112' in architecture_name:
        scale = 0.5

    imageSize=int(224 * scale)
    dims = (int(256 * scale),int(340 * scale),3,duration)
    duration = duration - 1
    average_duration = int(duration / num_seg)
    offsetMainIndexes = []
    offsets = []
    for seg_id in range(num_seg):
        if average_duration >= length:
            offsetMainIndexes.append(int((average_duration - length + 1)/2 + seg_id * average_duration))
        elif duration >=length:
            average_part_length = int(np.floor((duration-length)/num_seg))
            offsetMainIndexes.append(int((average_part_length*(seg_id) + average_part_length*(seg_id+1))/2))
        else:
            increase = int(duration / num_seg)
            offsetMainIndexes.append(0 + seg_id * increase)
    for mainOffsetValue in offsetMainIndexes:
        for lengthID in range(1, length+1):
            loaded_frame_index = lengthID + mainOffsetValue
            moded_loaded_frame_index = loaded_frame_index % (duration + 1)
            if moded_loaded_frame_index == 0:
                moded_loaded_frame_index = (duration + 1)
            offsets.append(moded_loaded_frame_index)
             
    imageList=[]
    imageList1=[]
    imageList2=[]
    imageList3=[]
    imageList4=[]    
    imageList5=[]  
    imageList6=[]
    imageList7=[]
    imageList8=[]
    imageList9=[]    
    imageList10=[] 
    imageList11=[] 
    imageList12=[] 
    interpolation = cv2.INTER_LINEAR
    
    for index in offsets:
        if 'rgb' in architecture_name or 'pose' in architecture_name:
            img_file = os.path.join(vid_name, extension.format(index))
            img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
    
            img = cv2.resize(img, dims[1::-1],interpolation)
    
            #img2 = cv2.resize(img, dims2[1::-1],interpolation)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_flip = img[:,::-1,:].copy()
        elif 'flow' in architecture_name:
            flow_x_file = os.path.join(vid_name, extension.format('x',index))
            flow_y_file = os.path.join(vid_name, extension.format('y',index))
            img_x = cv2.imread(flow_x_file, cv2.IMREAD_GRAYSCALE)
            img_y = cv2.imread(flow_y_file, cv2.IMREAD_GRAYSCALE)
            img_x = np.expand_dims(img_x,-1)
            img_y = np.expand_dims(img_y,-1)
            img = np.concatenate((img_x,img_y),2)    
            img = cv2.resize(img, dims[1::-1],interpolation)
            img_flip = img[:,::-1,:].copy()
        #img_flip2 = img2[:,::-1,:].copy()
        imageList1.append(img[int(16 * scale):int(16 * scale + imageSize), int(58 * scale) : int(58 * scale + imageSize), :])
        imageList2.append(img[:imageSize, :imageSize, :])
        imageList3.append(img[:imageSize, -imageSize:, :])
        imageList4.append(img[-imageSize:, :imageSize, :])
        imageList5.append(img[-imageSize:, -imageSize:, :])
        imageList6.append(img_flip[int(16 * scale):int(16 * scale + imageSize), int(58 * scale) : int(58 * scale + imageSize), :])
        imageList7.append(img_flip[:imageSize, :imageSize, :])
        imageList8.append(img_flip[:imageSize, -imageSize:, :])
        imageList9.append(img_flip[-imageSize:, :imageSize, :])
        imageList10.append(img_flip[-imageSize:, -imageSize:, :])
#        imageList11.append(img2)
#        imageList12.append(img_flip2)

    if ten_crop:
        imageList=imageList1+imageList2+imageList3+imageList4+imageList5+imageList6+imageList7+imageList8+imageList9+imageList10
    else:
        imageList=imageList1
    
    #imageList=imageList11+imageList12
    
    rgb_list=[]     

    for i in range(len(imageList)):
        cur_img = imageList[i]
        cur_img_tensor = val_transform(cur_img)
        rgb_list.append(np.expand_dims(cur_img_tensor.numpy(), 0))
         
    input_data=np.concatenate(rgb_list,axis=0)   

    with torch.no_grad():
        imgDataTensor = torch.from_numpy(input_data).type(torch.FloatTensor).cuda()
        if 'rgb' in architecture_name or 'pose' in architecture_name:
            if 'tsm' in architecture_name:
                imgDataTensor = imgDataTensor.view(-1,length,3,imageSize,imageSize)
            else:
                imgDataTensor = imgDataTensor.view(-1,length,3,imageSize,imageSize).transpose(1,2)
        elif 'flow' in architecture_name:
            imgDataTensor = imgDataTensor.view(-1,length,2,imageSize,imageSize).transpose(1,2)
            
        if 'bert' in architecture_name or 'pooling' in architecture_name:
            output, input_vectors, sequenceOut, maskSample = net(imgDataTensor)
        else:
            output = net(imgDataTensor)
#        outputSoftmax=soft(output)
        result = output.data.cpu().numpy()
        mean_result=np.mean(result,0)
        prediction=np.argmax(mean_result)
        top3 = mean_result.argsort()[::-1][:3]
        
    return prediction, mean_result, top3