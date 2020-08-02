"""
Created on Tue Apr 28 19:31:09 2020

@author: esat
"""

from .video_transforms import Normalize, ToTensor2, ToTensor, Scale, Compose
def determine_architecture_transform(architecture_name_list, num_seg, length):
    transform_list = []
    for architecture_name in architecture_name_list:
        if 'I3D' in architecture_name:
            if 'resnet' in architecture_name:
                clip_mean = [0.45, 0.45, 0.45] * num_seg * length
                clip_std = [0.225, 0.225, 0.225] * num_seg * length
            else:
                clip_mean = [0.5, 0.5, 0.5] * num_seg * length
                clip_std = [0.5, 0.5, 0.5] * num_seg * length
            #clip_std = [0.25, 0.25, 0.25] * num_seg * length
        elif 'MFNET3D' in architecture_name:
            clip_mean = [0.48627451, 0.45882353, 0.40784314] * num_seg * length
            clip_std = [0.234, 0.234, 0.234]  * num_seg * length
        elif "3D" in architecture_name:
            clip_mean = [114.7748, 107.7354, 99.4750] * num_seg * length
            clip_std = [1, 1, 1] * num_seg * length
        elif "r2plus1d" in architecture_name:
            clip_mean = [0.43216, 0.394666, 0.37645] * num_seg * length
            clip_std = [0.22803, 0.22145, 0.216989] * num_seg * length
        elif "rep_flow" in architecture_name:
            clip_mean = [0.5, 0.5, 0.5] * num_seg * length
            clip_std = [0.5, 0.5, 0.5] * num_seg * length      
        elif "slowfast" in architecture_name:
            clip_mean = [0.45, 0.45, 0.45] * num_seg * length
            clip_std = [0.225, 0.225, 0.225] * num_seg * length
        else:
            clip_mean = [0.485, 0.456, 0.406] * num_seg * length
            clip_std = [0.229, 0.224, 0.225] * num_seg * length
            
        if '3D' in architecture_name:
            if 'I3D' in architecture_name or 'MFNET3D' in architecture_name:
                if '112' in architecture_name:
                    scale = 0.5
                else:
                    scale = 1
            else:
                scale = 0.5
        elif 'r2plus1d' in architecture_name:
            scale = 0.5
        else:
            scale = 1
            
        if scale == 1:
            size = 224
        else:
            size = 112
            
        normalize = Normalize(mean=clip_mean, std=clip_std)  
        scale_transform = Scale(size)
          
        if "3D" in architecture_name and not ('I3D' in architecture_name or 'MFNET3D' in architecture_name):
            tensor_transform = ToTensor2()
        else:
            tensor_transform = ToTensor()
            
        transform = Compose([
                scale_transform,
                tensor_transform,
                normalize,
            ])
        transform_list.append(transform)
        print(architecture_name)
        print('size: %d' %(size))
    return transform_list


def determine_architecture_transform2(architecture_name_list, num_seg, length):
    transform_list = []
    for architecture_name in architecture_name_list:
        if 'I3D' in architecture_name:
            if 'resnet' in architecture_name:
                clip_mean = [0.45, 0.45, 0.45, 0.5, 0.5] * num_seg * length
                clip_std = [0.225, 0.225, 0.225, 0.5, 0.5] * num_seg * length
            else:
                clip_mean = [0.5, 0.5, 0.5, 0.5, 0.5] * num_seg * length
                clip_std = [0.5, 0.5, 0.5, 0.5, 0.5] * num_seg * length
        elif "3D" in architecture_name:
            clip_mean = [114.7748, 107.7354, 99.4750, 127.5, 127.5] * num_seg * length
            clip_std = [1, 1, 1, 1, 1] * num_seg * length
        elif "r2plus1d" in architecture_name:
            clip_mean = [0.43216, 0.394666, 0.37645, 0.5, 0.5] * num_seg * length
            clip_std = [0.22803, 0.22145, 0.216989, 0.225, 0.225] * num_seg * length    
        elif "slowfast" in architecture_name:
            clip_mean = [0.45, 0.45, 0.45, 0.5, 0.5] * num_seg * length
            clip_std = [0.225, 0.225, 0.225, 0.225, 0.225] * num_seg * length
        else:
            clip_mean = [0.485, 0.456, 0.406] * num_seg * length
            clip_std = [0.229, 0.224, 0.225] * num_seg * length
            
        if '3D' in architecture_name:
            if 'I3D' in architecture_name or 'MFNET3D' in architecture_name:
                if '112' in architecture_name:
                    scale = 0.5
                else:
                    scale = 1
            else:
                scale = 0.5
        elif 'r2plus1d' in architecture_name:
            scale = 0.5
        else:
            scale = 1
            
        if scale == 1:
            size = 224
        else:
            size = 112
            
        normalize = Normalize(mean=clip_mean, std=clip_std)  
        scale_transform = Scale(size)
          
        if "3D" in architecture_name and not ('I3D' in architecture_name or 'MFNET3D' in architecture_name):
            tensor_transform = ToTensor2()
        else:
            tensor_transform = ToTensor()
            
        transform = Compose([
                scale_transform,
                tensor_transform,
                normalize,
            ])
        transform_list.append(transform)
        print(architecture_name)
        print('size: %d' %(size))
    return transform_list
        
