import math

#import rep_flow_layer as rf
from .rep_flow_layer import FlowLayer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

import os
import sys


class SamePadding(nn.Module):

    def __init__(self, kernel_size, stride):
        super(SamePadding, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
    
    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)
      
    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        #print t,h,w
        out_t = np.ceil(float(t) / float(self.stride[0]))
        out_h = np.ceil(float(h) / float(self.stride[1]))
        out_w = np.ceil(float(w) / float(self.stride[2]))
        #print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        #print pad_t, pad_h, pad_w
        
        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f
        
        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        #print x.size()
        #print pad
        x = F.pad(x, pad)
        return x


class Bottleneck3D(nn.Module): 
    def __init__(self, inputs, filters, is_training, strides,
                 use_projection=False, T=3, data_format='channels_last', non_local=False):
        super(Bottleneck3D, self).__init__()
                
        df = 'NDHWC' if data_format == 'channels_last' else 'NCDHW'
        self.shortcut = None
        if use_projection:   
            # Projection shortcut only in first block within a group. Bottleneck blocks
            # end with 4 times the number of filters.
            filters_out = 4 * filters
            self.shortcut = nn.Sequential(SamePadding((1,1,1),(1,strides,strides)),
                nn.Conv3d(inputs, filters_out, kernel_size=1, stride=(1,strides,strides), bias=False, padding=0),
                nn.BatchNorm3d(filters_out),
                nn.BatchNorm3d(filters_out)) # there are two, due to old models having it. To load weights, 2 batch norms are needed here...
      
        self.layers = nn.Sequential(SamePadding((T,1,1), (1,1,1)),
            nn.Conv3d(inputs, filters, kernel_size=(T,1,1), stride=1, padding=(0,0,0), bias=False), #1
            nn.BatchNorm3d(filters), #2
            nn.ReLU(),
            SamePadding((1,3,3),(1,strides,strides)),
            nn.Conv3d(filters, filters, kernel_size=(1,3,3), stride=(1,strides,strides), bias=False, padding=0), #5
            nn.BatchNorm3d(filters),#6
            nn.ReLU(),
            nn.Conv3d(filters, 4*filters, kernel_size=1, stride=1, bias=False, padding=0),#8
            nn.BatchNorm3d(4*filters))#9
    

    def forward(self, x):
        #print('block', x.size())
        if self.shortcut:
          res = self.shortcut(x)
        else:
          res = x
        #print('b2',x.size())
        return F.relu(self.layers(x) + res)


  
class Block3D(nn.Module):
    def __init__(self, inputs, filters, block_fn, blocks, strides, is_training, name,
                     data_format='channels_last', non_local=0):
        super(Block3D, self).__init__()
          
        self.blocks = nn.ModuleList()
        # Only the first block per block_group uses projection shortcut and strides.
        self.blocks.append(Bottleneck3D(inputs, filters, is_training, strides,
                                        use_projection=True, data_format=data_format))
        inputs = filters * 4
        T = 3
        for i in range(1, blocks):
          self.blocks.append(Bottleneck3D(inputs, filters, is_training, 1, T=T,
                                          data_format=data_format, non_local=0))
          # only use 1 3D conv per 2 residual blocks (per Non-local NN paper)
          T = (3 if T==1 else 1)
      
    
    def forward(self, x):
        for block in self.blocks:
          x = block(x)
        return x

class ResNet3D(nn.Module):
  
    def __init__(self, block_fn, layers, num_classes,
                 data_format='channels_last', non_local=[], rep_flow=[],
                 dropout_keep_prob=0.5):
    
        super(ResNet3D, self).__init__()
        is_training = False # no effect in pytorch
         
        """Creation of the model graph."""
        self.stem = nn.Conv3d(
          3, 64, kernel_size=7, bias=False, stride=2)
        
        self.bn1 = nn.BatchNorm3d(64, eps=0.001, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        self.pad = SamePadding((3,3,3),(2,2,2))
        self.maxpool = nn.MaxPool3d(kernel_size=3,
                                    stride=2, padding=0)

    
    # res 2
        inputs = 64
        self.res2 = Block3D(
          inputs=inputs, filters=64, block_fn=block_fn, blocks=layers[0],
          strides=1, is_training=is_training, name='block_group1',
          data_format=data_format, non_local=non_local[0])
        
        # res 3
        inputs = 64*4
        self.res3 = Block3D(
          inputs=inputs, filters=128, block_fn=block_fn, blocks=layers[1],
          strides=2, is_training=is_training, name='block_group2',
          data_format=data_format, non_local=non_local[1])
        
        self.rep_flow = FlowLayer(512)
        
        # res 4
        inputs = 128*4
        self.res4 = Block3D(
          inputs=inputs, filters=256, block_fn=block_fn, blocks=layers[2],
          strides=2, is_training=is_training, name='block_group3',
          data_format=data_format, non_local=non_local[2])
        
        # res 5
        inputs = 256*4
        self.res5 = Block3D(
            inputs=inputs, filters=512, block_fn=block_fn, blocks=layers[3],
            strides=2, is_training=is_training, name='block_group4',
            data_format=data_format, non_local=non_local[3])
        
        self.dropout = nn.Dropout(0.5)
        self.classify = nn.Conv3d(512*4, num_classes, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.stem(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(self.pad(x))
        x = self.res2(x)
        x = self.res3(x)
      
        x = self.rep_flow(x)
      
        x = self.res4(x)
        x = self.res5(x)
        x = x.mean(3).mean(3).unsqueeze(3).unsqueeze(3) # spatial average
        x = self.dropout(x)
        x = self.classify(x)
        x = x.mean(2) # temporal average
        return x


def resnet_3d_v1(resnet_depth, num_classes, data_format='channels_last', is_3d=True, non_local=[0,0,0,0], rep_flow=[0,0,0,0,0]):
    """Returns the ResNet model for a given size and number of output classes."""
    model_params = {
        18: {'block': None, 'layers': [2, 2, 2, 2]},
        34: {'block': None, 'layers': [3, 4, 6, 3]},
        50: {'block': None, 'layers': [3, 4, 6, 3]},
        101: {'block': None, 'layers': [3, 4, 23, 3]},
        152: {'block': None, 'layers': [3, 8, 36, 3]},
        200: {'block': None, 'layers': [3, 24, 36, 3]}
    }
    
    if resnet_depth not in model_params:
        raise ValueError('Not a valid resnet_depth:', resnet_depth)
    
    params = model_params[resnet_depth]
    return ResNet3D(
        params['block'], params['layers'], num_classes, data_format, non_local, rep_flow)


def resnet_50_rep_flow(model_path=''):
    model = resnet_3d_v1(50, 400)   
    
    if not model_path == '':
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
    return model