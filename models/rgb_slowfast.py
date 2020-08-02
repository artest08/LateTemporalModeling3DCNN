#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 20:27:26 2020

@author: esat
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np


import os
import sys
from collections import OrderedDict


from .BERT.bert import BERT, BERT2, BERT3, BERT4, BERT5, BERT6
from .SlowFast.slowfast_connector import slowfast_50


__all__ = ['rgb_slowfast64f_50', 
           'rgb_slowfast64f_50_bert10_FRAB_late', 'rgb_slowfast64f_50_bert10_FRAB_early',
           'rgb_slowfast64f_50_bert10_FRMB_early', 'rgb_slowfast64f_50_bert10_FRMB_late']

class rgb_slowfast64f_50(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_slowfast64f_50, self).__init__()
        self.model = slowfast_50(modelPath)
        self.num_classes=num_classes
        self.model.head.dropout = nn.Dropout(0.8)
        self.fc_action = nn.Linear(2304, num_classes)
        self.avgpool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        self.model.head.projection = self.fc_action
    def forward(self, x):
        fast_input = x[:, :, ::2, :, :]
        slow_input = x[:, :, ::8, :, :]
        x = self.model.forward([slow_input, fast_input])
        x = x.view(-1, self.num_classes)
        #x = self.model.forward([fast_input, slow_input])
        return x
    
    
    
#rgb_slowfast64f_50_bert10X
class rgb_slowfast64f_50_bert10_FRAB_late(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_slowfast64f_50_bert10_FRAB_late, self).__init__()
        self.hidden_size_fast=256
        self.hidden_size_slow=512
        self.n_layers=1
        self.attn_heads=8
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        
        self.avgpool = nn.AvgPool3d((1, 7, 7), stride=1)
        
        self.model = slowfast_50(modelPath)   
        self.model.head.dropout = nn.Dropout(0.8)
        
        self.bert_fast = BERT5(self.hidden_size_fast, 32 , hidden=self.hidden_size_fast, 
                               n_layers=self.n_layers, attn_heads=self.attn_heads)
        
        self.bert_slow = BERT5(self.hidden_size_slow, 8 , hidden=self.hidden_size_slow, 
                               n_layers=self.n_layers, attn_heads=self.attn_heads)
        
        downsample = nn.Sequential(
            nn.Conv3d(2048, 512,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(512)
        )


        self.mapper = Bottleneck(2048, 256, stride = 1, downsample = downsample)

        for m in self.mapper.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
                #m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()           
        self.fc_action = nn.Linear(self.hidden_size_fast + self.hidden_size_slow, num_classes)
            
        for param in self.model.parameters():
            param.requires_grad = True

        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        
    def forward(self, x):
        fast_input = x[:, :, ::2, :, :]
        slow_input = x[:, :, ::8, :, :]
        x = self.model.forward_feature([slow_input, fast_input])
        slow_feature = x[0]
        fast_feature = x[1]
        slow_feature = self.mapper(slow_feature)
        slow_feature = self.avgpool(slow_feature)
        fast_feature = self.avgpool(fast_feature)
        
        slow_feature = slow_feature.view(slow_feature.size(0), self.hidden_size_slow, 8)
        slow_feature = slow_feature.transpose(1,2)   
        output_slow , maskSample = self.bert_slow(slow_feature)
        slow_feature_out = output_slow[:,0,:]
        
        fast_feature = fast_feature.view(fast_feature.size(0), self.hidden_size_fast, 32)
        fast_feature = fast_feature.transpose(1,2)
        input_vectors = fast_feature
        output_fast , maskSample = self.bert_fast(fast_feature)
        fast_feature_out = output_fast[:,0,:]
        
        sequenceOut=output_fast[:,1:,:]
        classificationOut = torch.cat([slow_feature_out, fast_feature_out], 1)
        output=self.dp(classificationOut)
        x = self.fc_action(output)
        return x, input_vectors, sequenceOut, maskSample
    
    

    
#rgb_slowfast64f_50_bert10B
class rgb_slowfast64f_50_bert10_FRAB_early(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_slowfast64f_50_bert10_FRAB_early, self).__init__()
        self.hidden_size_fast=128
        self.hidden_size_slow=512
        self.n_layers=1
        self.attn_heads=8
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        
        self.avgpool_slow = nn.AdaptiveAvgPool3d(output_size=(8, 1, 1))
        self.avgpool_fast = nn.AdaptiveAvgPool3d(output_size=(8, 1, 1))
        
        self.model = slowfast_50(modelPath)   
        self.model.head.dropout = nn.Dropout(0.8)
        
        self.bert = BERT5(self.hidden_size_fast + self.hidden_size_slow, 8 , 
                               hidden=self.hidden_size_fast + self.hidden_size_slow, 
                               n_layers=self.n_layers, attn_heads=self.attn_heads)
        
        
        downsample_slow = nn.Sequential(
            nn.Conv3d(2048, self.hidden_size_slow,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(self.hidden_size_slow)
        )


        self.mapper_slow = Bottleneck(2048, int(self.hidden_size_slow / 2), stride = 1, 
                                      downsample = downsample_slow)
        
        downsample_fast = nn.Sequential(
            nn.Conv3d(256, self.hidden_size_fast,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(self.hidden_size_fast)
        )


        self.mapper_fast = Bottleneck(256, int(self.hidden_size_fast / 2), stride = 1, 
                                      downsample = downsample_fast)

        for m in self.mapper_slow.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
                #m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_() 
                
        for m in self.mapper_fast.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
                #m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_() 
                
        self.fc_action = nn.Linear(self.hidden_size_fast + self.hidden_size_slow, num_classes)
            
        for param in self.model.parameters():
            param.requires_grad = True

        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        
    def forward(self, x):
        fast_input = x[:, :, ::2, :, :]
        slow_input = x[:, :, ::8, :, :]
        x = self.model.forward_feature([slow_input, fast_input])
        slow_feature = x[0]
        fast_feature = x[1]
        slow_feature = self.mapper_slow(slow_feature)
        fast_feature = self.mapper_fast(fast_feature)
        
        slow_feature = self.avgpool_slow(slow_feature)
        slow_feature = slow_feature.view(slow_feature.size(0), self.hidden_size_slow, 8)
        slow_feature = slow_feature.transpose(1,2) 
        
        fast_feature = self.avgpool_fast(fast_feature)
        fast_feature = fast_feature.view(fast_feature.size(0), self.hidden_size_fast, 8)
        fast_feature = fast_feature.transpose(1,2)
        
        x = torch.cat([slow_feature, fast_feature], -1)
 
        input_vectors=x
        norm = input_vectors.norm(p=2, dim = -1, keepdim=True)
        input_vectors = input_vectors.div(norm)
        output , maskSample = self.bert(x)
        classificationOut = output[:,0,:]
        sequenceOut=output[:,1:,:]
        norm = sequenceOut.norm(p=2, dim = -1, keepdim=True)
        sequenceOut = sequenceOut.div(norm)
        output=self.dp(classificationOut)
        x = self.fc_action(output)
        return x, input_vectors, sequenceOut, maskSample
    
    
#rgb_slowfast64f_50_bert10SS_early   
class rgb_slowfast64f_50_bert10_FRMB_early(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_slowfast64f_50_bert10_FRMB_early, self).__init__()
        self.hidden_size_fast=128
        self.hidden_size_slow=512
        self.n_layers=1
        self.attn_heads=8
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        
        self.avgpool_slow = nn.AdaptiveAvgPool3d(output_size=(8, 1, 1))
        self.avgpool_fast = nn.AdaptiveAvgPool3d(output_size=(8, 1, 1))
        
        self.model = slowfast_50(modelPath)   
        self.model.head.dropout = nn.Dropout(0.8)
        
        self.bert = BERT5(self.hidden_size_fast + self.hidden_size_slow, 8 , 
                               hidden=self.hidden_size_fast + self.hidden_size_slow, 
                               n_layers=self.n_layers, attn_heads=self.attn_heads)
        
        
        downsample_slow = nn.Sequential(
            nn.Conv3d(2048, self.hidden_size_slow,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(self.hidden_size_slow)
        )


        self.mapper_slow = Bottleneck(2048, int(self.hidden_size_slow / 2), stride = 1, 
                                      downsample = downsample_slow)
        
        downsample_fast = nn.Sequential(
            nn.Conv3d(256, self.hidden_size_fast,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(self.hidden_size_fast)
        )


        self.mapper_fast = Bottleneck(256, int(self.hidden_size_fast / 2), stride = 1, 
                                      downsample = downsample_fast)


        for m in self.mapper_slow.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
                #m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_() 
                
        for m in self.mapper_fast.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
                #m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_() 

        
        self.model.s5.pathway0_res2 = self.mapper_slow
        self.model.s5.pathway1_res2 = self.mapper_fast
                
        self.fc_action = nn.Linear(self.hidden_size_fast + self.hidden_size_slow, num_classes)
            
        for param in self.model.parameters():
            param.requires_grad = True

        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        
    def forward(self, x):
        fast_input = x[:, :, ::2, :, :]
        slow_input = x[:, :, ::8, :, :]
        x = self.model.forward_feature([slow_input, fast_input])
        slow_feature = x[0]
        fast_feature = x[1]
        
        slow_feature = self.avgpool_slow(slow_feature)
        slow_feature = slow_feature.view(slow_feature.size(0), self.hidden_size_slow, 8)
        slow_feature = slow_feature.transpose(1,2) 
        
        fast_feature = self.avgpool_fast(fast_feature)
        fast_feature = fast_feature.view(fast_feature.size(0), self.hidden_size_fast, 8)
        fast_feature = fast_feature.transpose(1,2)
        
        x = torch.cat([slow_feature, fast_feature], -1)
 
        input_vectors=x
        norm = input_vectors.norm(p=2, dim = -1, keepdim=True)
        input_vectors = input_vectors.div(norm)
        output , maskSample = self.bert(x)
        classificationOut = output[:,0,:]
        sequenceOut=output[:,1:,:]
        norm = sequenceOut.norm(p=2, dim = -1, keepdim=True)
        sequenceOut = sequenceOut.div(norm)
        output=self.dp(classificationOut)
        x = self.fc_action(output)
        return x, input_vectors, sequenceOut, maskSample
    
#rgb_slowfast64f_50_bert10SS_late
class rgb_slowfast64f_50_bert10_FRMB_late(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_slowfast64f_50_bert10_FRMB_late, self).__init__()
        self.hidden_size_fast=256
        self.hidden_size_slow=512
        self.n_layers=1
        self.attn_heads=8
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        
        self.avgpool = nn.AvgPool3d((1, 7, 7), stride=1)
        
        self.model = slowfast_50(modelPath)   
        self.model.head.dropout = nn.Dropout(0.8)
        
        self.bert_fast = BERT5(self.hidden_size_fast, 32 , hidden=self.hidden_size_fast, 
                               n_layers=self.n_layers, attn_heads=self.attn_heads)
        
        self.bert_slow = BERT5(self.hidden_size_slow, 8 , hidden=self.hidden_size_slow, 
                               n_layers=self.n_layers, attn_heads=self.attn_heads)
        
        downsample = nn.Sequential(
            nn.Conv3d(2048, 512,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(512)
        )


        self.mapper = Bottleneck(2048, 256, stride = 1, downsample = downsample)

        for m in self.mapper.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
                #m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()           
        self.fc_action = nn.Linear(self.hidden_size_fast + self.hidden_size_slow, num_classes)
            
        self.model.s5.pathway0_res2 = self.mapper
        
        for param in self.model.parameters():
            param.requires_grad = True

        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        
    def forward(self, x):
        fast_input = x[:, :, ::2, :, :]
        slow_input = x[:, :, ::8, :, :]
        x = self.model.forward_feature([slow_input, fast_input])
        slow_feature = x[0]
        fast_feature = x[1]
        slow_feature = self.avgpool(slow_feature)
        fast_feature = self.avgpool(fast_feature)
        
        slow_feature = slow_feature.view(slow_feature.size(0), self.hidden_size_slow, 8)
        slow_feature = slow_feature.transpose(1,2)   
        output_slow , maskSample = self.bert_slow(slow_feature)
        slow_feature_out = output_slow[:,0,:]
        
        fast_feature = fast_feature.view(fast_feature.size(0), self.hidden_size_fast, 32)
        fast_feature = fast_feature.transpose(1,2)
        input_vectors = fast_feature
        output_fast , maskSample = self.bert_fast(fast_feature)
        fast_feature_out = output_fast[:,0,:]
        
        sequenceOut=output_fast[:,1:,:]
        classificationOut = torch.cat([slow_feature_out, fast_feature_out], 1)
        output=self.dp(classificationOut)
        x = self.fc_action(output)
        return x, input_vectors, sequenceOut, maskSample
        

        
    
class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 2, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 2)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
