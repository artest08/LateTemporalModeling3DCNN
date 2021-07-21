#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 10:42:00 2019

@author: esat
"""

import os
import time
import argparse
import shutil
import numpy as np


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from tensorboardX import SummaryWriter

from torch.optim import lr_scheduler
import video_transforms
import models
import datasets
from opt.AdamW import AdamW
from utils.model_path import rgb_3d_model_path_selection


model_names = sorted(name for name in models.__dict__
    if not name.startswith("__")
    and callable(models.__dict__[name]))

dataset_names = sorted(name for name in datasets.__all__)

parser = argparse.ArgumentParser(description='PyTorch Two-Stream Action Recognition')
#parser.add_argument('--data', metavar='DIR', default='./datasets/ucf101_frames',
#                    help='path to dataset')
parser.add_argument('--settings', metavar='DIR', default='./datasets/settings',
                    help='path to datset setting files')
#parser.add_argument('--modality', '-m', metavar='MODALITY', default='rgb',
#                    choices=["rgb", "flow"],
#                    help='modality: rgb | flow')
parser.add_argument('--dataset', '-d', default='hmdb51',
                    choices=["ucf101", "hmdb51", "smtV2", "window"],
                    help='dataset: ucf101 | hmdb51 | smtV2')

parser.add_argument('--arch', '-a', default='rgb_resneXt3D64f101_bert10_FRMB',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: rgb_resneXt3D64f101_bert10_FRMB)')

parser.add_argument('-s', '--split', default=1, type=int, metavar='S',
                    help='which split of data to work on (default: 1)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N', help='mini-batch size (default: 8)')
parser.add_argument('--iter-size', default=16, type=int,
                    metavar='I', help='iter size to reduce memory usage (default: 16)')
parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-3, type=float,
                    metavar='W', help='weight decay (default: 1e-3)')
parser.add_argument('--print-freq', default=400, type=int,
                    metavar='N', help='print frequency (default: 400)')
parser.add_argument('--save-freq', default=1, type=int,
                    metavar='N', help='save frequency (default: 1)')
parser.add_argument('--num-seg', default=1, type=int,
                    metavar='N', help='Number of segments in dataloader (default: 1)')
#parser.add_argument('--resume', default='./dene4', type=str, metavar='PATH',
#                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-c', '--continue', dest='contine', action='store_true',
                    help='evaluate model on validation set')

best_prec1 = 0
best_loss = 30
warmUpEpoch=5

smt_pretrained = False

HALF = False
training_continue = False

def main():
    global args, best_prec1,model,writer,best_loss, length, width, height, input_size, scheduler
    args = parser.parse_args()
    training_continue = args.contine
    if '3D' in args.arch:
        if 'I3D' in args.arch or 'MFNET3D' in args.arch:
            if '112' in args.arch:
                scale = 0.5
            else:
                scale = 1
        else:
            if '224' in args.arch:
                scale = 1
            else:
                scale = 0.5
    elif 'r2plus1d' in args.arch:
        scale = 0.5
    else:
        scale = 1
        
    print('scale: %.1f' %(scale))
    
    input_size = int(224 * scale)
    width = int(340 * scale)
    height = int(256 * scale)
    
    saveLocation="./checkpoint/"+args.dataset+"_"+args.arch+"_split"+str(args.split)
    if not os.path.exists(saveLocation):
        os.makedirs(saveLocation)
    writer = SummaryWriter(saveLocation)
   
    # create model

    if args.evaluate:
        print("Building validation model ... ")
        model = build_model_validate()
        optimizer = AdamW(model.parameters(), lr= args.lr, weight_decay=args.weight_decay)
    elif training_continue:
        model, startEpoch, optimizer, best_prec1 = build_model_continue()
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            #param_group['lr'] = lr
        print("Continuing with best precision: %.3f and start epoch %d and lr: %f" %(best_prec1,startEpoch,lr))
    else:
        print("Building model with ADAMW... ")
        model = build_model()
        optimizer = AdamW(model.parameters(), lr= args.lr, weight_decay=args.weight_decay)
        startEpoch = 0

    
    if HALF:
        model.half()  # convert to half precision
        for layer in model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.float()
    
    print("Model %s is loaded. " % (args.arch))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    criterion2 = nn.MSELoss().cuda()
    
    
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=5, verbose=True)
    

    print("Saving everything to directory %s." % (saveLocation))
    if args.dataset=='ucf101':
        dataset='./datasets/ucf101_frames'
    elif args.dataset=='hmdb51':
        dataset='./datasets/hmdb51_frames'
    elif args.dataset=='smtV2':
        dataset='./datasets/smtV2_frames'
    elif args.dataset=='window':
        dataset='./datasets/window_frames'
    else:
        print("No convenient dataset entered, exiting....")
        return 0
    
    cudnn.benchmark = True
    modality=args.arch.split('_')[0]
    if "3D" in args.arch or 'tsm' in args.arch or 'slowfast' in args.arch or 'r2plus1d' in args.arch:
        if '64f' in args.arch:
            length=64
        elif '32f' in args.arch:
            length=32
        else:
            length=16
    else:
        length=1
    # Data transforming
    if modality == "rgb" or modality == "pose":
        is_color = True
        scale_ratios = [1.0, 0.875, 0.75, 0.66]
        if 'I3D' in args.arch:
            if 'resnet' in args.arch:
                clip_mean = [0.45, 0.45, 0.45] * args.num_seg * length
                clip_std = [0.225, 0.225, 0.225] * args.num_seg * length
            else:
                clip_mean = [0.5, 0.5, 0.5] * args.num_seg * length
                clip_std = [0.5, 0.5, 0.5] * args.num_seg * length
            #clip_std = [0.25, 0.25, 0.25] * args.num_seg * length
        elif 'MFNET3D' in args.arch:
            clip_mean = [0.48627451, 0.45882353, 0.40784314] * args.num_seg * length
            clip_std = [0.234, 0.234, 0.234]  * args.num_seg * length
        elif "3D" in args.arch:
            clip_mean = [114.7748, 107.7354, 99.4750] * args.num_seg * length
            clip_std = [1, 1, 1] * args.num_seg * length
        elif "r2plus1d" in args.arch:
            clip_mean = [0.43216, 0.394666, 0.37645] * args.num_seg * length
            clip_std = [0.22803, 0.22145, 0.216989] * args.num_seg * length
        elif "rep_flow" in args.arch:
            clip_mean = [0.5, 0.5, 0.5] * args.num_seg * length
            clip_std = [0.5, 0.5, 0.5] * args.num_seg * length      
        elif "slowfast" in args.arch:
            clip_mean = [0.45, 0.45, 0.45] * args.num_seg * length
            clip_std = [0.225, 0.225, 0.225] * args.num_seg * length
        else:
            clip_mean = [0.485, 0.456, 0.406] * args.num_seg * length
            clip_std = [0.229, 0.224, 0.225] * args.num_seg * length
    elif modality == "pose":
        is_color = True
        scale_ratios = [1.0, 0.875, 0.75, 0.66]
        clip_mean = [0.485, 0.456, 0.406] * args.num_seg
        clip_std = [0.229, 0.224, 0.225] * args.num_seg
    elif modality == "flow":
        is_color = False
        scale_ratios = [1.0, 0.875, 0.75, 0.66]
        if 'I3D' in args.arch:
            clip_mean = [0.5, 0.5] * args.num_seg * length
            clip_std = [0.5, 0.5] * args.num_seg * length
        elif "3D" in args.arch:
            clip_mean = [127.5, 127.5] * args.num_seg * length
            clip_std = [1, 1] * args.num_seg * length        
        else:
            clip_mean = [0.5, 0.5] * args.num_seg * length
            clip_std = [0.226, 0.226] * args.num_seg * length
    elif modality == "both":
        is_color = True
        scale_ratios = [1.0, 0.875, 0.75, 0.66]
        clip_mean = [0.485, 0.456, 0.406, 0.5, 0.5] * args.num_seg * length
        clip_std = [0.229, 0.224, 0.225, 0.226, 0.226] * args.num_seg * length
    else:
        print("No such modality. Only rgb and flow supported.")

    
    normalize = video_transforms.Normalize(mean=clip_mean,
                                           std=clip_std)

    if "3D" in args.arch and not ('I3D' in args.arch):
        train_transform = video_transforms.Compose([
                video_transforms.MultiScaleCrop((input_size, input_size), scale_ratios),
                video_transforms.RandomHorizontalFlip(),
                video_transforms.ToTensor2(),
                normalize,
            ])
    
        val_transform = video_transforms.Compose([
                video_transforms.CenterCrop((input_size)),
                video_transforms.ToTensor2(),
                normalize,
            ])
    else:
        train_transform = video_transforms.Compose([
                video_transforms.MultiScaleCrop((input_size, input_size), scale_ratios),
                video_transforms.RandomHorizontalFlip(),
                video_transforms.ToTensor(),
                normalize,
            ])
    
        val_transform = video_transforms.Compose([
                video_transforms.CenterCrop((input_size)),
                video_transforms.ToTensor(),
                normalize,
            ])

    # data loading
    train_setting_file = "train_%s_split%d.txt" % (modality, args.split)
    train_split_file = os.path.join(args.settings, args.dataset, train_setting_file)
    val_setting_file = "val_%s_split%d.txt" % (modality, args.split)
    val_split_file = os.path.join(args.settings, args.dataset, val_setting_file)
    if not os.path.exists(train_split_file) or not os.path.exists(val_split_file):
        print("No split file exists in %s directory. Preprocess the dataset first" % (args.settings))

    train_dataset = datasets.__dict__[args.dataset](root=dataset,
                                                    source=train_split_file,
                                                    phase="train",
                                                    modality=modality,
                                                    is_color=is_color,
                                                    new_length=length,
                                                    new_width=width,
                                                    new_height=height,
                                                    video_transform=train_transform,
                                                    num_segments=args.num_seg)
    
    val_dataset = datasets.__dict__[args.dataset](root=dataset,
                                                  source=val_split_file,
                                                  phase="val",
                                                  modality=modality,
                                                  is_color=is_color,
                                                  new_length=length,
                                                  new_width=width,
                                                  new_height=height,
                                                  video_transform=val_transform,
                                                  num_segments=args.num_seg)

    print('{} samples found, {} train samples and {} test samples.'.format(len(val_dataset)+len(train_dataset),
                                                                           len(train_dataset),
                                                                           len(val_dataset)))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        prec1,prec3,lossClassification = validate(val_loader, model, criterion,criterion2,modality)
        return

    for epoch in range(startEpoch, args.epochs):
#        if learning_rate_index > max_learning_rate_decay_count:
#            break
#        adjust_learning_rate(optimizer, epoch)
        train(train_loader, model, criterion,criterion2, optimizer, epoch,modality)

        # evaluate on validation set
        prec1 = 0.0
        lossClassification = 0
        if (epoch + 1) % args.save_freq == 0:
            prec1,prec3,lossClassification = validate(val_loader, model, criterion,criterion2,modality)
            writer.add_scalar('data/top1_validation', prec1, epoch)
            writer.add_scalar('data/top3_validation', prec3, epoch)
            writer.add_scalar('data/classification_loss_validation', lossClassification, epoch)
            scheduler.step(lossClassification)
        # remember best prec@1 and save checkpoint
        
        is_best = prec1 >= best_prec1
        best_prec1 = max(prec1, best_prec1)
#        best_in_existing_learning_rate = max(prec1, best_in_existing_learning_rate)
#        
#        if best_in_existing_learning_rate > prec1 + 1:
#            learning_rate_index = learning_rate_index 
#            best_in_existing_learning_rate = 0        

        if (epoch + 1) % args.save_freq == 0:
            checkpoint_name = "%03d_%s" % (epoch + 1, "checkpoint.pth.tar")
            if is_best:
                print("Model son iyi olarak kaydedildi")
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                    'best_loss': best_loss,
                    'optimizer' : optimizer.state_dict(),
                }, is_best, checkpoint_name, saveLocation)
    
    checkpoint_name = "%03d_%s" % (epoch + 1, "checkpoint.pth.tar")
    save_checkpoint({
        'epoch': epoch + 1,
        'arch': args.arch,
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
        'best_loss': best_loss,
        'optimizer' : optimizer.state_dict(),
    }, is_best, checkpoint_name, saveLocation)
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()

def build_model():
    modality=args.arch.split('_')[0]
    if modality == "rgb":
        model_path = rgb_3d_model_path_selection(args.arch)
        #model_path = os.path.join(modelLocation,'model_best.pth.tar') 
        
    elif modality == "flow":
        model_path=''
        if "3D" in args.arch:
            if 'I3D' in args.arch:
                 model_path='./weights/flow_imagenet.pth'   
            elif '3D' in args.arch:
                 model_path='./weights/Flow_Kinetics_64f.pth'         
        #model_path = os.path.join(modelLocation,'model_best.pth.tar') 
    elif modality == "both":
        model_path='' 
        
    if args.dataset=='ucf101':
        print('model path is: %s' %(model_path))
        model = models.__dict__[args.arch](modelPath=model_path, num_classes=101,length=args.num_seg)
    elif args.dataset=='hmdb51':
        print('model path is: %s' %(model_path))
        model = models.__dict__[args.arch](modelPath=model_path, num_classes=51, length=args.num_seg)
    elif args.dataset=='smtV2':
        print('model path is: %s' %(model_path))
        model = models.__dict__[args.arch](modelPath=model_path, num_classes=174, length=args.num_seg)
    elif args.dataset=='window':
        print('model path is: %s' %(model_path))
        model = models.__dict__[args.arch](modelPath=model_path, num_classes=3, length=args.num_seg)
    
    if torch.cuda.device_count() > 1:
        model=torch.nn.DataParallel(model)
    model = model.cuda()
    
    return model

def build_model_validate():
    modelLocation="./checkpoint/"+args.dataset+"_"+args.arch+"_split"+str(args.split)
    model_path = os.path.join(modelLocation,'model_best.pth.tar') 
    params = torch.load(model_path)
    print(modelLocation)
    if args.dataset=='ucf101':
        model=models.__dict__[args.arch](modelPath='', num_classes=101,length=args.num_seg)
    elif args.dataset=='hmdb51':
        model=models.__dict__[args.arch](modelPath='', num_classes=51,length=args.num_seg)
   
    if torch.cuda.device_count() > 1:
        model=torch.nn.DataParallel(model) 

    model.load_state_dict(params['state_dict'])
    model.cuda()
    model.eval() 
    return model

def build_model_continue():
    modelLocation="./checkpoint/"+args.dataset+"_"+args.arch+"_split"+str(args.split)
    model_path = os.path.join(modelLocation,'model_best.pth.tar') 
    params = torch.load(model_path)
    print(modelLocation)
    if args.dataset=='ucf101':
        model=models.__dict__[args.arch](modelPath='', num_classes=101,length=args.num_seg)
    elif args.dataset=='hmdb51':
        model=models.__dict__[args.arch](modelPath='', num_classes=51,length=args.num_seg)
   
    if torch.cuda.device_count() > 1:
        model=torch.nn.DataParallel(model) 
        
    model.load_state_dict(params['state_dict'])
    model = model.cuda()
    optimizer = AdamW(model.parameters(), lr= args.lr, weight_decay=args.weight_decay)
    optimizer.load_state_dict(params['optimizer'])
    
    startEpoch = params['epoch']
    best_prec = params['best_prec1']
    return model, startEpoch, optimizer, best_prec



def train(train_loader, model, criterion, criterion2, optimizer, epoch,modality):
    batch_time = AverageMeter()
    lossesClassification = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    

    # switch to train mode
    model.train()

    end = time.time()
    optimizer.zero_grad()
    loss_mini_batch_classification = 0.0
    acc_mini_batch = 0.0
    acc_mini_batch_top3 = 0.0
    totalSamplePerIter=0
    for i, (inputs, targets) in enumerate(train_loader):
        if modality == "rgb" or modality == "pose":
            if "3D" in args.arch or "r2plus1d" in args.arch or 'slowfast' in args.arch:
                inputs=inputs.view(-1,length,3,input_size,input_size).transpose(1,2)
        elif modality == "flow":
            if "3D" in args.arch or "r2plus1d" in args.arch:
                inputs=inputs.view(-1,length,2,input_size,input_size).transpose(1,2)
            else:
                inputs=inputs.view(-1,2*length,input_size,input_size)          
        elif modality == "both":
            inputs=inputs.view(-1,5*length,input_size,input_size)
            
        if HALF:
            inputs = inputs.cuda().half()
        else:
            inputs = inputs.cuda()
        targets = targets.cuda()
        output, input_vectors, sequenceOut, maskSample = model(inputs)
        
#        maskSample=maskSample.cuda()
#        input_vectors=(1-maskSample[:,1:]).unsqueeze(2)*input_vectors
#        sequenceOut=(1-maskSample[:,1:]).unsqueeze(2)*sequenceOut
        # measure accuracy and record loss
        
#        input_vectors_rank=input_vectors.view(-1,input_vectors.shape[-1])
#        targetRank=torch.tensor(range(args.num_seg)).repeat(input_vectors.shape[0]).cuda()
#        rankingFC = nn.Linear(input_vectors.shape[-1], args.num_seg).cuda()
#        out_rank = rankingFC(input_vectors_rank)
        prec1, prec3 = accuracy(output.data, targets, topk=(1, 3))
        acc_mini_batch += prec1.item()
        acc_mini_batch_top3 += prec3.item()
        
        lossClassification = criterion(output, targets)
        
        lossClassification = lossClassification / args.iter_size
        
        #totalLoss=lossMSE
        totalLoss=lossClassification 
        #totalLoss = lossMSE + lossClassification 
        loss_mini_batch_classification += lossClassification.data.item()
        totalLoss.backward()
        totalSamplePerIter +=  output.size(0)
        if (i+1) % args.iter_size == 0:
            # compute gradient and do SGD step
            optimizer.step()
            optimizer.zero_grad()
            lossesClassification.update(loss_mini_batch_classification, totalSamplePerIter)
            top1.update(acc_mini_batch/args.iter_size, totalSamplePerIter)
            top3.update(acc_mini_batch_top3/args.iter_size, totalSamplePerIter)
            batch_time.update(time.time() - end)
            end = time.time()
            loss_mini_batch_classification = 0
            acc_mini_batch = 0
            acc_mini_batch_top3 = 0.0
            totalSamplePerIter = 0.0
            #scheduler.step()
            
        if (i+1) % args.print_freq == 0:
            print('[%d] time: %.3f loss: %.4f' %(i,batch_time.avg,lossesClassification.avg))
          
    print(' * Epoch: {epoch} Prec@1 {top1.avg:.3f} Prec@3 {top3.avg:.3f} Classification Loss {lossClassification.avg:.4f}\n'
          .format(epoch = epoch, top1=top1, top3=top3, lossClassification=lossesClassification))
          
    writer.add_scalar('data/classification_loss_training', lossesClassification.avg, epoch)
    writer.add_scalar('data/top1_training', top1.avg, epoch)
    writer.add_scalar('data/top3_training', top3.avg, epoch)
def validate(val_loader, model, criterion,criterion2,modality):
    batch_time = AverageMeter()
    lossesClassification = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_loader):
            if modality == "rgb" or modality == "pose":
                if "3D" in args.arch or "r2plus1d" in args.arch or 'slowfast' in args.arch:
                    inputs=inputs.view(-1,length,3,input_size,input_size).transpose(1,2)
            elif modality == "flow":
                if "3D" in args.arch or "r2plus1d" in args.arch:
                    inputs=inputs.view(-1,length,2,input_size,input_size).transpose(1,2)
                else:
                    inputs=inputs.view(-1,2*length,input_size,input_size)
            elif modality == "both":
                inputs=inputs.view(-1,5*length,input_size,input_size)
                
            if HALF:
                inputs = inputs.cuda().half()
            else:
                inputs = inputs.cuda()
            targets = targets.cuda()
    
            # compute output
            output, input_vectors, sequenceOut, _ = model(inputs)
            
            
            lossClassification = criterion(output, targets)
    
            # measure accuracy and record loss
            prec1, prec3 = accuracy(output.data, targets, topk=(1, 3))
            
            lossesClassification.update(lossClassification.data.item(), output.size(0))
            
            top1.update(prec1.item(), output.size(0))
            top3.update(prec3.item(), output.size(0))
    
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    
    
        print(' * * Prec@1 {top1.avg:.3f} Prec@3 {top3.avg:.3f} Classification Loss {lossClassification.avg:.4f}\n' 
              .format(top1=top1, top3=top3, lossClassification=lossesClassification))

    return top1.avg, top3.avg, lossesClassification.avg

def save_checkpoint(state, is_best, filename, resume_path):
    cur_path = os.path.join(resume_path, filename)
    torch.save(state, cur_path)
    best_path = os.path.join(resume_path, 'model_best.pth.tar')
    if is_best:
        shutil.copyfile(cur_path, best_path)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 150 epochs"""

    decay = 0.1 ** (sum(epoch >= np.array(args.lr_steps)))
    lr = args.lr * decay
    print("Current learning rate is %4.6f:" % lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def adjust_learning_rate2(optimizer, epoch):
    isWarmUp=epoch < warmUpEpoch
    decayRate=0.2
    if isWarmUp:
        lr=args.lr*(epoch+1)/warmUpEpoch
    else:
        lr=args.lr*(1/(1+(epoch+1-warmUpEpoch)*decayRate))
    
    #decay = 0.1 ** (sum(epoch >= np.array(args.lr_steps)))
    print("Current learning rate is %4.6f:" % lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def adjust_learning_rate3(optimizer, epoch):
    isWarmUp=epoch < warmUpEpoch
    decayRate=0.97
    if isWarmUp:
        lr=args.lr*(epoch+1)/warmUpEpoch
    else:
        lr = args.lr * decayRate**(epoch+1-warmUpEpoch)
    
    #decay = 0.1 ** (sum(epoch >= np.array(args.lr_steps)))
    print("Current learning rate is %4.6f:" % lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def adjust_learning_rate4(optimizer, learning_rate_index):
    """Sets the learning rate to the initial LR decayed by 10 every 150 epochs"""

    decay = 0.1 ** learning_rate_index
    lr = args.lr * decay
    print("Current learning rate is %4.8f:" % lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()
