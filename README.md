# LateTemporalModeling3DCNN

Pytorch implementation of [Late Temporal Modeling in 3D CNN Architectures with BERT for Action Recognition]

This the repository which implements late temporal modeling on top of the 3D CNN architectures and mainly focus on BERT for this aim. 

## Installation
	#For the installation, you need to install conda. The environment may contain also unnecessary packages but we want to give complete environment that we are using. 

	#Create the environment with the command
	conda env create -f LateTemporalModeling3D.yml

	#Then you can activate the environment with the command
	conda activate LateTemporalModeling3D

Later, please download the necessary files from the link, and copy them into the main directory.
https://1drv.ms/u/s!AqKP51Rjkz1Gaifd54VbdRBn6qM?e=7OxYLa

## Dataset Format
In order to implement training and validation the list of training and validation samples should be created as txt files in \datasets\settings folder. 

As an example, the settings for hmdb51 is added. In the file names train-val, modality and split of the dataset is specified. In the txtx files, the folder of images, the number of frames in the folder and the index of the class is denoted, respectively.

The image folders should be created in \datasets as and hmdb51_frames and ucf101_frames for hmdb and ucf datasets. If you want use this code in separate dataset, there is a need to create .py file like hmdb51.py and ucf101.py existing in \dataset. You can just copy these py. files and change the class name of the dataset. Then the init file in the dataset should be modified properly.  

The format of the rgb and flow images is determined for hmdb51 as
"img_%05d.jpg", "flow_x_%05d", "flow_y_%05d" 
and for ucf101 as 
"img_%05d.jpg", "flow_x_%05d.jpg", "flow_y_%05d.jpg" 

Name patterns of the dataset can be modified but the test of the datasets should also be modified. These are also specified in the variable called extension in the test files.

##Training of the dataset
There are two seperate training files called two_stream2.py and two_stream_bert2.py. These are almost identical two training files. Select the first for SGD training and select the second for ADAMW trainings. 

#### Models
For the models listed below, use two_stream2.py

- **rgb_resneXt3D64f101**
- **flow_resneXt3D64f101**
- **rgb_slowfast64f_50**
- **rgb_I3D64f**
- **flow_I3D64f**
- **rgb_r2plus1d_32f_34**

For the models listed below, use two_stream_bert2.py
- **rgb_resneXt3D64f101_bert10_FRAB**
- **flow_resneXt3D64f101_bert10_FRAB**
- **rgb_resneXt3D64f101_bert10_FRMB**
- **flow_resneXt3D64f101_bert10_FRMB**
- **rgb_resneXt3D64f101_FRMB_adamw**
- **rgb_resneXt3D64f101_adamw**
- **rgb_resneXt3D64f101_FRMB_NLB_concatenation**
- **rgb_resneXt3D64f101_FRMB_lstm**
- **rgb_resneXt3D64f101_concatenation**

- **rgb_slowfast64f_50_bert10_FRAB_late**
- **rgb_slowfast64f_50_bert10_FRAB_early**
- **rgb_slowfast64f_50_bert10_FRMB_early**
- **rgb_slowfast64f_50_bert10_FRMB_late**

- **rgb_I3D64f_bert2**
- **flow_I3D64f_bert2**
- **rgb_I3D64f_bert2_FRMB**
- **flow_I3D64f_bert2_FRMB**
- **rgb_I3D64f_bert2_FRAB**
- **flow_I3D64f_bert2_FRAB**

- **rgb_r2plus1d_32f_34_bert10**
- **rgb_r2plus1d_64f_34_bert10**

#### Training Commands

python two_stream_bert2.py --split=1 --arch=rgb_resneXt3D64f101_bert10_FRMB --workers=2 --batch-size=8 --iter-size=16 --print-freq=400 --dataset=hmdb51 --lr=1e-5

python two_stream2.py --split=1 --arch=rgb_resneXt3D64f101 --workers=2 --batch-size=8 --iter-size=16 --print-freq=400 --dataset=hmdb51 --lr=1e-2


For multi-gpu training, comment the two lines below
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

To continue the training from the best model, add -c. 
To evaluate the single clip single crop performance of best model, add -e

## Test of the dataset
For the test of the files, there are three seperate files, namely 
spatial_demo3D.py  -- which is for multiple clips test
spatial_demo_bert.py  -- which is for single clip test
combined_demo.py  -- which is for two-stream tests

Firstly, set your current directory where the test files exists which is:
scripts/eval/

Then enter the example commands below:
python spatial_demo3D.py --arch=rgb_resneXt3D64f101_bert10_FRMB --split=2

python spatial_demo_bert.py --arch=flow_resneXt3D64f101_bert10_FRMB --split=2

python combined_demo.py --arch_rgb=rgb_resneXt3D64f101_bert10_FRMB  --arch_flow=flow_resneXt3D64f101_bert10_FRMB --split=2

If your training is implemented with multi-GPU, manually set multiGPUTrain to True
As default, the tests are implemented ten crops. For single crops test, manually set ten_crop_enabled to False

## Related Projects
[Toward Good Practices](https://github.com/bryanyzhu/two-stream-pytorch): PyTorch implementation of popular two-stream frameworks for video action recognition

[ResNeXt101](https://github.com/kenshohara/video-classification-3d-cnn-pytorch): Video Classification Using 3D ResNet

[SlowFast](https://github.com/facebookresearch/SlowFast): PySlowFast

[R2+1D-IG65](https://github.com/moabitcoin/ig65m-pytorch): IG-65M PyTorch

[I3D](https://github.com/piergiaj/pytorch-i3d): I3D models trained on Kinetics












