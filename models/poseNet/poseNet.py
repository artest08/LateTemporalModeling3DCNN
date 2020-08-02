import torch.nn as nn
import collections, torch, torchvision, numpy, h5py
import math
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict 
#from .BERT.bert import BERT

__all__ = ['openPoseL2Part','openPose']

config = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'part1': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512],
    'part2': [256,128],
}

class OpenPose(nn.Module):
    def __init__(self):
        super(OpenPose, self).__init__()
        self.feature=VGG()
        self.L2Part=L2Part(128,52)
        self.L1Part=L1Part(128+52,26)
    def forward(self, x):
        features=self.feature(x)
        L2Out=self.L2Part(features)
        L1Out=self.L1Part(features,L2Out)
        return L1Out,L2Out
        
# class rgb_poseNet_bert(nn.Module):
#     def __init__(self, num_classes , length):
#         super(rgb_poseNet_bert, self).__init__()
#         self.hidden_size=1225
#         self.n_layers=4
#         self.attn_heads=25
#         self.num_classes=num_classes
#         self.length=length
#         self.dp = nn.Dropout(p=0.8)
        
#         self.bert = BERT(self.hidden_size,length, hidden=self.hidden_size, n_layers=self.n_layers, attn_heads=self.attn_heads)
#         self.fc_action = nn.Linear(self.hidden_size, num_classes)
#         self.openPose=openPose()
#         self.avgpool = nn.AvgPool2d(4)
        
#         for param in self.openPose.parameters():
#             param.requires_grad = False
        
        
#     def forward(self, x):        
#         x,_=self.openPose(x)
#         x=self.avgpool(x[:,:25,:,:])
#         x=x.view(-1,self.length,self.hidden_size)
#         input_vectors=x
#         output , maskSample = self.bert(x)
#         classificationOut = output[:,0,:]
#         sequenceOut=output[:,1:,:]
#         output=self.dp(classificationOut)
#         x = self.fc_action(output)
#         return x, input_vectors, sequenceOut, maskSample
    
    
# class rgb_poseNet_bert2(nn.Module):
#     def __init__(self, num_classes , length):
#         super(rgb_poseNet_bert2, self).__init__()
#         self.hidden_size=49
#         self.n_layers=4
#         self.attn_heads=7
#         self.num_classes=num_classes
#         self.length=length*25
#         self.dp = nn.Dropout(p=0.8)
        
#         self.bert = BERT(self.hidden_size,self.length, hidden=self.hidden_size, n_layers=self.n_layers, attn_heads=self.attn_heads)
#         self.fc_action = nn.Linear(self.hidden_size, num_classes)
#         self.openPose=openPose()
#         self.avgpool = nn.AvgPool2d(4)
        
#         for param in self.openPose.parameters():
#             param.requires_grad = False
        
        
#     def forward(self, x):        
#         x,_=self.openPose(x)
#         x=self.avgpool(x[:,:25,:,:])
#         x=x.view(-1,self.length,self.hidden_size)
#         input_vectors=x
#         output , maskSample = self.bert(x)
#         classificationOut = output[:,0,:]
#         sequenceOut=output[:,1:,:]
#         output=self.dp(classificationOut)
#         x = self.fc_action(output)
#         return x, input_vectors, sequenceOut, maskSample

        
class L2Part(nn.Module):
    
    def __init__(self, in_channels, stage_out_channels):
        super(L2Part, self).__init__()
        self.firstStage=stage(0,in_channels,96,in_channels*2,stage_out_channels,'L2')
        self.secondStage=stage(1,in_channels+stage_out_channels,in_channels,in_channels*4,stage_out_channels,'L2')
        self.thirdStage=stage(2,in_channels+stage_out_channels,in_channels,in_channels*4,stage_out_channels,'L2')
        self.fourthStage=stage(3,in_channels+stage_out_channels,in_channels,in_channels*4,stage_out_channels,'L2')
    def forward(self, features):
        x=self.firstStage(features)
        x=torch.cat([features, x], 1)
        x=self.secondStage(x)
        x=torch.cat([features, x], 1)
        x=self.thirdStage(x)
        x=torch.cat([features, x], 1)
        out=self.fourthStage(x)
        return out
    
class L1Part(nn.Module):
    
    def __init__(self, in_channels, stage_out_channels):
        super(L1Part, self).__init__()
        self.firstStage=stage(0,in_channels,96,256,stage_out_channels,'L1')
        self.secondStage=stage(1,in_channels+stage_out_channels,128,512,stage_out_channels,'L1')
    def forward(self, features,L2Out):
        x=torch.cat([features, L2Out], 1)
        x=self.firstStage(x)
        x=torch.cat([features, x, L2Out], 1)
        out=self.secondStage(x)
        return out
    
class stage(nn.Module):
    
    def __init__(self,stageID,in_channels,out_channels_perSub,mid_channels,out_channels,appendix):
        super(stage, self).__init__()
        self.firstConcat=concatLayer(in_channels,out_channels_perSub,1,stageID,appendix)
        self.secondConcat=concatLayer(3*out_channels_perSub,out_channels_perSub,2,stageID,appendix)
        self.thirdConcat=concatLayer(3*out_channels_perSub,out_channels_perSub,3,stageID,appendix)
        self.fourthConcat=concatLayer(3*out_channels_perSub,out_channels_perSub,4,stageID,appendix)
        self.fifthConcat=concatLayer(3*out_channels_perSub,out_channels_perSub,5,stageID,appendix)
        conv2d = nn.Conv2d(3*out_channels_perSub, mid_channels, kernel_size=1, padding=0)
        prelu= nn.PReLU(mid_channels)
        self.afterConcatsFirst=nn.Sequential(OrderedDict({'Mconv6_stage%d_%s' %(stageID,appendix) :conv2d,'Mprelu6_stage%d_%s' %(stageID,appendix):prelu}))
        conv2d = nn.Conv2d(mid_channels, out_channels, kernel_size=1, padding=0)
        self.afterConcatsSecond=nn.Sequential(OrderedDict({'Mconv7_stage%d_%s' %(stageID,appendix) :conv2d}))       
    def forward(self,x):
        x=self.firstConcat(x)
        x=self.secondConcat(x)
        x=self.thirdConcat(x)
        x=self.fourthConcat(x)
        x=self.fifthConcat(x)
        x=self.afterConcatsFirst(x)
        out=self.afterConcatsSecond(x)
        return out
    
        
class concatLayer(nn.Module):
    
    def __init__(self,in_channels,out_channels_perSub,i,j,appendix):
        super(concatLayer, self).__init__()
        self.firstSub=self.concatLayerSub(in_channels,out_channels_perSub,'%d_stage%d_' %(i,j) + appendix+'_0')
        self.secondSub=self.concatLayerSub(out_channels_perSub,out_channels_perSub,'%d_stage%d_' %(i,j)+appendix+'_1')
        self.thirdSub=self.concatLayerSub(out_channels_perSub,out_channels_perSub,'%d_stage%d_' %(i,j)+appendix+'_2')
    def forward(self, x):
        firstSub=self.firstSub(x)
        secondSub=self.secondSub(firstSub)
        thirdSub=self.thirdSub(secondSub)
        out=torch.cat([firstSub, secondSub, thirdSub], 1)
        return out
      
    def concatLayerSub(self,in_channels,out_channels,layerName):
        concatLayerSubOrdered=OrderedDict()
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        concatLayerSubOrdered.update({'Mconv'+layerName:conv2d})
        concatLayerSubOrdered.update({'Mprelu'+layerName: nn.PReLU(out_channels)})
        return nn.Sequential(concatLayerSubOrdered)

class VGG(nn.Module):
    
    def __init__(self):
        super(VGG, self).__init__()
        self.vggPart1=self._vggPart1()
        self.conv4_2=nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.prelu4_2= nn.PReLU(512)
        self.vggPart2=self._vggPart2()
        self._initialize_weights()

    def forward(self, x):
        x = self.vggPart1(x)
        x = self.conv4_2(x)
        x = self.prelu4_2(x)
        x = self.vggPart2(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()          
    def _vggPart1(self):
        vggOrdered=OrderedDict()
        in_channels = 3
        cfg=config['part1']
        i=1
        j=1
        for v in cfg:
            if v == 'M':
                vggOrdered.update({'pool%d_stage1'%(i): nn.MaxPool2d(kernel_size=2, stride=2)})
                i=i+1
                j=1
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                vggOrdered.update({'conv%d_%d' %(i,j):conv2d})
                vggOrdered.update({'relu%d_%d' %(i,j): nn.ReLU(inplace=True)})
                in_channels = v
                j=j+1
        return nn.Sequential(vggOrdered)
    def _vggPart2(self):
        vggOrdered=OrderedDict()
        in_channels = 512
        cfg=config['part2']
        i=4
        j=3
        for v in cfg:
            if v == 'M':
                vggOrdered.update({'pool%d_stage1'%(i): nn.MaxPool2d(kernel_size=2, stride=2)})
                i=i+1
                j=1
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                vggOrdered.update({'conv%d_%d_CPM' %(i,j):conv2d})
                vggOrdered.update({'prelu%d_%d_CPM' %(i,j): nn.PReLU(v)})
                in_channels = v
                j=j+1
        return nn.Sequential(vggOrdered)

        
def openPose(pretrained=True, **kwargs):    
    model = OpenPose()
    model_dict = model.state_dict()
    if pretrained:
        weightFile='models/openPose.h5'
        state_dict = h5py.File(weightFile, 'r')
        pretrained_dict={l : torch.from_numpy(numpy.array(v)).view_as(p) for k, v in state_dict.items() for l, p in model.named_parameters() if k in l}
        model_dict.update(pretrained_dict) 
        model.load_state_dict(model_dict)
    return model

def openPoseL2Part():
    model=openPose()
    newModel=nn.Sequential(*list(model.children())[:-1])
    return newModel



