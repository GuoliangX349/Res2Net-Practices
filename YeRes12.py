import os
import sys
import numpy as np

import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter

SRM_npy = np.load('/home/linux123/Documents/SiaStegNet_mymodi/src/models/SRM_Kernels.npy')

class SRM_conv2d(nn.Module):
    def __init__(self, stride=1, padding=0): # padding =0 for default
        super(SRM_conv2d, self).__init__()
        self.in_channels = 1
        self.out_channels = 30
        self.kernel_size = (5, 5)
        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding
        self.dilation = (1,1)
        self.transpose = False
        self.output_padding = (0,)
        self.groups = 1
        self.weight = Parameter(torch.Tensor(30, 1, 5, 5), \
                                requires_grad=True)
        self.bias = Parameter(torch.Tensor(30), \
                              requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.numpy()[:] = SRM_npy
        self.bias.data.zero_()

    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, \
                        self.stride, self.padding, self.dilation, \
                        self.groups)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, \
                 stride=1, with_bn=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, \
                              stride)
        self.relu = nn.ReLU()
        self.with_bn = with_bn
        if with_bn:
            self.norm = nn.BatchNorm2d(out_channels)
        else:
            self.norm = lambda x: x
        self.reset_parameters()

    def forward(self, x):
        return self.norm(self.relu(self.conv(x)))

    def reset_parameters(self):
        nn.init.xavier_uniform(self.conv.weight)
        self.conv.bias.data.fill_(0.2)
        if self.with_bn:
            self.norm.reset_parameters()

class YeNet(nn.Module):
    def __init__(self, with_bn=False, threshold=3):
        super(YeNet, self).__init__()
        self.with_bn = with_bn
        self.preprocessing = SRM_conv2d(1, 0)
        self.TLU = nn.Hardtanh(-threshold, threshold, True)
        if with_bn:
            self.norm1 = nn.BatchNorm2d(30)
        else:
            self.norm1 = lambda x: x
            
        self.block2 = ConvBlock(30, 30, 3, with_bn=self.with_bn)
        self.block3 = ConvBlock(30, 30, 3, with_bn=self.with_bn)
        self.block4 = ConvBlock(30, 30, 3, with_bn=self.with_bn)
        self.pool1 = nn.AvgPool2d(2, 2)
        self.block5 = ConvBlock(30, 32, 5, with_bn=self.with_bn)
        self.pool2 = nn.AvgPool2d(3, 2)
        self.block6 = ConvBlock(32, 32, 5, with_bn=self.with_bn)
        self.pool3 = nn.AvgPool2d(3, 2)
        self.block7 = ConvBlock(32, 32, 5, with_bn=self.with_bn)
        self.pool4 = nn.AvgPool2d(3, 2)
        self.block8 = ConvBlock(32, 16, 3, with_bn=self.with_bn)
        self.block9 = ConvBlock(16, 16, 3, 3, with_bn=self.with_bn)
        self.ip1 = nn.Linear(3 * 3 * 16, 2)
        self.reset_parameters()

    def forward(self, x):
        x = x.float()
        x = self.preprocessing(x)
        x = self.TLU(x)
        x = self.norm1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool1(x)
        x = self.block5(x)
        x = self.pool2(x)
        x = self.block6(x)
        x = self.pool3(x)
        x = self.block7(x)
        x = self.pool4(x)
        x = self.block8(x)
        x = self.block9(x)
        x = x.view(x.size(0), -1)
        x = self.ip1(x)
        return x

    def reset_parameters(self):
        for mod in self.modules():
            if isinstance(mod, SRM_conv2d) or \
                    isinstance(mod, nn.BatchNorm2d) or \
                    isinstance(mod, ConvBlock):
                mod.reset_parameters()
            elif isinstance(mod, nn.Linear):
                nn.init.normal(mod.weight, 0. ,0.01)
                mod.bias.data.zero_()

class Bottle2neck(nn.Module):
    expansion = 4
                                     
    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=36, scale = 4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth/64.0))) # n=s*w, n is the channel number
        self.conv1 = nn.Conv2d(inplanes, width*scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width*scale)
        
        if scale == 1:
          self.nums = 1
        else:
          self.nums = scale -1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride = stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
          convs.append(nn.Conv2d(width, width, kernel_size=3, stride = stride, padding=1, bias=False))
          bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width*scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width  = width

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)

        for i in range(self.nums):
          if i==0 or self.stype=='stage':
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(self.bns[i](sp))
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype=='normal':
          out = torch.cat((out, spx[self.nums]),1)
        elif self.scale != 1 and self.stype=='stage':
          out = torch.cat((out, self.pool(spx[self.nums])),1)

        out = self.conv3(out)
        out = self.bn3(out)
        #print(out.shape) #[16,120,256,256]
        #print(residual.shape)
        if self.downsample is not None:
            residual = self.downsample(x)
            
        out += residual
        out = self.relu(out)

        return out
    
class Bottle2neck2(nn.Module):
    expansion = 4
    #My copy version
    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale = 2, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck2, self).__init__()

        width = int(math.floor(planes * (baseWidth/64.0))) # n=s*w, n is the channel number
        self.conv1 = nn.Conv2d(inplanes, width*scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width*scale)
        
        if scale == 1:
          self.nums = 1
        else:
          self.nums = scale -1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride = stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
          convs.append(nn.Conv2d(width, width, kernel_size=3, stride = stride, padding=1, bias=False))
          bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width*scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width  = width
        self.conv_shink = nn.Conv2d(planes * self.expansion, inplanes, kernel_size=1, bias=False)

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)

        for i in range(self.nums):
          if i==0 or self.stype=='stage':
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(self.bns[i](sp))
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype=='normal':
          out = torch.cat((out, spx[self.nums]),1)
        elif self.scale != 1 and self.stype=='stage':
          out = torch.cat((out, self.pool(spx[self.nums])),1)

        out = self.conv3(out)
        out = self.bn3(out)
        
        out=self.conv_shink(out) # modified XGL
        # print(out.shape) #[16,120,256,256]
        # print(residual.shape)
        if self.downsample is not None:
            residual = self.downsample(x)
        
        
        out += residual
        out = self.relu(out)

        return out


    
    

class Res2Net(nn.Module):

    #def __init__(self, block, layers, baseWidth = 26, scale = 4, num_classes=1000):
    def __init__(self, block, layers, baseWidth = 26, scale = 4, num_classes=2,p=0.5):
        self.inplanes = 32
        #self.inplanes = 30
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        
        #self.preprocessing = SRM_conv2d(1, 0) # padding =2    
        
        self.conv = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1,
                        bias=False)
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
                
        
      
        
        # # block 1
        self.conv_block1 =Bottle2neck(32,8,1)  # 32/8=4
        
        self.conv12 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0,
                        bias=False)
        self.bn12 = nn.BatchNorm2d(32)
        
        self.conv13= nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0,
                        bias=False)
        self.bn13 = nn.BatchNorm2d(32)
        
        # # block 2
        self.conv_block2 =Bottle2neck(32,8,1)  # 
        #
        self.conv22 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0,
                        bias=False)
        self.bn22 = nn.BatchNorm2d(32)
        
  
        
        
        #self.maxpool23 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.conv_block3 =Bottle2neck(32,8,1)  
        #self.conv_block3 =Bottle2neck2(64,16,1)  
        self.conv32 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0,
                        bias=False)
        self.bn32 = nn.BatchNorm2d(32)
        
        # self.conv33 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0,
        #                 bias=False)
        # self.bn33 = nn.BatchNorm2d(32)
        
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        
        
        
        #self.conv_block4 =Bottle2neck(32,8,1)  
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1,
                        bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1,
                        bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)
        
    

        
        # self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1,
        #                 bias=False)
        # self.bn3 = nn.BatchNorm2d(32)
        # self.relu3 = nn.ReLU(inplace=True)
        
        
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=0)
        
        self.layer1 = self._make_layer(block, 32, layers[0])
        
        # self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1,
        #                 bias=False)
        # self.bn6 = nn.BatchNorm2d(128)
        # self.relu6 = nn.ReLU(inplace=True)
        
        
        
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # #self.conv_block4 =Bottle2neck2(32,32,1)  # 
        #
        
        self.conv4 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0,
                        bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.relu4 = nn.ReLU(inplace=True)
        

  
        #self.conv_block4 =Bottle2neck2(64,32,1)  # 

        
        
        # self.conv_block5 =Bottle2neck2(48,48,1)  #   
        
        # self.conv_block6 =Bottle2neck2(48,48,1)  #   
        
        # # self.conv_block7 =Bottle2neck2(48,48,1)  #   
        
        


        # self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        #self.layer5 = self._make_layer(block, 1024, layers[3], stride=2)
        #self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool2d = nn.AdaptiveMaxPool2d(1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc = nn.Linear(512*4+1, num_classes)# the number should match fc's shape
        self.dropout = nn.Dropout(p=p)
        self.reset_parameters()# if without this, the network would probably converge at 24.
        # with reset_parameters, the network would converge for SUNI 0.4 at 25 epochs.
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.xavier_uniform_(m.weight)
                # nn.init.constant_(m.bias, 0.2)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample, 
                        stype='stage', baseWidth = self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth = self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)
    
    
    def extract_feat(self, x):
        ## For 0p5:
        ## This version has 2.71M paras, but achieves 82.46% acc for HILL.
        # 79.30% for MIPOD at 150 epoch.
        # 88.69% for WOW at 150 epoch.
        
        #For 0p4:
        # 500epoch    
        x = x.float()
        #x = self.preprocessing(x)
        x=self.conv(x)
        x = self.bn(x)
        x = self.relu(x)  
        
        #x=self.conv12(x)
        
        #x0=x
        # block 1
        x=self.conv_block1(x)
        x=self.conv12(x)
        x = self.bn12(x)
        
        x=self.conv13(x)
        x = self.bn13(x)
        
        x=self.conv_block2(x)
        x=self.conv22(x)
        x = self.bn22(x)
        
        #x=self.maxpool23(x)
        
        x=self.conv_block3(x)
        x=self.conv32(x)
        x = self.bn32(x)
        
        # x=self.conv33(x)
        # x = self.bn33(x)
        x=self.maxpool(x)
        
       # x1=x;
        
        x=self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        #print(x.shape)
        x=self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        #print(x.shape)
        # #x=xc1+x;
        
        # x=self.conv12(x)
        # x = self.bn12(x)

        #x=x+x1;
        # x=self.conv4(x)
        # x = self.bn4(x)
        # x = self.relu4(x)

        # #x=x+x1+x0+x2+x3+x4+x5
        # x_0 = self.avgpool(x0)
        # x_1 = self.avgpool(x1)
        # x_2 = self.avgpool(x2)
        # x_3 = self.avgpool(x3)
        # x_4 = self.avgpool(x4)
        # x_5 = self.avgpool(x5)
        # x_6 = self.avgpool(x)
        # x=x_0+x_1+x_2+x_3+x_4+x_5+x_6
        x=self.avgpool(x)
       
        x = self.layer1(x)
        #print(x.shape)
        
        # x=self.conv6(x)
        # x = self.bn6(x)
        # x = self.relu6(x)
        
        x = self.layer2(x)
        
        x = self.layer3(x)
        #print(x.shape)
        
        x=self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        
        # x=self.conv5(x)
        # x = self.bn5(x)
        # x = self.relu5(x)
        
        #print(x.shape)
        # x = self.layer3(x)
        # x = self.layer4(x)

        # # x1 = self.avgpool(x1)
        # # x2 = self.avgpool(x2)
        # # x3 = self.avgpool(x3)
        # # x4 = self.avgpool(x4)
        # # x5 = self.avgpool(x5)

        x=self.maxpool2d(x)
  
        x = x.view(x.size(0), -1)
        #print(x.shape)
        
        # x = self.fc(x)
     
        return x
    
    def forward(self, *args):
        ############# statistics fusion start #############
        feats = torch.stack(
            [self.extract_feat(subarea) for subarea in args], dim=0
        )

        euclidean_distance = F.pairwise_distance(feats[0], feats[1], eps=1e-6,
                                                 keepdim=True)

        if feats.shape[0] == 1:
            final_feat = feats.squeeze(dim=0)
        else:
            # feats_sum = feats.sum(dim=0)
            # feats_sub = feats[0] - feats[1]
            feats_mean = feats.mean(dim=0)
            feats_var = feats.var(dim=0)
            feats_min, _ = feats.min(dim=0)
            feats_max, _ = feats.max(dim=0)

            '''feats_sum = feats.sum(dim=0)
            feats_sub = abs(feats[0] - feats[1])
            feats_prod = feats.prod(dim=0)
            feats_max, _ = feats.max(dim=0)'''
            
            #final_feat = torch.cat(
            #    [feats[0], feats[1], feats[0], feats[1]], dim=-1
            #    #[euclidean_distance, feats_sum, feats_sub, feats_prod, feats_max], dim=-1
            #)

            final_feat = torch.cat(
                [euclidean_distance, feats_mean, feats_var, feats_min, feats_max], dim=-1
                #[euclidean_distance, feats_sum, feats_sub, feats_prod, feats_max], dim=-1
            )

        out = self.dropout(final_feat)
        # out = self.fcfusion(out)
        # out = self.relu(out)
        out = self.fc(out)

        return out, feats[0], feats[1]




def accuracy(outputs, labels):
    _, argmax = torch.max(outputs, 1)
    return (labels == argmax.squeeze()).float().mean()
