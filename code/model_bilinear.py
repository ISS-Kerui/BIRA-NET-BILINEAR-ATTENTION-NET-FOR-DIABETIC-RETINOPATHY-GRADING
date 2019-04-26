# encoding: utf-8
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable 
import torch.cuda
import torchvision.transforms as transforms
import torch.nn.functional as F

class Lambda(nn.Module):
	def __init__(self, lambd):
		super(Lambda, self).__init__()
		self.lambd = lambd
	def forward(self, x):
		return self.lambd(x)


class KeNet(nn.Module):
    def __init__(self,classes_num, M, pretrained=True):
        super(KeNet, self).__init__()
        self.num_classes = classes_num
        self.M = M
        features = torchvision.models.resnet50(pretrained=pretrained)
        
        self.conv_features = nn.Sequential(*list(features.children())[:-2])
        
        self.conv = nn.Conv2d(512, self.num_classes * self.M, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(self.num_classes*self.M,self.num_classes)
        self.softmax = nn.Softmax()

        if pretrained:
            for parameter in self.conv_features.parameters():
          
            nn.init.xavier_uniform_(self.conv.weight.data)
            nn.init.constant_(self.conv.bias, 0.1)
            
            nn.init.xavier_uniform_(self.fc.weight)
            nn.init.constant_(self.fc, 0.1)

    def forward(self, input):
        features = self.conv_features(input)
        features = self.conv(features)

      
        features = features.view(features.size(0), self.num_classes, self.M, 14 * 14)


        features_T = torch.transpose(features, 2, 3)
        features = torch.matmul(features, features_T) / (14 * 14)

        features = features.view(features.size(0), self.num_classes*self.M*self.M)
        # 带符号的开方
        features = torch.sign(features) * torch.sqrt(torch.abs(features) + 1e-12)
        # l2正则化
        features = torch.nn.functional.normalize(features)

        #features = features.view(features.size(0), self.num_classes, self.M*self.M)
        fc_out = self.fc(features)
        softmax = self.softmax(fc_out)

        return fc_out, softmax

