# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 17:22:35 2018

@author: daryl.tanyj
"""
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=stride, bias=True)

        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes , kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        self.LeakyReLU = nn.LeakyReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.LeakyReLU(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.LeakyReLU(out)
        out = self.conv2(out)

#        out = self.bn3(out)
#        out = self.LeakyReLU(out)
#        out = self.conv3(out)
#        
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual

        return out


class Hourglass(nn.Module):
    def __init__(self, block, num_blocks, planes, depth):
        super(Hourglass, self).__init__()
        self.depth = depth
        self.block = block
        self.upsample = nn.Upsample(scale_factor=2)
        self.hg = self._make_hour_glass(block, num_blocks, planes, depth)

    def _make_residual(self, block, num_blocks, planes):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(planes*block.expansion, planes))
        return nn.Sequential(*layers)

    def _make_hour_glass(self, block, num_blocks, planes, depth):
        hg = []
        for i in range(4):
            res = []
            for j in range(3):
                res.append(self._make_residual(block, num_blocks, planes))
            if i == 0:
                res.append(self._make_residual(block, num_blocks, planes))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, n, x):
        up1 = self.hg[n-1][0](x)
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n-1][1](low1)

        if n > 1:
            low2 = self._hour_glass_forward(n-1, low1)
        else:
            low2 = self.hg[n-1][3](low1)
        low3 = self.hg[n-1][2](low2)
        up2 = self.upsample(low3)
        out = up1 + up2
        return out

    def forward(self, x):
        return self._hour_glass_forward(self.depth, x)


class HourglassNet(nn.Module):
    def __init__(self, num_stacks=1, num_blocks=1, nlandmarks=18):
        super(HourglassNet, self).__init__()

        self.inplanes = 12
        self.num_feats = 24
        self.num_stacks = num_stacks
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=1, padding=3,
                               bias=True)
        self.bn1 = nn.BatchNorm2d(self.inplanes) 
        self.LeakyReLU = nn.LeakyReLU(inplace=True)
        self.layer1 = self._make_residual(Bottleneck, self.inplanes, 1)
        self.layer2 = self._make_residual(Bottleneck, self.inplanes, 1)
        self.layer3 = self._make_residual(Bottleneck, self.num_feats, 1)
        #self.maxpool = nn.MaxPool2d(2, stride=2)

        # build hourglass modules
        ch = self.num_feats*Bottleneck.expansion
        hg, res, fc, score, fc_, score_ = [], [], [], [], [], []
        for i in range(num_stacks):
            hg.append(Hourglass(Bottleneck, num_blocks, self.num_feats, 4))
            res.append(self._make_residual(Bottleneck, self.num_feats, num_blocks))
            fc.append(self._make_fc(ch, ch))
            score.append(nn.Conv2d(ch, nlandmarks, kernel_size=1, bias=True))
            if i < num_stacks-1:
                fc_.append(nn.Conv2d(ch, ch, kernel_size=1, bias=True))
                score_.append(nn.Conv2d(nlandmarks, ch, kernel_size=1, bias=True))
        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self.score = nn.ModuleList(score)
        self.fc_ = nn.ModuleList(fc_) 
        self.score_ = nn.ModuleList(score_)

    def _make_residual(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_fc(self, inplanes, outplanes):
        bn = nn.BatchNorm2d(inplanes)
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=True)
        return nn.Sequential(
                conv,
                bn,
                self.LeakyReLU,
            )

    def forward(self, x):
        out = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.LeakyReLU(x) 
        x = self.layer1(x)  
        x = self.layer2(x)  
        x = self.layer3(x)  

        for i in range(self.num_stacks):
            y = self.hg[i](x)
            y = self.res[i](y)
            y = self.fc[i](y)
            score = self.score[i](y)
            out.append(score)
            if i < self.num_stacks-1:
                fc_ = self.fc_[i](y)
                score_ = self.score_[i](score)
                x = x + fc_ + score_

        return out

if __name__ == "__main__":
    from torch.autograd import Variable
    import time
    import torch

    model = HourglassNet(nlandmarks=98)
    
    
    input = torch.randn(2, 1, 48, 64)
    mean_fps = torch.zeros(60)
    for i in range(60):
        timenow = time.time()
        out = model(Variable(input))
        mean_fps[i] = (time.time()-timenow)
        print('fps', 1/(time.time()-timenow))
    
    print('output', out[-1].shape)
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    print('mean fps: ', len(mean_fps)/mean_fps.sum().item())
