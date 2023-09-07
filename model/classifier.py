import torch
from torchvision import models
from torch import nn
import os.path as osp
import sys
from tools.path import Paths

import pdb


class ImageClassifier(nn.Module):
    def __init__(self, cfg):
        super(ImageClassifier, self).__init__()
        self.cfg = cfg
        self.dataset = cfg.Dataset
        self.model_name = cfg.Image_Classifier
        self.num_classes = {'MNIST': 10, 'CIFAR10': 10, 'TinyImageNet': 200, 'RESISC45': 45}[self.dataset]
        self.model = getattr(models, self.model_name)()
        if self.model_name == 'vgg19_bn':
            self.model.classifier = nn.Sequential(nn.Linear(25088, 4096),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(p=0.5, inplace=False),
                                        nn.Linear(4096, 256),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(p=0.5, inplace=False),
                                        nn.Linear(256, self.num_classes))
    
        elif self.model_name in ['resnet50', 'resnext50_32x4d', 'inception_v3']:
            self.model.fc = nn.Linear(2048, self.num_classes)
            if self.model_name == 'inception_v3':
                self.model.aux_logits = False
        
        elif self.model_name == 'densenet121':
            self.model.classifier = nn.Linear(1024, self.num_classes)

        self.load_weights()

        # No need to update Image Classifier Weights!
        for param in self.model.parameters():
            param.requires_grad = False
        
    def load_weights(self):
        ckpt = torch.load(osp.join(Paths.ModelsPath,'{}_{}_64.pth.tar'.format(self.dataset, self.model_name)), map_location='cpu')['net_state_dict']
        self.model.load_state_dict(ckpt)
        print('ImageClassifier {} weights for {} is loaded!'.format(self.model_name, self.dataset))

    def forward(self, x):
        return self.model(x)


class AttackClassifier(nn.Module):
    def __init__(self, cfg):
        super(AttackClassifier, self).__init__()
        self.cfg = cfg
        if self.cfg.Attack_Classifier != 'linear':
            self.feat = nn.Sequential(
                *list(getattr(models, self.cfg.Attack_Classifier)(
                    pretrained=self.cfg.Attack_Classifier_Pretrained).children())[0:9]) # gives 512 long feature vector
            
            if self.cfg.Attack_Classifier_Cat:
                self.feat[0] = nn.Conv2d(6, 64, 7, 2, 3)
            self.cls = nn.Sequential(nn.Linear(512, 32, bias=True), nn.ReLU(), nn.Linear(32, self.cfg.Num_Attack_Types + 1))
        else:
            W, H = self.cfg.Input_Size
            num_dim = W * H * 3 * 2 if self.cfg.Attack_Classifier_Cat else W * H * 3
            self.cls = nn.Linear(num_dim, self.cfg.Num_Attack_Types + 1)
        
        device = torch.device(self.cfg.Device)
        self.to(device)

    def forward(self, recon_img=None, input_img=None, clean_img=None):
        if not self.cfg.Baseline:
            res_img = input_img - torch.clamp(recon_img, min=0, max=1)
        else:
            res_img = input_img - clean_img

        if self.cfg.Attack_Classifier != 'linear':
            if self.cfg.Attack_Classifier_Cat:
                feat = self.feat(torch.cat([res_img, input_img], dim=1))
            elif self.cfg.Baseline and self.cfg.Baseline_Source == 'RES':
                feat = self.feat(res_img)
            elif self.cfg.Baseline and self.cfg.Baseline_Source == 'IMG':
                feat = self.feat(input_img)
            else:
                feat = self.feat(res_img)
        else:
            if self.cfg.Attack_Classifier_Cat:
                feat = torch.cat([res_img, input_img], dim=1)
            else:
                feat = res_img
                if self.cfg.Baseline and self.cfg.Baseline_Source == 'IMG':
                    feat = input_img
        
        feat = feat.view(feat.shape[0], -1)
        return self.cls(feat)





