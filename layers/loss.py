import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from model.discriminator import Discriminator
from model.classifier import *


class PerceptualLoss(nn.Module):
    def __init__(self, cfg, ssp_layer=23, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
            super(PerceptualLoss, self).__init__()
            vgg16 = models.vgg16_bn(pretrained=True)
            self.vgg16 = nn.Sequential(*list(vgg16.features))[:ssp_layer].eval()
            self.mse = nn.MSELoss()
            self.mean = torch.Tensor([[[[mean[0]]],[[mean[1]]],[[mean[2]]]]]).cuda()
            self.std = torch.Tensor([[[[std[0]]],[[std[1]]],[[std[2]]]]]).cuda()
    
    def forward(self, fake_imgs, real_imgs):
        B = fake_imgs.shape[0]
        fake_feat = self.vgg16((fake_imgs - self.mean) / self.std)
        real_feat = self.vgg16((real_imgs - self.mean) / self.std)
        loss = torch.norm((fake_feat - real_feat).view(B,-1), dim=1).mean()
        return loss


class ImageClassificationLoss(nn.Module):
    def __init__(self, cfg, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super(ImageClassificationLoss, self).__init__()
        self.classifier = ImageClassifier(cfg).eval()
        self.cls_loss = nn.CrossEntropyLoss()
        self.mean = torch.Tensor([[[[mean[0]]],[[mean[1]]],[[mean[2]]]]]).cuda()
        self.std = torch.Tensor([[[[std[0]]],[[std[1]]],[[std[2]]]]]).cuda()

    def forward(self, fake_imgs, real_imgs):
        fake_predictions = self.classifier((fake_imgs - self.mean) / self.std)
        real_predictions = self.classifier((real_imgs - self.mean) / self.std)
        targets = torch.max(real_predictions, dim=1)[1]
        loss = self.cls_loss(fake_predictions, targets)
        return loss


class AttackDetLoss(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(self.cfg.Device)
        self.loss = nn.CrossEntropyLoss()

        if self.cfg.Label_Smoothing:
            self.num_classes = self.cfg.Num_Attack_Types + 1
            self.epsilon = self.cfg.Label_Smoothing_epsilon
            self.logsoftmax = nn.LogSoftmax(dim=1)
        
    def __call__(self, predictions, targets=None):
        if not self.cfg.Label_Smoothing:
            loss = self.loss(predictions, targets)
        else:
            log_probs = self.logsoftmax(predictions)
            targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1).to(targets.device)
            targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
            loss = (-targets * log_probs).mean(0).sum()
        return loss


class AdversarialLoss(nn.Module):
    # DCGAN Loss
    def __init__(self, cfg, target_real_label=1.0, target_fake_label=0.0):
        super(AdversarialLoss, self).__init__()
        self.cfg = cfg
        self.gan_mode = cfg.Gan_Mode
        self.register_buffer('real_label', torch.tensor(target_real_label).cuda())
        self.register_buffer('fake_label', torch.tensor(target_fake_label).cuda())
        self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)
    
    def compute_loss(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['Vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        else:
            raise NotImplementedError('gan mode %s not implemented' % self.gan_mode)
        return loss
        
    def loss_dcgan_dis(self, dis_fake, dis_real):
        L1 = torch.mean(F.softplus(-dis_real))
        L2 = torch.mean(F.softplus(dis_fake))
        return (L1 + L2) / 2

    def loss_dcgan_gen(self, dis_fake):
        loss = torch.mean(F.softplus(-dis_fake))
        return loss





class CLSLoss(object):
    def __init__(self, num_classes=2, ls=False, eps=0.1):
        self.cls_loss = nn.CrossEntropyLoss()
        self.device = torch.device(self.cfg.Device)
        self.ls = ls

        if self.ls:
            self.num_classes = num_classes
            self.epsilon = eps
            self.logsoftmax = nn.LogSoftmax(dim=1)
        
    def __call__(self, predictions, targets):
        if not self.ls:
            loss = self.cls_loss(predictions, targets)
        else:
            log_probs = self.logsoftmax(predictions)
            targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1).to(self.device)
            targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
            loss = (-targets * log_probs).mean(0).sum()
        return loss

