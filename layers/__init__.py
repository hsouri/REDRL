from .loss import AdversarialLoss, ImageClassificationLoss, AttackDetLoss, PerceptualLoss
from torch import nn
import torch
import pdb


class LossCalculator():
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # Recostruction Loss
        if cfg.Reconstruction_Loss: 
            if cfg.L2:
                self.reconstruction_loss = nn.MSELoss()
            else:
                self.reconstruction_loss = nn.L1Loss()
        # Adversarial Loss
        if cfg.Adversarial_Loss:
            self.adversarial_loss = AdversarialLoss(cfg)
        
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Image Classification Loss 
        #if cfg.Image_Classification_Loss: 
        self.image_classification_loss = ImageClassificationLoss(cfg)
        self.image_classification_loss.to(device)
        if torch.cuda.device_count() > 1 and len(cfg.GPU_IDs) > 1:
            self.image_classification_loss.classifier = nn.DataParallel(self.image_classification_loss.classifier)
        self.image_classification_loss.classifier.eval()
        
        # Percetual Loss
        if cfg.Perceptual_Loss:
            self.perceptual_loss = PerceptualLoss(cfg)
            self.perceptual_loss.to(device)
            if torch.cuda.device_count() > 1 and len(cfg.GPU_IDs) > 1:
                self.perceptual_loss.vgg16 = nn.DataParallel(self.perceptual_loss.vgg16)
            self.perceptual_loss.vgg16.eval()

        # Attack Classification Loss
        if cfg.Attack_Det_Loss:
            self.attack_det_loss = AttackDetLoss(cfg)

        self.loss_value_dict = {'reconstruction_loss': None,
                                'disc_adv_loss': None,
                                'gen_adv_loss': None,
                                'im_cls_loss': None,
                                'perceptual_loss': None,
                                'attack_det_loss': None}

    def att_det_loss(self, prediction, label):
        loss = self.attack_det_loss(prediction, label)
        self.loss_value_dict['attack_det_loss'] = loss.item()
        return loss
    
    def percep_loss(self, recon_img, gt_img):
        loss = self.perceptual_loss(recon_img, gt_img)
        self.loss_value_dict['perceptual_loss'] = loss.item()
        return loss
    
    def disc_adv_loss(self, pred, is_real, write_value=None):
        loss = self.adversarial_loss.compute_loss(pred, is_real)
        if write_value:
            self.loss_value_dict['disc_adv_loss'] = write_value + loss.item()
        return loss

    def gen_adv_loss(self, dis_fake, is_real=True):
        loss = self.adversarial_loss.compute_loss(dis_fake, is_real)
        self.loss_value_dict['gen_adv_loss'] = loss.item()
        return loss

    def recon_loss(self, recon_img, gt_img):
        #loss = self.reconstruction_loss(recon_img, gt_img) 
        loss = torch.norm((recon_img - gt_img).view(recon_img.shape[0], -1), p=1, dim=1).mean()
        self.loss_value_dict['reconstruction_loss'] = loss.item()
        return loss

    def im_cls_loss(self, recon_img, real_img):
        loss = self.image_classification_loss(recon_img, real_img)
        self.loss_value_dict['im_cls_loss'] = loss.item()
        return loss

    def calc_loss(self, recon_img=None, dis_fake=None, target_img=None, residual_pred=None, target_attack_label=None):
        loss = None
        if self.cfg.Reconstruction_Loss:
            loss = self.cfg.Reconstruction_Loss_Lambda * self.recon_loss(recon_img, target_img)
        
        if self.cfg.Perceptual_Loss:
            if loss:
                loss += (self.cfg.Perceptual_Loss_Lambda * self.percep_loss(recon_img, target_img))
            else:
                loss = self.cfg.Perceptual_Loss_Lambda * self.percep_loss(recon_img, target_img)
        
        if self.cfg.Image_Classification_Loss:
            if loss:
                loss += (self.cfg.Image_Classification_Loss_Lambda * self.im_cls_loss(recon_img, target_img))
            else:
                loss = self.cfg.Image_Classification_Loss_Lambda * self.im_cls_loss(recon_img, target_img)
        
        if self.cfg.Adversarial_Loss:
            if loss:
                loss += (self.cfg.Adversarial_Loss_Lambda * self.gen_adv_loss(dis_fake))
            else:
                loss = self.cfg.Adversarial_Loss_Lambda * self.gen_adv_loss(dis_fake)

        if self.cfg.Attack_Det_Loss:
            if loss:
                loss += (self.cfg.Attack_Det_Loss_Lambda * self.att_det_loss(residual_pred, target_attack_label))
            else:
                loss = self.cfg.Attack_Det_Loss_Lambda * self.att_det_loss(residual_pred, target_attack_label)

        return loss
