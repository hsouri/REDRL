import torch
from torchvision.utils import make_grid
import pdb
from torchvision.utils import save_image
import os.path as osp
import os


def set_requires_grad(net, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        net -- a networks
        requires_grad (bool) -- whether the networks require gradients or not
    """
    if net is not None:
        for param in net.parameters():
            param.requires_grad = requires_grad
    return net

def train_iter(cfg, iteration, in_batch, reconstructor, attack_classifier, optimizer, loss_func, 
    discriminator=None, disc_optimizer=None, disc_scheduler=None, writer=None):
    adv_imgs, clean_imgs, labels, att_labels = in_batch
    if cfg.Device == 'cuda':
        adv_imgs, clean_imgs = adv_imgs.cuda(), clean_imgs.cuda()
        labels, att_labels = labels.cuda(), att_labels.cuda()
    
    if not cfg.Baseline:
        # Reconstruction
        if 'SegNet' in cfg.Generator_Type:
            recon_imgs = reconstructor(adv_imgs)
            KLD = None
        else:
            recon_imgs = reconstructor(adv_imgs)
            KLD = None
        
        residual_pred = attack_classifier(recon_imgs, adv_imgs, clean_imgs)

        report = ""
        if cfg.Adversarial_Loss:
            # Update Discriminator
            set_requires_grad(discriminator, True)
            disc_optimizer.zero_grad()
            # Fake
            # We use conditional GANs; we need to feed both input and output to the discriminator
            fake_AB = torch.cat((adv_imgs, torch.clamp(recon_imgs, min=0, max=1)), dim=1)
            pred_fake = discriminator(fake_AB.detach())
            loss_fake = loss_func.disc_adv_loss(pred_fake, False, None)
            # Real
            real_AB = torch.cat((adv_imgs, clean_imgs), 1)
            pred_real = discriminator(real_AB)
            loss_real = loss_func.disc_adv_loss(pred_real, True, loss_fake.item())
            loss_D = cfg.Adversarial_Loss_Lambda * (loss_fake + loss_real) * 0.5
            loss_D.backward()
            disc_optimizer.step()
            writer.add_scalar('Loss/Discriminator', loss_func.loss_value_dict['disc_adv_loss'], iteration)
            report += "Disc_Loss: {:.3f} ".format(loss_func.loss_value_dict['disc_adv_loss'])
        # Update Attack Detection Pipeline
        optimizer.zero_grad()
        if cfg.Adversarial_Loss:
            set_requires_grad(discriminator, False)
            fake_AB = torch.cat((adv_imgs, torch.clamp(recon_imgs, min=0, max=1)), 1)
            dis_fake = discriminator(fake_AB)
        else:
            dis_fake = None
        
        loss = loss_func.calc_loss(recon_imgs, dis_fake, clean_imgs, residual_pred, att_labels)
        loss.backward()
        optimizer.step()

        if iteration % cfg.Log_Period == 0:
            writer.add_image('Images/Input', make_grid(adv_imgs[:16], nrow=4), iteration)
            writer.add_image('Images/Clean', make_grid(clean_imgs[:16], nrow=4), iteration)
            writer.add_image('Images/Reconstructed', make_grid(torch.clamp(recon_imgs[:16], min=0, max=1), nrow=4), iteration)
            #writer.add_image('Images/Residual', make_grid((residual_imgs[:16] + 1)/2, nrow=4), iteration)

        if cfg.Reconstruction_Loss:
            writer.add_scalar('Loss/Reconstruction', loss_func.loss_value_dict['reconstruction_loss'], iteration)
            report += "Recon_Loss: {:.3f} ".format(loss_func.loss_value_dict['reconstruction_loss'])
            if KLD is not None:
                writer.add_scalar('Loss/KLD', loss_func.loss_value_dict['kl_loss'], iteration)
                report += "KLD_Loss: {:.3f} ".format(loss_func.loss_value_dict['kl_loss'])
        
        if cfg.Attack_Det_Loss:
            writer.add_scalar('Loss/Attack_Detection', loss_func.loss_value_dict['attack_det_loss'], iteration)
            report += "Att_Det_Loss: {:.3f} ".format(loss_func.loss_value_dict['attack_det_loss'])

        if cfg.Perceptual_Loss:
            writer.add_scalar('Loss/Perceptual', loss_func.loss_value_dict['perceptual_loss'], iteration)
            report += "Percep_Loss: {:.3f} ".format(loss_func.loss_value_dict['perceptual_loss'])
        
        if cfg.Adversarial_Loss:
            writer.add_scalar('Loss/Adversarial', loss_func.loss_value_dict['gen_adv_loss'], iteration)
            report += "Adv_Loss : {:.3f} ".format(loss_func.loss_value_dict['gen_adv_loss'])

        if cfg.Image_Classification_Loss:
            writer.add_scalar('Loss/Image_Classification', loss_func.loss_value_dict['im_cls_loss'], iteration)
            report += "IMCLS_Loss : {:.3f}".format(loss_func.loss_value_dict['im_cls_loss'])
        
        writer.add_scalar('Loss/Total', loss.item(), iteration)    
    else:
        recon_imgs = None
        residual_pred = attack_classifier(recon_imgs, adv_imgs, clean_imgs)
        optimizer.zero_grad()
        loss = loss_func.calc_loss(recon_imgs, None, clean_imgs, residual_pred, att_labels)
        
        loss.backward()
        optimizer.step()

        if iteration % cfg.Log_Period == 0:
            writer.add_image('Images/Input', make_grid(adv_imgs[:16], nrow=4), iteration)
            writer.add_image('Images/Clean', make_grid(clean_imgs[:16], nrow=4), iteration)
            if cfg.Baseline_Source == 'RES':
                writer.add_image('Images/GT_Residual', make_grid((adv_imgs - clean_imgs)[:16], nrow=4), iteration)
        writer.add_scalar('Loss/Attack_Detection', loss_func.loss_value_dict['attack_det_loss'], iteration)
        report = "Attack_Det_Loss: {:.3f} ".format(loss_func.loss_value_dict['attack_det_loss'])
        writer.add_scalar('Loss/Total', loss.item(), iteration) 
    return report

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
MEAN = torch.Tensor([[[[mean[0]]],[[mean[1]]],[[mean[2]]]]]).cuda()
STD = torch.Tensor([[[[std[0]]],[[std[1]]],[[std[2]]]]]).cuda()

def eval_iter(cfg, 
              in_batch, 
              reconstructor, 
              attack_classifier, 
              loss_func,
              im_cls_results,
              att_det_results,
              att_mapper,
              confusion_meter,
              phase='train'):
    
    Classes = [torch.tensor([i]).cuda() for i in range(0, len(att_mapper))]

    with torch.no_grad():
        try:
            adv_imgs, clean_imgs, labels, att_labels = in_batch
        except:
            adv_imgs, clean_imgs, labels = in_batch
            att_labels = None

        if cfg.Device == 'cuda':
            adv_imgs, labels = adv_imgs.cuda(), labels.cuda()
            clean_imgs = clean_imgs.cuda()
            if att_labels is not None:
                att_labels = att_labels.cuda()

        if not cfg.Baseline:
            recon_imgs = reconstructor(adv_imgs)
            recon_imgs = recon_imgs.clamp_(0,1)
            attack_predictions = attack_classifier(recon_imgs, adv_imgs)
            
            if im_cls_results is not None:
                im_cls_prediction = loss_func.image_classification_loss.classifier((recon_imgs - MEAN) / STD)
                _, predicted = torch.max(im_cls_prediction.data, 1)
                
                if att_labels is not None:
                    for j, att_label in enumerate(Classes):
                        im_cls_results[j,1] += labels[att_labels == att_label].shape[0]
                        im_cls_results[j,0] += (labels[att_labels == att_label] == predicted[att_labels == att_label]).sum()
                else:
                    im_cls_results[0,1] += labels.shape[0]
                    im_cls_results[0,0] += (labels == predicted).sum()

        else:
            attack_predictions = attack_classifier(None, adv_imgs, clean_imgs)
            
        _, predicted = torch.max(attack_predictions.data, 1)
        confusion_meter.update(predicted, att_labels.squeeze())
        att_det_results[:, 1] += torch.histc(att_labels.float(),
                                bins=cfg.Num_Attack_Types+1,
                                min=0, 
                                max=cfg.Num_Attack_Types).int().cpu().numpy()
        try:
            att_det_results[:, 0] += torch.histc(att_labels[att_labels == predicted].float(),
                                bins=cfg.Num_Attack_Types+1, 
                                min=0, 
                                max=cfg.Num_Attack_Types).cpu().numpy()
        except:
            att_det_results[:, 0] += 0            
            
    return im_cls_results, att_det_results