from re import S
import torch
from torch import nn
import os.path as osp
import numpy as np
from .tools import eval_iter
from tqdm import tqdm
import sys
import pdb
from tools.confusion_meter import ConfusionMeter
from PIL import Image
from torchvision.transforms import ToTensor


def do_eval(cfg,
            reconstructor,
            attack_classifier,
            eval_loader,
            att_mapper,
            loss_func,
            writer=None, 
            logger=None,
            epoch=None):
    
    if logger is not None:
        logger.info('Start evaluation')
    else:
        print('Start evaluation')
    
    reconstructor.eval()
    if attack_classifier is not None:
        attack_classifier.eval()
    
    im_cls_results = np.zeros((len(att_mapper),2))
    if cfg.Baseline:
        im_cls_results = None
    att_det_results = np.zeros((cfg.Num_Attack_Types + 1, 2))
    
    
    if writer is not None:
        confusion_meter = ConfusionMeter(normalize=True, save_path=osp.join(cfg.Output_Path, 'ATT_Detection_CM_{}.jpg'.format(epoch)), labels=list(att_mapper.keys()))
    else:
        confusion_meter = ConfusionMeter(normalize=True, save_path=osp.join(cfg.Save_Path, 'ATT_Detection_CM_{}.jpg'.format(epoch)), labels=list(att_mapper.keys()))
    
    with torch.no_grad():   
        with tqdm(total=len(eval_loader), ncols=0, file=sys.stdout, desc='Evaluating...') as pbar:
            for i, in_batch in enumerate(eval_loader):
                im_cls_results, att_det_results = eval_iter(cfg, 
                                            in_batch, 
                                            reconstructor, 
                                            attack_classifier, 
                                            loss_func,
                                            im_cls_results,
                                            att_det_results,
                                            att_mapper,
                                            confusion_meter)
                pbar.update()
    
    cm = Image.open(confusion_meter.save_confusion_matrix())
    if writer:
        T = ToTensor()
        writer.add_image('Images/Confusion_Matrix', T(cm), epoch)
    
    if logger:
        logger.info("Ealuation Results:")
    else:
        print(("Ealuation Results:"))
    
    logger.info('---Attack Recognition Accuracy:----')
    correct, total = att_det_results.sum(axis=0)
    Classes = list(att_mapper.keys())
    for i in range(att_det_results.shape[0]):
        logger.info('Class: {0}, Accuracy: {1:.2f}%'.format(Classes[i], att_det_results[i, 0] / att_det_results[i, 1] * 100))
    att_det_acc = correct / total * 100
    logger.info('Total Accuracy: {0:.2f}%'.format(att_det_acc))
    
    im_cls_acc = None
    if im_cls_results is not None:
        if logger is not None:
            logger.info('---Reconstruction Image Classification Accuracy:----')
        for item, j in att_mapper.items():
            im_cls_acc = im_cls_results[j,0] / im_cls_results[j,1] * 100
            if logger:
                logger.info('Total Image Classification Accuracy on {0}: {1:.2f}%'.format(item, im_cls_acc))
            else:
                print('Total Image Classification Accuracy on {0}: {1:.2f}%'.format(item, im_cls_acc))
        im_cls_acc = im_cls_results[:,0].sum() / im_cls_results[:,1].sum() * 100
        if logger:
            logger.info('Total Image Classification Accuracy: {0:.2f}%'.format(im_cls_acc))
        else:
            print('Total Image Classification Accuracy: {0:.2f}%'.format(im_cls_acc))
    
    
    if writer:
        if im_cls_results is not None:
            writer.add_scalar('Accuracy/IM_Classification_Acc', im_cls_acc, epoch)
        if att_det_results is not None:
            writer.add_scalar('Accuracy/Att_Recognition_Acc', att_det_acc, epoch)

    reconstructor.train()
    if attack_classifier is not None:
        attack_classifier.train()
    
    return im_cls_acc, att_det_acc