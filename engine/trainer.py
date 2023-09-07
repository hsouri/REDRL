import logging
import torch
from torch import nn
import os.path as osp
import numpy as np
import sys
from .tools import train_iter, eval_iter
from torch.utils.tensorboard import SummaryWriter
from .evaluator import do_eval
from time import time
import pdb

def do_train(cfg,
        reconstructor,
        attack_classifier, 
        train_loader, 
        val_loader, 
        optimizer, 
        scheduler, 
        loss_func, 
        start_epoch,
        att_mapper,
        discriminator=None, disc_optimizer=None, disc_scheduler=None):
    
    log_period = cfg.Log_Period
    checkpoint_period = cfg.Checkpoint_Period
    eval_period = cfg.Eval_Period
    output_dir = cfg.Output_Path
    epochs = cfg.Num_Epochs
    writer = SummaryWriter(osp.join(output_dir, 'TensorBoard'))
    best_imcls_acc = 0.0
    best_att_det_acc = 0.0
    logger = logging.getLogger('Adversarial Attack Removal.train')
    logger.info('Start training')

    iterations = 0
    start = time()
    logger.info('Initial Evaluation')
    _, _ = do_eval(cfg, reconstructor, attack_classifier, val_loader, att_mapper, loss_func, writer, logger, 0)
    for epoch in range(start_epoch, epochs):    
        scheduler.step()
        for i, in_batch in enumerate(train_loader):
            report = train_iter(cfg, iterations, in_batch, reconstructor, attack_classifier, optimizer, 
            loss_func, discriminator, disc_optimizer, disc_scheduler, writer)
            if i % log_period == 0:
                logger.info("Epoch[{0}] Iter[{1}/{2}] LR [{3:.2f}] ".format(epoch + 1, i + 1, len(train_loader), scheduler.get_lr()[0] * 1e5) + report)
            iterations += 1
        if epoch % eval_period == 0:
            im_cls_acc, att_det_acc = do_eval(cfg, reconstructor, attack_classifier, val_loader, att_mapper, loss_func, writer, logger, epoch + 1)
            if not im_cls_acc is None:
                if im_cls_acc > best_imcls_acc:
                    best_imcls_acc = im_cls_acc
            if att_det_acc > best_att_det_acc:
                best_att_det_acc = att_det_acc
        if epoch % checkpoint_period == 0:
            torch.save({'reconstructor_state_dict': reconstructor.module.state_dict(),
                        'attack_classifier_state_dict': attack_classifier.module.state_dict()},
                            osp.join(output_dir, 'Epoch_{}_ckpts.pth.tar'.format(epoch + 1)))
    end = time()
    h = (end - start) // 3600
    m = ((end - start) % 3600) // 60
    logger.info('Finished training in {} hours & {} minutes'.format(h, m))
    if not im_cls_acc is None:
        logger.info('Best Image Accuracy Obtained: {0:.2f}%'.format(best_imcls_acc))
    logger.info('Best Attack Detection Accuracy Obtained: {0:.2f}%'.format(best_att_det_acc))
    return



