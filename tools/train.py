import argparse
import os
import sys
import torch
import warnings
import sys
sys.path.append('.')
from logger import setup_logger
from config.cfg import Configs
from data import make_data_loader
from model import make_discriminator, make_general_models
from solver import make_optimizer, lr_scheduler
from layers import LossCalculator
from engine.trainer import do_train
import pdb
import shutil


def train(cfg):
    train_loader, val_loader, _, att_mapper = make_data_loader(cfg)
    reconstructor, attack_classifier = make_general_models(cfg)
    start_epoch = 0
    optimizer = make_optimizer(cfg, reconstructor, attack_classifier)
    scheduler = lr_scheduler.WarmupMultiStepLR(optimizer, cfg.Steps, cfg.Gamma, cfg.Warmup_Factor,
                        cfg.Warmup_Iters, cfg.Warmup_Method)
    if cfg.Adversarial_Loss:
        discriminator = make_discriminator(cfg)
        disc_optimizer = make_optimizer(cfg, discriminator)
        disc_scheduler = lr_scheduler.WarmupMultiStepLR(disc_optimizer, cfg.Steps, cfg.Gamma, cfg.Warmup_Factor,
                        cfg.Warmup_Iters, cfg.Warmup_Method)
    else:
        discriminator, disc_optimizer, disc_scheduler = None, None, None
        
    loss_func = LossCalculator(cfg)
    
    do_train(cfg,
        reconstructor,
        attack_classifier, 
        train_loader, 
        val_loader, 
        optimizer, 
        scheduler, 
        loss_func, 
        start_epoch,
        att_mapper,
        discriminator, disc_optimizer, disc_scheduler)
    

def main():
    parser = argparse.ArgumentParser(description='Adversarial Attack Remover')
    parser.add_argument('opts', help='Modify config options using command-line', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg = Configs()
    cfg.apply_cmdline_cfgs(args)

    output_dir = cfg.Output_Path
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    logger = setup_logger("Adversarial Attack Removal", output_dir, 0)
    logger.info('Running with the following Configs:\n')
    logger.info('-' * 50)
    for key in cfg.__dict__.keys():
        logger.info('{} : {}'.format(key, cfg.__dict__[key]))
    
    cfg.save_cfg2json()

    train(cfg)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()