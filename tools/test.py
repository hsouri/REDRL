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
from engine.evaluator import do_eval
import shutil
import logging
from tools.utils import load_checkpoint

def test(cfg):
    _, _, test_loader, att_mapper = make_data_loader(cfg, phase='test')
    reconstructor, attack_classifier = make_general_models(cfg)
    reconstructor, attack_classifier = load_checkpoint(reconstructor, attack_classifier, path=cfg.TEST_CKPT_PATH)
    
    loss_func = LossCalculator(cfg)

    #logger = logging.getLogger('Adversarial_Attack_Removal.test')
    logger = None

    _ = do_eval(cfg, reconstructor, attack_classifier, test_loader, att_mapper, loss_func, logger=logger, epoch=0)



def main():
    parser = argparse.ArgumentParser(description='Adversarial Attack Detection')
    parser.add_argument('opts', help='Modify config options using command-line', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg = Configs()
    cfg.apply_cmdline_cfgs(args)

    if cfg.do_save:
        output_dir = cfg.Save_Path
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    
    #logger = setup_logger("Adversarial_Attack_Removal", output_dir, 0)
    print('Running with the following Configs:\n')
    print('-' * 50 + '\n')
    for key in cfg.__dict__.keys():
        print('{} : {}'.format(key, cfg.__dict__[key]))
    
    test(cfg)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()