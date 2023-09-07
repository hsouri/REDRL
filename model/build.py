from .classifier import AttackClassifier
from .discriminator import Discriminator
from . import generator
from .pix2pixmodels import define_G, define_D
from . import nrp
import torch
from torch import nn
import pdb

def make_general_models(cfg):
    attack_classifier = AttackClassifier(cfg)
    if 'Unet' in cfg.Generator_Type:
        reconstructor = define_G(cfg)
    elif 'NRP' in cfg.Generator_Type:
        reconstructor = getattr(nrp, cfg.Generator_Type)(cfg)
    else:
        reconstructor = getattr(generator, cfg.Generator_Type)(cfg)

    if torch.cuda.device_count() > 1 and len(cfg.GPU_IDs) > 1:
        reconstructor = nn.DataParallel(reconstructor)
        attack_classifier = nn.DataParallel(attack_classifier)
    return reconstructor, attack_classifier

def make_discriminator(cfg):
    if not 'PatchGan' == cfg.Discriminator_Type:
        discriminator = Discriminator(cfg)
    else:
        discriminator = define_D(cfg)
    if torch.cuda.device_count() > 1 and len(cfg.GPU_IDs) > 1:
        discriminator = nn.DataParallel(discriminator)
    return discriminator
