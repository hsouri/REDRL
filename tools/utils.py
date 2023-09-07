import torch
import os.path as osp

def load_checkpoint(reconstructor, attack_classifier=None, path=''):
    ckpt = torch.load(path)
    recon_ckpt = ckpt['reconstructor_state_dict']
    #attack_ckpt = ckpt['attack_classifier_state_dict']
    ckpt_1, ckpt_2 = {}, {}
    for key in recon_ckpt:
        ckpt_1['module.'+ key] = recon_ckpt[key]
    #for key in attack_ckpt:
    #    ckpt_2['module.'+ key] = attack_ckpt[key]
    
    reconstructor.load_state_dict(ckpt_1)
    print('Reconstructor Weights loaded!')
    #attack_classifier.load_state_dict(ckpt_2)
    #print('Attack Classifier Weights loaded!')
    return reconstructor, attack_classifier