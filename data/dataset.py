import torch
from torch.utils.data import Dataset, dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToPILImage
from tools.path import Paths
import os.path as osp
import os
from PIL import Image
import numpy as np
from random import sample
from tqdm import tqdm
import pdb


class DataSetContainer():
    def __init__(self, cfg):
        self.dataset = cfg.Dataset
        self.data_path = Paths.DataPath
        self.eps = cfg.TEST_EPS
        self.norm = cfg.TEST_NORM
        self.test_adv_data = cfg.TEST_ADV_DATA
        self._check_before_run()
        self.att_label_mapper = {}
        self.att_label_mapper['Clean'] = 0
        for i, att in enumerate([att for att in os.listdir(osp.join(self.data_path, self.dataset, 'Adversarial_RED')) if not 'DS_Store' in att]):
            self.att_label_mapper[att] = i + 1
        self.train_set = self._process_dir(split='train')
        self.val_set = self._process_dir(split='val')
        self.test_set = self._process_dir(split='test')
        self.class_to_idx = ImageFolder(osp.join(self.data_path, self.dataset, 'Clean', 'train')).class_to_idx

    def _check_before_run(self):
        if not osp.exists(osp.join(self.data_path, self.dataset)):
            raise RuntimeError("{} is not available".format(osp.join(self.data_path, self.dataset)))
        if not osp.exists(osp.join(self.data_path, self.dataset, 'Clean')):
            raise RuntimeError("{} is not available".format(osp.join(self.data_path, self.dataset, 'Clean')))
        if not osp.exists(osp.join(self.data_path, self.dataset, 'Adversarial_RED')):
            raise RuntimeError("{} is not available".format(osp.join(self.data_path, self.dataset, 'Adversarial_RED')))
    
    def _process_dir(self, split='train'):
        
        dataset = []
        for root, _, files in os.walk(osp.join(self.data_path, self.dataset, 'Clean', split)):
            for name in files:
                if not 'DS_Store' in name:
                    clean_path = osp.join(root, name)
                    dtype = name.split('.')[-1]
                    if split == 'val':
                        dataset.append((clean_path, clean_path, 0))                
                        for att_type, att_label in self.att_label_mapper.items():
                            if att_type == 'Clean':
                                continue
                            models = [model for model in os.listdir(osp.join(self.data_path, self.dataset, 'Adversarial_RED', att_type))]
                            for model in models:
                                configs = [config for config in os.listdir(osp.join(self.data_path, self.dataset, 'Adversarial_RED', att_type, model))]
                                for config in configs:
                                    adv_path = clean_path.replace('Clean', 'Adversarial_RED/{}/{}/{}'.format(att_type, model, config)).replace(dtype, 'tar')
                                    dataset.append((clean_path, adv_path, att_label))
                    
                    elif split == 'train':
                        adv_candidates = [{'Clean': [(clean_path, clean_path, 0)]}]
                        for att_type, att_label in self.att_label_mapper.items():
                            if att_type == 'Clean':
                                continue
                            tmp_dict = {att_type: []}
                            models = [model for model in os.listdir(osp.join(self.data_path, self.dataset, 'Adversarial_RED', att_type))]
                            for model in models:
                                configs = [config for config in os.listdir(osp.join(self.data_path, self.dataset, 'Adversarial_RED', att_type, model))]
                                for config in configs:
                                    adv_path = clean_path.replace('Clean', 'Adversarial_RED/{}/{}/{}'.format(att_type, model, config)).replace(dtype, 'tar')
                                    tmp_dict[att_type].append((clean_path, adv_path, att_label))
                            adv_candidates.append(tmp_dict)
                        dataset.append(adv_candidates)
                    else:
                        #att_type = '{}_eps{}_steps100'.format(self.norm, float(self.eps))
                        class_name = clean_path.split('/')[-2]
                        adv_path = osp.join(self.test_adv_data, class_name, name).replace(dtype, 'tar')
                        dataset.append((clean_path, adv_path, None))
        return dataset


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None, phase='train', test_attack='SSP', eps=16, norm='linf'):
        self.dataset = dataset
        self.data_path = osp.join(self.dataset.data_path, self.dataset.dataset)
        self.phase = phase
        self.transform = transform
        self.source = getattr(self.dataset, '{}_set'.format(self.phase))
        self.ToPILImage = ToPILImage()
        self.test_attack = test_attack
        

    def __len__(self):
        return len(self.source)
    
    def __getitem__(self, index):
        if self.phase != 'train':
            clean_path, adv_path, attack_label = self.source[index]
        else:
            adv_dict = sample(self.source[index], 1)[0]
            clean_path, adv_path, attack_label = sample(adv_dict[list(adv_dict.keys())[0]], 1)[0]

        class_label = clean_path.split('/')[-2]
        label = self.dataset.class_to_idx[class_label]
        
        clean_img = read_image(clean_path)
        adv_img = torch.load(adv_path) if adv_path.endswith('tar') else read_image(adv_path)
        if isinstance(adv_img, torch.Tensor):
            adv_img = self.ToPILImage(adv_img)

        if self.transform:
            clean_img = self.transform(clean_img)
            adv_img = self.transform(adv_img)
        if attack_label is not None:
            return clean_img, adv_img, int(label), int(attack_label)
        else:
            return clean_img, adv_img, int(label)


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img