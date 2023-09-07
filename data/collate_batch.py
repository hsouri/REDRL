import torch

from torch._C import dtype


def train_collate_fn(batch):
    clean_imgs, adv_imgs, labels, att_labels = zip(*batch)
    labels = torch.tensor(labels, dtype=torch.int64)
    att_labels = torch.tensor(att_labels, dtype=torch.int64)
    return torch.stack(adv_imgs, dim=0), torch.stack(clean_imgs, dim=0), labels, att_labels

def val_collate_fn(batch):
    clean_imgs, adv_imgs, labels, att_labels = zip(*batch)
    labels = torch.tensor(labels, dtype=torch.int64)
    att_labels = torch.tensor(att_labels, dtype=torch.int64)
    return torch.stack(adv_imgs, dim=0), torch.stack(clean_imgs, dim=0), labels, att_labels

def test_collate_fn(batch):
    clean_imgs, adv_imgs, labels = zip(*batch)
    labels = torch.tensor(labels, dtype=torch.int64)
    return torch.stack(adv_imgs, dim=0), torch.stack(clean_imgs, dim=0), labels

"""
def train_collate_fn(batch):
    imgs, gt_imgs, cls_labels, attack_labels, img_paths = zip(*batch)
    cls_labels = torch.tensor(cls_labels, dtype=torch.int64)
    attack_labels = torch.tensor(attack_labels, dtype=torch.int64)
    return torch.stack(imgs, dim=0), torch.stack(gt_imgs, dim=0), cls_labels, attack_labels


def val_collate_fn(batch):
    imgs, gt_imgs, cls_labels, attack_labels, img_paths = zip(*batch)
    cls_labels = torch.tensor(cls_labels, dtype=torch.int64)
    attack_labels = torch.tensor(attack_labels, dtype=torch.int64)
    return torch.stack(imgs, dim=0), torch.stack(gt_imgs, dim=0), cls_labels, attack_labels, img_paths
"""