from posixpath import split
from .dataset import ImageDataset, DataSetContainer
from .transform import build_transforms
from torch.utils.data import DataLoader
from .collate_batch import train_collate_fn, val_collate_fn, test_collate_fn


def make_data_loader(cfg, phase='train'):
    transfrom = build_transforms(cfg)
    dataset = DataSetContainer(cfg)
    if phase == 'train':
        att_label_mapper = dataset.att_label_mapper
    elif phase == 'test':
        att_label_mapper = {}

    train_dataset = ImageDataset(dataset, transform=transfrom, phase='train')
    val_dataset = ImageDataset(dataset, transform=transfrom, phase='val')
    test_dataset = ImageDataset(dataset, transform=transfrom, phase='test')

    train_loader = DataLoader(train_dataset, batch_size=cfg.Train_Batch_Size,shuffle=True, num_workers=cfg.Num_Workers, collate_fn=train_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=cfg.Val_Batch_Size, shuffle=False, num_workers=cfg.Num_Workers, collate_fn=val_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=cfg.Test_Batch_Size, shuffle=False, num_workers=cfg.Num_Workers, collate_fn=test_collate_fn)

    return train_loader, val_loader, test_loader, att_label_mapper
    