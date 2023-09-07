import copy
import random
import torch
from collections import defaultdict

import numpy as np
from torch.utils.data.sampler import Sampler
import pdb


class TripletRandomIdentitySampler(Sampler):
    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_cls_label_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, (_, _, cls_label) in enumerate(self.data_source):
            self.index_dic[cls_label].append(index)
        self.cls_labels = list(self.index_dic.keys())

        self.length = 0
        for cls_label in self.cls_labels:
            idxs = self.index_dic[cls_label]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances
    def __iter__(self):
        batch_idx_dict = defaultdict(list)

        for cls_label in self.cls_labels:
            idxs = copy.deepcopy(self.index_dic[cls_label])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idx_dict[cls_label].append(batch_idxs)
                    batch_idxs =[]
        
        avail_cls_labels = copy.deepcopy(self.cls_labels)
        final_idxs = []

        while len(avail_cls_labels) >= self.num_cls_label_per_batch:
            selected_cls_labels = random.sample(avail_cls_labels, self.num_cls_label_per_batch)
            for cls_label in selected_cls_labels:
                batch_idxs = batch_idx_dict[cls_label].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idx_dict[cls_label]) == 0:
                    avail_cls_labels.remove(cls_label)
        
        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length

