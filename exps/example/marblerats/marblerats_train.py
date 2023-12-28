#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import Exp as MyExp
from torch.utils.data import Dataset, DataLoader

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 1.33
        self.width = 1.25
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]


    def get_data_loader(self, batch_size, is_distributed, no_aug=False):

        dataset = Dataset(
            data_dir=os.path.join("/content/OC_SORT/datasets/", "marblerats"),
            json_file="/content/OC_SORT/datasets/marblerats/annotations/train.json",
            name='train',
            img_size=self.input_size
        )

        self.dataset = dataset

        train_loader = DataLoader(self.dataset)

        return train_loader

    def get_eval_loader(self, batch_size, is_distributed, testdev=False):

        valdataset = Dataset(
            data_dir=os.path.join("/content/OC_SORT/datasets/", "marblerats"),
            json_file="/content/OC_SORT/datasets/marblerats/annotations/val.json",
            img_size=self.test_size,
            name='val'
        )

        val_loader = DataLoader(valdataset)

        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False):
        from yolox.evaluators import COCOEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev=testdev)
        evaluator = COCOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )
        return evaluator
