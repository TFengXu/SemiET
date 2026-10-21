# -*- coding: utf-8 -*-
import isecret.utils.distributed as du
import isecret.utils.loggging as logging
from isecret.model.build import build_model
from isecret.dataset.build import build_loader
from isecret.model.test_model import TestModel
from isecret.utils.sender import EMailSender
import numpy as np
import random
import torch
import time



logger = logging.get_logger(__name__)


def test_func(args):
    # Initialize distributed training
    du.init_distributed_training(args)


    # Set torch flag
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Log training setting
    logging.setup_logging(args.experiment.log_dir)
    logger.info("Test with config:\n")
    logger.info(args)

    # set train step 为1 推理时 无需多次
    args.train.step = 1
    # Build data loader
    if args.overall.test_dir == '':
        train_dataloader, val_dataloader, test_dataloader = build_loader(args)
        test_item = val_dataloader
        print('test valdataloader')
        test_inform = 'testset'
    else:
        test_item = args.overall.test_dir
        test_inform = test_item

    # evaluate
    model = TestModel(args)
    metric = model.inference(test_item, output_dir=args.overall.output_dir)

    if du.is_master_proc():
        print(metric)
    