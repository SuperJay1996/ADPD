from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil
import random
import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss, JointsDPDLoss
from core.function import train, validate, fpd_train, adpd_train
from utils.utils import get_optimizer, save_checkpoint, load_checkpoint, create_logger, get_model_summary, save_yaml_file

import datasets
import models

def parse_args():
    parser = argparse.ArgumentParser(description="Train Human Pose Distillation Network.")
    parser.add_argument(
        '--cfg',
        help='student experiment configure file name',
        required=True,
        type=str
    )
    parser.add_argument(
        '--tcfg',
        help='teacher experiment configure file name',
        required=False,
        default=None,
        type=str
    )
    parser.add_argument('opts',
                        help='modify config options using the command-line',
                        default=None,
                        nargs=argparse.REMAINDER)
    
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()

    return args

def setup_seed(seed):
    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed=seed)
    random.seed(seed)

def get_train_type(train_type, checkpoint):
    exist_status = checkpoint and os.path.exists(checkpoint)
    if train_type == 'NORMAL': # NORMAL train, just return
        return train_type
    elif not exist_status:
        exit('ERROR: teacher checkpoint is not existed.')
    elif train_type == "FPD" or train_type =="DPD":
        return train_type
    else:
        exit('ERROR: please select train type {} in [NORMAL, FPD, DPD].'.format(train_type))

def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'train'
    )

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    t_checkpoints = cfg.KD.TEACHER
    train_type = cfg.KD.TRAIN_TYPE
    train_type = get_train_type(train_type, t_checkpoints)
    logger.info('=> Train type is {}'.format(train_type))

    if train_type == 'FPD' or 'DPD':
        cfg_name = 'student_' + os.path.basename(args.cfg).split('.')[0]
    else:
        cfg_name = os.path.basename(args.cfg).split('.')[0]

    save_yaml_file(cfg_name, cfg, final_output_dir)

    setup_seed(cfg.SEED)
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=True
    )

    if train_type != 'NORMAL':
        tcfg = cfg.clone()
        tcfg.defrost()
        tcfg.merge_from_file(args.tcfg)
        tcfg.freeze()
        tcfg_name = 'teacher_' + os.path.basename(args.tcfg).split('.')[0]
        save_yaml_file(tcfg_name, tcfg, final_output_dir)
        # teacher model
        tmodel = eval('models.'+tcfg.MODEL.NAME+'.get_pose_net')(
            tcfg, is_train=False
        )

        load_checkpoint(t_checkpoints, tmodel,
                        strict=True,
                        model_info='teacher_'+tcfg.MODEL.NAME)

        tmodel = torch.nn.DataParallel(tmodel, device_ids=cfg.GPUS).cuda()
    if train_type == 'FPD':
        kd_pose_criterion = JointsMSELoss(
            use_target_weight=tcfg.LOSS.USE_TARGET_WEIGHT
        ).cuda()
    elif train_type == 'ADPD':
        kd_pose_criterion = JointsDPDLoss(
            use_target_weight=tcfg.KD.LOSS_PARAM.USE_TARGET_WEIGHT, 
            loss_weight=tcfg.KD.LOSS_PARAM.LOSS_WEIGHT,
            thre=cfg.KD.LOSS_PARAM.THRE, 
            temperature=cfg.KD.LOSS_PARAM.TEMPERATURE
        ).cuda()
    
    this_dir = os.path.dirname(__file__)
    shutil.copy2(os.path.join(this_dir, '../lib/models', cfg.MODEL.NAME + '.py'), final_output_dir)

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_step': 0,
    }

    dump_input = torch.rand(
        (1, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0])
    )
    writer_dict['writer'].add_graph(model, (dump_input, ))

    logger.info(get_model_summary(model, dump_input))

    if cfg.TRAIN.CHECKPOINT:
        load_checkpoint(cfg.TRAIN.CHECKPOINT, model, strict=False,
                        model_info='student_' + cfg.MODEL.NAME)
    
    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    pose_criterion = JointsMSELoss(
        use_target_weight = cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    if cfg.KD.LOSS_TYPE == 'ADPD':
        weight = eval('models.'+cfg.WEIGHT.NAME+'.get_weight_net')(cfg)
        weight = torch.nn.DataParallel(weight, device_ids=cfg.GPUS).cuda()

    best_perf = 0.0
    best_model = False
    last_epoch = -1
    optimizer = get_optimizer(cfg, model, weight=weight if cfg.KD.LOSS_TYPE == 'ADPD' else None)
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint.pth'
    )
    
    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer'])

        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))
    
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
        last_epoch=last_epoch
    )

    # evaluate on validation set for teacher
    validate(
        tcfg, valid_loader, valid_dataset, tmodel, pose_criterion,
        final_output_dir, tb_log_dir, writer_dict
    )
    # evaluate on validation set for vanilla student
    validate(
        cfg, valid_loader, valid_dataset, model, pose_criterion,
        final_output_dir, tb_log_dir, writer_dict
    )

    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        
        # fpd method, default NORMAL
        if train_type == 'FPD':
            # train for one epoch
            fpd_train(cfg, train_loader, model, tmodel,
                      pose_criterion, kd_pose_criterion, optimizer, epoch,
                      final_output_dir, tb_log_dir, writer_dict)
        elif train_type == 'DPD':
            adpd_train(cfg, train_loader, model, tmodel, weight, 
                       pose_criterion, kd_pose_criterion, optimizer, epoch,
                       final_output_dir, tb_log_dir, writer_dict)
        else:
            # train for one epoch
            train(cfg, train_loader, model, pose_criterion, optimizer, epoch,
                  final_output_dir, tb_log_dir, writer_dict)

        lr_scheduler.step()
        # evaluate on validation set
        perf_indicator = validate(
            cfg, valid_loader, valid_dataset, model, pose_criterion,
            final_output_dir, tb_log_dir, writer_dict
        )

        if perf_indicator >= best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        if cfg.KD.LOSS_TYPE == 'ADPD':
            save_checkpoint({
                'epoch': epoch + 1,
                'model': cfg.MODEL.NAME,
                'state_dict': model.state_dict(),
                'best_state_dict': model.module.state_dict(),
                'perf': perf_indicator,
                'optimizer': optimizer.state_dict(),
                'optw': optw.state_dict(),
                'weight': weight.state_dict()
            }, best_model, final_output_dir)
        else:
            save_checkpoint({
                'epoch': epoch + 1,
                'model': cfg.MODEL.NAME,
                'state_dict': model.state_dict(),
                'best_state_dict': model.module.state_dict(),
                'perf': perf_indicator,
                'optimizer': optimizer.state_dict(),
            }, best_model, final_output_dir)

    final_model_state_file = os.path.join(
        final_output_dir, 'final_state.pth'
    )
    logger.info('=> saving final model state to {}'.format(
        final_model_state_file)
    )
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
