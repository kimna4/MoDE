'''
v2 는 loss 를 net 안에서 계산
'''

#!/usr/bin/env python
# coding=utf-8
'''
Author:Tai Lei
Date:Wed Sep 19 20:30:48 2018
Info:
References: https://github.com/pytorch/examples/tree/master/imagenet
'''
import argparse
import os
import random
import time
import datetime
import math
import logging

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from tensorboardX import SummaryWriter

''' from lusr '''
from torch.nn import functional as F
from torchvision.utils import save_image
from utils.utils import ExpDataset, reparameterize

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import models.VAE_net_pair as resnet_carla

from models.VAE_net_lusr_v2 import CarlaDisentangledVAE

# from data.carla_loader_lusr import CarlaH5Data
from data.carla_loader_lusr_shuffled_data import CarlaH5Data_Simple
from utils.helper import AverageMeter, save_checkpoint

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

parser = argparse.ArgumentParser(description='Carla CIL training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--batch_size', default=1, type=int, metavar='N', help='batch size of training')
parser.add_argument('--id', default="training_v2", type=str)
# parser.add_argument('--id', default="training_v2_vqvae", type=str)
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--evaluate-log', default="", type=str, metavar='PATH', help='path to log evaluation results (default: none)')
parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')

''' from lusr '''
parser.add_argument('--beta', default=10, type=int)
parser.add_argument('--bloss-coef', default=0.5, type=int, help='back cycle loss coefficient')
parser.add_argument('--class-latent-size', default=128, type=int)
parser.add_argument('--content-latent-size', default=128, type=int)
parser.add_argument('--resnet-pretrained', default=True, action='store_false')
parser.add_argument('--cur-file-name', default='train_CycleVAE_lusr_v2.py', type=str)

def output_log(output_str, logger=None):
    """
    standard output and logging
    """
    print("[{}]: {}".format(datetime.datetime.now(), output_str))
    if logger is not None:
        logger.critical("[{}]: {}".format(datetime.datetime.now(), output_str))

def log_args(logger):
    '''
    log args
    '''
    attrs = [(p, getattr(args, p)) for p in dir(args) if not p.startswith('_')]
    for key, value in attrs:
        output_log("{}: {}".format(key, value), logger=logger)

def main():
    global args
    args = parser.parse_args()
    log_dir = os.path.join("./", "logs", args.id)
    run_dir = os.path.join("./", "runs", args.id)
    img_dir = os.path.join("./", "checkimages", args.id)
    save_weight_dir = os.path.join("./save_models", args.id)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_weight_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    logging.basicConfig(filename=os.path.join(log_dir, "carla_training.log"), level=logging.ERROR)
    tsbd = SummaryWriter(log_dir=run_dir)
    log_args(logging)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        output_log(
            'You have chosen to seed training. '
            'This will turn on the CUDNN deterministic setting, '
            'which can slow down your training considerably! '
            'You may see unexpected behavior when restarting '
            'from checkpoints.', logger=logging)

    if args.gpu is not None:
        output_log('You have chosen a specific GPU. This will completely '
                   'disable data parallelism.', logger=logging)

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=0)

    # model = resnet_carla.resnet34_carla(True, args.class_latent_size, args.content_latent_size)

    Model = CarlaDisentangledVAE
    model = Model(class_latent_size=args.class_latent_size, content_latent_size=args.content_latent_size,\
                  img_channel=3, flatten_size=18432, net_type='Basic', pretrained=args.resnet_pretrained)

    # model = Model(class_latent_size=args.class_latent_size, content_latent_size=args.content_latent_size,\
    #               img_channel=3, flatten_size=18432, net_type='vqvae', pretrained=args.resnet_pretrained)

    # model = Model(class_latent_size=args.class_latent_size, content_latent_size=args.content_latent_size, \
    #               img_channel=3, flatten_size=18432, net_type='vgg16', pretrained=args.resnet_pretrained)

    # flatten_size = 18432


    criterion = nn.MSELoss()

    # tsbd.add_graph(model, (torch.zeros(1, 3, 88, 200),torch.zeros(1, 1)))

    if args.gpu is not None:
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    # TODO check other papers optimizers
    optimizer = optim.Adam(list(model.parameters()), args.lr, betas=(0.7, 0.85))
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # optionally resume from a checkpoint
    if args.resume:
        args.resume = os.path.join(save_weight_dir, args.resume)
        if os.path.isfile(args.resume):
            output_log("=> loading checkpoint '{}'".format(args.resume), logging)
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['scheduler'])
            output_log("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']), logging)
        else:
            output_log("=> no checkpoint found at '{}'".format(args.resume),logging)

    cudnn.benchmark = True

    train_pair = ["/SSD2/datasets/carla/pair_db_sampling/gen_data_pair_W1/",
                  "/SSD2/datasets/carla/pair_db_sampling/gen_data_pair_W3/",
                  "/SSD2/datasets/carla/pair_db_sampling/gen_data_pair_W6/",
                  "/SSD2/datasets/carla/pair_db_sampling/gen_data_pair_W8/"]
    eval_pair = ["/SSD2/datasets/carla/pair_db_sampling/gen_data_pair_W1/",
                  "/SSD2/datasets/carla/pair_db_sampling/gen_data_pair_W3/",
                  "/SSD2/datasets/carla/pair_db_sampling/gen_data_pair_W6/",
                  "/SSD2/datasets/carla/pair_db_sampling/gen_data_pair_W8/"]
    carla_data = CarlaH5Data_Simple(train_pair_folders=train_pair, eval_pair_folders=eval_pair,
                            batch_size=args.batch_size, num_workers=args.workers)

    train_loader = carla_data.loaders["train"]
    eval_loader = carla_data.loaders["eval"]
    best_prec = math.inf

    if args.evaluate:
        args.id = args.id+"_test"
        if not os.path.isfile(args.resume):
            output_log("=> no checkpoint found at '{}'".format(args.resume), logging)
            return
        if args.evaluate_log == "":
            output_log("=> please set evaluate log path with --evaluate-log <log-path>")

        # TODO add test func
        evaluate(eval_loader, model, criterion, 0, tsbd)
        return

    for epoch in range(args.start_epoch, args.epochs):
        losses = train(train_loader, model, criterion, optimizer, epoch, tsbd)

        prec = evaluate(eval_loader, model, criterion, epoch, tsbd)

        lr_scheduler.step()

        # remember best prec@1 and save checkpoint
        is_best = prec < best_prec
        best_prec = min(prec, best_prec)
        save_checkpoint(
            {'epoch': epoch + 1,
             'state_dict': model.state_dict(),
             'best_prec': best_prec,
             'scheduler': lr_scheduler.state_dict(),
             'optimizer': optimizer.state_dict()},
            args.id,
            is_best,
            os.path.join(
                save_weight_dir,
                "{}_{}.pth".format(epoch+1, args.id))
            )

def train(loader, model, criterion, optimizer, epoch, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    flosses = AverageMeter()
    blosses = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    step = epoch * len(loader)
    for i, (img1, img2, img3, img4) in enumerate(loader):
        data_time.update(time.time() - end)

        # if args.gpu is not None:
        img1 = img1.squeeze(0).cuda(args.gpu, non_blocking=True)
        img2 = img2.squeeze(0).cuda(args.gpu, non_blocking=True)
        img3 = img3.squeeze(0).cuda(args.gpu, non_blocking=True)
        img4 = img4.squeeze(0).cuda(args.gpu, non_blocking=True)

        floss, bloss = model(img1, img2, img3, img4, args.beta, mode=0, latentcode=None)
        floss = floss.mean()
        bloss = bloss.mean()
        loss = floss + bloss * args.bloss_coef
        losses.update(loss.item(), img1.shape[0])
        flosses.update(floss.item(), img1.shape[0])
        blosses.update(bloss.item(), img1.shape[0])

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 or i == len(loader):
            writer.add_scalar('train/loss', losses.val, step+i)
            output_log(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'FLoss {floss.val:.4f} ({floss.avg:.4f})\t'
                'BLoss {bloss.val:.4f} ({bloss.avg:.4f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                .format(epoch, i, len(loader), batch_time=batch_time,
                    data_time=data_time, floss=flosses, bloss=blosses, loss=losses), logging)

    return losses.avg

def evaluate(loader, model, criterion, epoch, writer):
    batch_time = AverageMeter()
    losses = AverageMeter()
    flosses = AverageMeter()
    blosses = AverageMeter()

    # switch to evaluate mode
    model.eval()
    step = epoch * len(loader)
    with torch.no_grad():
        end = time.time()
        for i, (img1, img2, img3, img4) in enumerate(loader):

            img1 = img1.squeeze(0).cuda(args.gpu, non_blocking=True)
            img2 = img2.squeeze(0).cuda(args.gpu, non_blocking=True)
            img3 = img3.squeeze(0).cuda(args.gpu, non_blocking=True)
            img4 = img4.squeeze(0).cuda(args.gpu, non_blocking=True)
            sequence_num = img1.shape[0]

            floss, bloss = model(img1, img2, img3, img4, args.beta, mode=0, latentcode=None)
            floss = floss.mean()
            bloss = bloss.mean()

            img_cat = torch.cat([img1, img2, img3, img4], dim=0)

            loss = floss + bloss * args.bloss_coef
            losses.update(loss.item(), img1.shape[0])
            flosses.update(floss.item(), img1.shape[0])
            blosses.update(bloss.item(), img1.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 or i == len(loader):
                writer.add_scalar('eval/loss', losses.val, step+i)
                output_log('Test: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'FLoss {floss.val:.4f} ({floss.avg:.4f})\t'
                        'BLoss {bloss.val:.4f} ({bloss.avg:.4f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        .format(i, len(loader), batch_time=batch_time,
                                floss=flosses, bloss=blosses, loss=losses), logging)

                ''' save the eval images '''
                rand_idx = torch.randperm(img_cat.shape[0])
                imgs1 = img_cat[rand_idx[:9]]
                imgs2 = img_cat[rand_idx[-9:]]
                # imgs1 = img1[:9]
                # imgs2 = img2[:9]
                imgs1 = torch.cat([img1[1].unsqueeze(0), img1[2].unsqueeze(0), img1[3].unsqueeze(0),
                                   img2[1].unsqueeze(0), img2[2].unsqueeze(0), img2[3].unsqueeze(0),
                                   img3[1].unsqueeze(0), img3[2].unsqueeze(0), img3[3].unsqueeze(0),
                                   img4[1].unsqueeze(0), img4[2].unsqueeze(0), img4[3].unsqueeze(0),
                                   ], dim=0)
                imgs2 = torch.cat([img2[sequence_num-1].unsqueeze(0), img3[sequence_num-2].unsqueeze(0), img4[sequence_num-3].unsqueeze(0),
                                   img1[sequence_num-1].unsqueeze(0), img3[sequence_num-2].unsqueeze(0), img4[sequence_num-3].unsqueeze(0),
                                   img1[sequence_num-1].unsqueeze(0), img2[sequence_num-2].unsqueeze(0), img4[sequence_num-3].unsqueeze(0),
                                   img1[sequence_num-1].unsqueeze(0), img2[sequence_num-2].unsqueeze(0), img3[sequence_num-3].unsqueeze(0),
                                   ], dim=0)

                with torch.no_grad():
                    mu, _, classcode1 = model(imgs1, None, None, None, 0, mode=1, latentcode=None)
                    mu2, _, classcode2 = model(imgs2, None, None, None, 0, mode=1, latentcode=None)
                    recon_imgs1 = model(None, None, None, None, 0, mode=2, latentcode=torch.cat([mu, classcode1], dim=1))
                    recon_combined = model(None, None, None, None, 0, mode=2, latentcode=torch.cat([mu, classcode2], dim=1))
                saved_imgs = torch.cat([imgs1, imgs2, recon_imgs1, recon_combined], dim=0)
                save_image(saved_imgs, "./checkimages/%s/%s_%d_%d.png" % (args.id, args.id, epoch, i), nrow=12)

            if i > 100:
                break

    return losses.avg

if __name__ == '__main__':
    main()
