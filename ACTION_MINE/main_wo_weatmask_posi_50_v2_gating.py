'''
resnet 50을 이용해 학습하는 버전
v2는 MI minimization도 추가한 버전
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
from VAE_net_lusr import CarlaDisentangledVAE

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
# import moe_net as resnet_carla
from mine_net_posi50_v2_gating import CarlaDisentangled
from carla_loader_pre_shuffled_data_posi import CarlaH5Data_Simple
from helper import AverageMeter, save_checkpoint
import torch.nn.functional as F
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "4, 5, 6, 7"

parser = argparse.ArgumentParser(description='Carla CIL training')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--batch_size', default=1, type=int, metavar='N',
                    help='batch size of training')
parser.add_argument('--speed_weight', default=1, type=float, help='speed weight')
parser.add_argument('--branch-weight', default=1, type=float, help='branch weight')
parser.add_argument('--id', default="training_wo_weatmask_posi50_v2_gating", type=str)
parser.add_argument('--train-dir', default="/SSD1/datasets/carla/collected_datas_210621/gen_data_simple/",
                    type=str, metavar='PATH', help='training dataset')
parser.add_argument('--eval-dir', default="/SSD1/datasets/carla/collected_datas_210621/gen_data_simple_val/",
                    type=str, metavar='PATH', help='evaluation dataset')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning-rate', default=2e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--evaluate-log', default="", type=str, metavar='PATH',
                    help='path to log evaluation results (default: none)')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--class-latent-size', default=128, type=int)
parser.add_argument('--content-latent-size', default=128, type=int)
parser.add_argument('--flatten-size', default=1024, type=int)
parser.add_argument('--vae-model-dir',
                    default="48_training_v2.pth",
                    type=str, metavar='PATH')

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
    save_weight_dir = os.path.join("./save_models", args.id)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_weight_dir, exist_ok=True)

    logging.basicConfig(filename=os.path.join(log_dir, "carla_training.log"),
                        level=logging.ERROR)
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

    # model = resnet_carla.resnet34_carla(True)
    model = CarlaDisentangled(encoder_name='Basic', class_latent_size=args.class_latent_size,
                              content_latent_size=args.content_latent_size)
    model_VAE = CarlaDisentangledVAE(class_latent_size=args.class_latent_size,
                                     content_latent_size=args.content_latent_size)
    criterion = [nn.MSELoss(), nn.CrossEntropyLoss(), nn.BCELoss()]

    if args.gpu is not None:
        model = model.cuda(args.gpu)
        model_VAE = model_VAE.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()
        model_VAE = torch.nn.DataParallel(model_VAE).cuda()

    ''' model_VAE load '''
    if not os.path.exists(args.vae_model_dir):
        raise RuntimeError('failed to find the models path: %s' % args.vae_model_dir)
    vae_model_path = args.vae_model_dir
    # pretrained_state_dict = torch.load(vae_model_path, map_location='cuda:0')
    pretrained_state_dict = torch.load(vae_model_path)
    now_state_dict = model_VAE.state_dict()
    # 1. filter out unnecessary keys
    pretrained_state_dict = {k: v for k, v in pretrained_state_dict['state_dict'].items() if k in now_state_dict}
    # 2. overwrite entries in the existing state dict
    now_state_dict.update(pretrained_state_dict)
    # 3. load the new state dict
    model_VAE.load_state_dict(now_state_dict)

    # TODO check other papers optimizers
    optimizer = optim.Adam(model.parameters(), args.lr, betas=(0.7, 0.85))
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    # optionally resume from a checkpoint
    if args.resume:
        args.resume = os.path.join(save_weight_dir, args.resume)
        if os.path.isfile(args.resume):
            output_log("=> loading checkpoint '{}'".format(args.resume),
                       logging)
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['scheduler'])
            output_log("=> loaded checkpoint '{}' (epoch {})"
                       .format(args.resume, checkpoint['epoch']), logging)
        else:
            output_log("=> no checkpoint found at '{}'".format(args.resume),
                       logging)

    cudnn.benchmark = True
    carla_data = CarlaH5Data_Simple(train_folder=args.train_dir, eval_folder=args.eval_dir,
        batch_size=args.batch_size, num_workers=args.workers)

    train_loader = carla_data.loaders["train"]
    eval_loader = carla_data.loaders["eval"]
    best_prec = math.inf

    if args.evaluate:
        args.id = args.id+"_test"
        if not os.path.isfile(args.resume):
            output_log("=> no checkpoint found at '{}'"
                       .format(args.resume), logging)
            return
        if args.evaluate_log == "":
            output_log("=> please set evaluate log path with --evaluate-log <log-path>")

        # TODO add test func
        evaluate(eval_loader, model, model_VAE, criterion, 0, tsbd)
        return

    for epoch in range(args.start_epoch, args.epochs):

        losses = train(train_loader, model, model_VAE, criterion, optimizer, epoch, tsbd)

        prec = evaluate(eval_loader, model, model_VAE, criterion, epoch, tsbd)

        lr_scheduler.step()

        # remember best prec@1 and save checkpoint
        saved_weight_name = "{}_{}.pth".format(epoch+1, args.id)
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
                saved_weight_name)
            )

def weighted_mse_loss(input, target, weight):
    return torch.sum(weight * (input - target) ** 2)

def train(loader, model, model_VAE, criterion, optimizer, epoch, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    branch_losses = AverageMeter()
    weat_losses = AverageMeter()
    # adv_weat_losses = AverageMeter()
    speed_losses = AverageMeter()

    mi_real_losses = AverageMeter()
    mi_fake_losses = AverageMeter()
    mi_max_losses = AverageMeter()
    posi_losses = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    step = epoch * len(loader)
    for i, (img, speed, target, mask, weather, posi_list) in enumerate(loader):
        data_time.update(time.time() - end)

        # if args.gpu is not None:
        img = img.squeeze(0).cuda(args.gpu, non_blocking=True)
        speed = speed.squeeze(0).cuda(args.gpu, non_blocking=True)
        target = target.squeeze(0).cuda(args.gpu, non_blocking=True)
        mask = mask.squeeze(0).cuda(args.gpu, non_blocking=True)
        weather = weather.squeeze(0).cuda(args.gpu, non_blocking=True)
        speed = speed.view(-1, 1)
        posi = posi_list.squeeze(0).cuda(args.gpu, non_blocking=True)
        sequence_num = img.shape[0]

        with torch.no_grad():
            mu, logsigma, classcode = model_VAE(img)

        weat_out, pred_speed, pred_posi, act_output1, act_output2, act_output3, act_output4, \
                mi_real, mi_fake, Ej_l, Em_l, Ej_g, Em_g, term_a, term_b \
                             = model(img, speed, mu, logsigma, classcode)

        ''' weather '''
        weat_loss = criterion[1](weat_out, weather)

        ''' actions '''
        weat_out_prob = F.softmax(weat_out, dim=1)
        mask_out1 = torch.mul(act_output1 * mask, weat_out_prob[:, 0].view(-1, 1))
        mask_out2 = torch.mul(act_output2 * mask, weat_out_prob[:, 1].view(-1, 1))
        mask_out3 = torch.mul(act_output3 * mask, weat_out_prob[:, 2].view(-1, 1))
        mask_out4 = torch.mul(act_output4 * mask, weat_out_prob[:, 3].view(-1, 1))
        mask_out = mask_out1 + mask_out2 + mask_out3 + mask_out4

        branch_loss = criterion[0](mask_out, target) * 4
        speed_loss = criterion[0](pred_speed, speed)
        posi_loss = criterion[0](pred_posi, posi)

        ''' MI loss '''
        y_real = torch.ones(sequence_num).cuda(args.gpu, non_blocking=True)
        y_fake = torch.zeros(sequence_num).cuda(args.gpu, non_blocking=True)
        mi_real_loss = criterion[2](mi_real.squeeze(), y_real) * 0.1
        mi_fake_loss = criterion[2](mi_fake.squeeze(), y_fake) * 0.1

        alpha = 0.2
        beta = 0.2
        gamma = 0.1
        LOCAL = (Em_l.mean() - Ej_l.mean()) * beta
        GLOBAL = (Em_g.mean() - Ej_g.mean()) * alpha
        PRIOR = - (term_a.mean() + term_b.mean()) * gamma
        mi_max_loss = LOCAL + GLOBAL + PRIOR
        mi_loss = (mi_real_loss + mi_fake_loss + mi_max_loss) * 0.1

        loss = args.speed_weight * speed_loss + branch_loss + weat_loss + mi_loss + posi_loss * 0.2

        losses.update(loss.item(), sequence_num)
        weat_losses.update(weat_loss.item(), sequence_num)
        branch_losses.update(branch_loss.item(), sequence_num)
        speed_losses.update(speed_loss.item(), sequence_num)

        mi_real_losses.update(mi_real_loss.item(), sequence_num)
        mi_fake_losses.update(mi_fake_loss.item(), sequence_num)
        mi_max_losses.update(mi_max_loss.item(), sequence_num)
        posi_losses.update(posi_loss.item(), sequence_num)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 or i == len(loader):
            writer.add_scalar('train/weat_loss', weat_losses.val, step+i)
            writer.add_scalar('train/branch_losses', branch_losses.val, step + i)
            writer.add_scalar('train/speed_loss', speed_losses.val, step+i)
            writer.add_scalar('train/mi_real_loss', mi_real_losses.val, step+i)
            writer.add_scalar('train/mi_fake_loss', mi_fake_losses.val, step+i)
            writer.add_scalar('train/mi_max_loss', mi_max_losses.val, step+i)
            writer.add_scalar('train/posi_loss', posi_losses.val, step+i)
            writer.add_scalar('train/loss', losses.val, step+i)
            output_log(
                '[{0}][{1}/{2}] '
                'Time {batch_time.val:.5f} ({batch_time.avg:.5f}) '
                'weat {weat_loss.val:.5f} ({weat_loss.avg:.5f}) '
                'brahch {branch_loss.val:.5f} ({branch_loss.avg:.5f}) '
                'Speed {speed_loss.val:.5f} ({speed_loss.avg:.5f}) '
                'MI_R {mi_real_loss.val:.5f} ({mi_real_loss.avg:.5f}) '
                'MI_F {mi_fake_loss.val:.5f} ({mi_fake_loss.avg:.5f}) '
                'MI_M {mi_max_loss.val:.5f} ({mi_max_loss.avg:.5f}) '
                'Posi {posi_loss.val:.5f} ({posi_loss.avg:.5f}) '
                'Loss {loss.val:.5f} ({loss.avg:.5f}) '
                .format(
                    epoch, i, len(loader), batch_time=batch_time,
                    weat_loss=weat_losses,
                    branch_loss=branch_losses,
                    speed_loss=speed_losses,
                    mi_real_loss=mi_real_losses,
                    mi_fake_loss=mi_fake_losses,
                    mi_max_loss=mi_max_losses,
                    posi_loss=posi_losses,
                    loss=losses), logging)

    return losses.avg

def evaluate(loader, model, model_VAE, criterion, epoch, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    branch_losses = AverageMeter()
    weat_losses = AverageMeter()
    # adv_weat_losses = AverageMeter()
    speed_losses = AverageMeter()

    mi_real_losses = AverageMeter()
    mi_fake_losses = AverageMeter()
    mi_max_losses = AverageMeter()
    posi_losses = AverageMeter()

    steer_losses = AverageMeter()
    gas_losses = AverageMeter()
    brake_losses = AverageMeter()
    # switch to evaluate mode
    model.eval()
    step = epoch * len(loader)
    with torch.no_grad():
        end = time.time()
        for i, (img, speed, target, mask, weather, posi_list) in enumerate(loader):
            data_time.update(time.time() - end)

            # if args.gpu is not None:
            img = img.squeeze(0).cuda(args.gpu, non_blocking=True)
            speed = speed.squeeze(0).cuda(args.gpu, non_blocking=True)
            target = target.squeeze(0).cuda(args.gpu, non_blocking=True)
            mask = mask.squeeze(0).cuda(args.gpu, non_blocking=True)
            weather = weather.squeeze(0).cuda(args.gpu, non_blocking=True)
            posi = posi_list.squeeze(0).cuda(args.gpu, non_blocking=True)
            speed = speed.view(-1, 1)

            sequence_num = img.shape[0]

            mu, logsigma, classcode = model_VAE(img)

            weat_out, pred_speed, pred_posi, act_output1, act_output2, act_output3, act_output4, \
            mi_real, mi_fake, Ej_l, Em_l, Ej_g, Em_g, term_a, term_b \
                = model(img, speed, mu, logsigma, classcode)

            ''' weather '''
            # weat_loss = criterion[1](weat_out, weather)

            ''' actions '''
            weat_out_prob = F.softmax(weat_out, dim=1)
            mask_out1 = torch.mul(act_output1 * mask, weat_out_prob[:, 0].view(-1, 1))
            mask_out2 = torch.mul(act_output2 * mask, weat_out_prob[:, 1].view(-1, 1))
            mask_out3 = torch.mul(act_output3 * mask, weat_out_prob[:, 2].view(-1, 1))
            mask_out4 = torch.mul(act_output4 * mask, weat_out_prob[:, 3].view(-1, 1))
            mask_out = mask_out1 + mask_out2 + mask_out3 + mask_out4

            branch_loss = criterion[0](mask_out, target) * 4
            speed_loss = criterion[0](pred_speed, speed)
            posi_loss = criterion[0](pred_posi, posi)

            ''' MI loss '''
            y_real = torch.ones(sequence_num).cuda(args.gpu, non_blocking=True)
            y_fake = torch.zeros(sequence_num).cuda(args.gpu, non_blocking=True)
            mi_real_loss = criterion[2](mi_real.squeeze(), y_real) * 0.5
            mi_fake_loss = criterion[2](mi_fake.squeeze(), y_fake) * 0.5

            alpha = 0.2
            beta = 0.2
            gamma = 0.1
            LOCAL = (Em_l.mean() - Ej_l.mean()) * beta
            GLOBAL = (Em_g.mean() - Ej_g.mean()) * alpha
            PRIOR = - (term_a.mean() + term_b.mean()) * gamma
            mi_max_loss = LOCAL + GLOBAL + PRIOR
            mi_loss = (mi_real_loss + mi_fake_loss + mi_max_loss) * 0.2

            loss = args.speed_weight * speed_loss + branch_loss  + mi_loss + posi_loss * 0.5

            losses.update(loss.item(), sequence_num)
            # weat_losses.update(weat_loss.item(), sequence_num)
            branch_losses.update(branch_loss.item(), sequence_num)
            speed_losses.update(speed_loss.item(), sequence_num)

            mi_real_losses.update(mi_real_loss.item(), sequence_num)
            mi_fake_losses.update(mi_fake_loss.item(), sequence_num)
            mi_max_losses.update(mi_max_loss.item(), sequence_num)
            posi_losses.update(posi_loss.item(), sequence_num)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            ''' loss split for debugging'''
            # steer gas brake
            target_steer = torch.cat([target[:, 0].unsqueeze(1), target[:, 3].unsqueeze(1),
                                      target[:, 6].unsqueeze(1), target[:, 9].unsqueeze(1)], dim=1)
            target_gas = torch.cat([target[:, 1].unsqueeze(1), target[:, 4].unsqueeze(1),
                                    target[:, 7].unsqueeze(1), target[:, 10].unsqueeze(1)], dim=1)
            target_brake = torch.cat([target[:, 2].unsqueeze(1), target[:, 5].unsqueeze(1),
                                      target[:, 8].unsqueeze(1), target[:, 11].unsqueeze(1)], dim=1)
            mask_steer = torch.cat([mask_out[:, 0].unsqueeze(1), mask_out[:, 3].unsqueeze(1),
                                    mask_out[:, 6].unsqueeze(1), mask_out[:, 9].unsqueeze(1)], dim=1)
            mask_gas = torch.cat([mask_out[:, 1].unsqueeze(1), mask_out[:, 4].unsqueeze(1),
                                  mask_out[:, 7].unsqueeze(1), mask_out[:, 10].unsqueeze(1)], dim=1)
            mask_brake = torch.cat([mask_out[:, 2].unsqueeze(1), mask_out[:, 5].unsqueeze(1),
                                    mask_out[:, 8].unsqueeze(1), mask_out[:, 11].unsqueeze(1)], dim=1)

            steer_loss = criterion[0](mask_steer, target_steer)
            gas_loss = criterion[0](mask_gas, target_gas)
            brake_loss = criterion[0](mask_brake, target_brake)
            steer_losses.update(steer_loss.item(), img.shape[0])
            gas_losses.update(gas_loss.item(), img.shape[0])
            brake_losses.update(brake_loss.item(), img.shape[0])


            if i % args.print_freq == 0 or i == len(loader):
                writer.add_scalar('eval/loss', losses.val, step+i)
                output_log(
                  'Test: [{0}/{1}] '
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                'weat {weat_loss.val:.5f} ({weat_loss.avg:.5f}) '
                'brahch {branch_loss.val:.5f} ({branch_loss.avg:.5f}) '
                'Speed {speed_loss.val:.5f} ({speed_loss.avg:.5f}) '
                'MI_R {mi_real_loss.val:.5f} ({mi_real_loss.avg:.5f}) '
                'MI_F {mi_fake_loss.val:.5f} ({mi_fake_loss.avg:.5f}) '
                'MI_M {mi_max_loss.val:.5f} ({mi_max_loss.avg:.5f}) '
                's {steer_loss.val:.5f} ({steer_loss.avg:.5f}) '
                'g {gas_loss.val:.5f} ({gas_loss.avg:.5f}) '
                'b {brake_loss.val:.5f} ({brake_loss.avg:.5f}) '
                'Posi {posi_loss.val:.5f} ({posi_loss.avg:.5f}) '
                  'Loss {loss.val:.4f} ({loss.avg:.4f}) '
                  .format(
                      i, len(loader), batch_time=batch_time,
                        weat_loss=weat_losses,
                        branch_loss=branch_losses,
                        speed_loss=speed_losses,
                        mi_real_loss=mi_real_losses,
                        mi_fake_loss=mi_fake_losses,
                        mi_max_loss=mi_max_losses,
                        steer_loss=steer_losses,
                        gas_loss=gas_losses,
                        brake_loss=brake_losses,
                        posi_loss=posi_losses,
                      loss=losses), logging)
                # print(weat_out_prob, weather)
    return losses.avg


if __name__ == '__main__':
    main()

