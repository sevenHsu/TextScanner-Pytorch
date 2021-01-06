# /usr/bin/python
# -*- coding:utf-8 -*-
"""
    @description: training script
    @detail:
    @copyright: xxx
    @author: Seven Hsu
    @e-mail: xxx
    @date: xxx
"""
import sys

sys.path.append('.')
import os
import time
import math
import torch
import models
from torchnet import meter
from cfgs.config import opt
from tensorboardX import SummaryWriter
from dataset.dataset import OCRDataset
from torch.utils.data import DataLoader, ConcatDataset
from torch.nn.functional import cross_entropy, smooth_l1_loss
from utils.utils import load_state, create_logger, AverageMeter

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'


def gpu(obj, opt):
    try:
        return obj.to(opt.gpus[0])
    except Exception as e:
        return obj


def target_weight(target, num_classes):
    n = target.shape[0]
    h, w = target.shape[1:]
    n_total = n * h * w
    n_neg = (target == 0).sum().item()

    weight_p = 1  # torch.sum(target) / (n*h*w)
    weight_n = n_neg / (n_total - n_neg)  # 1 - weight_p
    res = torch.ones(num_classes) * weight_n
    res[0] = weight_p

    return res


def get_loss(chars_seg, order_seg, pos_seg, gt_chars_seg, gt_order_seg, gt_pos_seg, opt):
    lambda_l, lambda_o = 1, 1
    maxpool = gpu(torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1), opt)
    target_chars = maxpool(gt_chars_seg.float())
    target_order = maxpool(gt_order_seg.float())
    target_pos = maxpool(gt_pos_seg.float())

    chars_weight = target_weight(target_chars, opt.num_classes).to(opt.gpus[0])
    order_weight = target_weight(target_order, opt.max_seq).to(opt.gpus[0])

    loss_char = cross_entropy(chars_seg, target_chars.long(), chars_weight)
    loss_order = cross_entropy(order_seg, target_order.long(), order_weight)
    loss_pos = smooth_l1_loss(pos_seg, target_pos.unsqueeze(1))

    return loss_char + lambda_o * loss_order + lambda_l * loss_pos


def train(**kwargs):
    opt.parse(kwargs)
    if not os.path.exists(opt.save_folder):
        os.mkdir(opt.save_folder)
    tb_logger = SummaryWriter(opt.save_folder)
    logger = create_logger('global_logger', opt.save_folder + '/log.txt')
    batch_time = AverageMeter(10)
    data_time = AverageMeter(10)
    losses = AverageMeter(10)
    loss_meter = meter.AverageValueMeter()

    train_sets = []
    for data_txt in opt.train_txt:
        data_root, gt_root, list_file = data_txt.split(' ')
        train_sets.append(
            OCRDataset(data_root, gt_root, list_file, opt.input_size, 'train', opt.chars_list, opt.max_seq))
    train_data = ConcatDataset(train_sets)
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_works)

    valid_sets = []
    for valid_txt in opt.valid_txt:
        data_root, gt_root, list_file = valid_txt.split(' ')
        valid_sets.append(
            OCRDataset(data_root, gt_root, list_file, opt.input_size, 'valid', opt.chars_list, opt.max_seq))
    valid_data = ConcatDataset(valid_sets)
    valid_loader = DataLoader(valid_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_works)

    model = getattr(models, opt.model)(opt.basenet, opt.input_size, opt.max_seq,
                                       opt.num_classes, mode='train', attn=opt.attn)

    if opt.load_model_path is not None:
        load_state(model, opt.load_model_path, 'cuda:%d' % opt.gpus[0])
    if len(opt.gpus) > 1:
        model = torch.nn.DataParallel(model, device_ids=opt.gpus)
    model = gpu(model, opt)

    if len(opt.gpus) > 1:
        optimizer = torch.optim.Adam(model.module.parameters(), lr=opt.lr, betas=opt.betas,
                                     weight_decay=opt.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=opt.betas, weight_decay=opt.weight_decay)

    curr_step = 0
    total_step = int(len(train_data) / opt.batch_size * opt.epoches)
    best_val_error = 1e10
    previous_loss = 1e10
    # warmup
    warmup_epoches = opt.epoches // 10
    warmup_rate = math.pow(100, 1 / warmup_epoches)

    for epoch in range(opt.epoches):
        model.train()
        end = time.time()
        # loss_meter.reset()
        for i, (imgs, gt_chars_seg, gt_order_seg, gt_pos_seg) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            # zero the parameter gradients
            optimizer.zero_grad()

            imgs = gpu(imgs, opt)
            gt_chars_seg = gpu(gt_chars_seg, opt)
            gt_order_seg = gpu(gt_order_seg, opt)
            gt_pos_seg = gpu(gt_pos_seg, opt)

            chars_seg, ord_seg, pos_seg = model(imgs)
            loss = get_loss(chars_seg, ord_seg, pos_seg, gt_chars_seg, gt_order_seg, gt_pos_seg, opt)

            loss.backward()
            optimizer.step()
            losses.update(loss.item())
            loss_meter.add(loss.item())
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            curr_step += 1
            current_lr = optimizer.param_groups[0]['lr']
            if curr_step % opt.print_freq == 0:
                tb_logger.add_scalar('loss_train', losses.avg, curr_step)
                tb_logger.add_scalar('lr', current_lr, curr_step)
                logger.info('Iter: [{0}/{1}]\t'
                            'Epoch: {2}\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'LR {lr:.4f}'.format(curr_step, total_step, epoch, batch_time=batch_time,
                                                 data_time=data_time, loss=losses, lr=current_lr))

        # val
        model.eval()
        val_error = val(model, valid_loader, opt)
        logger.info('Mean error: {0}\t'.format(val_error))
        if not tb_logger is None:
            tb_logger.add_scalar('error_val', val_error, curr_step)
        if val_error < best_val_error:
            best_val_error = val_error
            if len(opt.gpus) > 1:
                torch.save(model.module.state_dict(), os.path.join(opt.save_folder, "best_val_error.pth"))
            else:
                torch.save(model.state_dict(), os.path.join(opt.save_folder, "best_val_error.pth"))
        # warmup
        if epoch < warmup_epoches:
            for param_group in optimizer.param_groups:
                param_group["lr"] *= warmup_rate
                # decay lr if loss no longer decrease
        else:
            if opt.lr_immediate_decay and loss_meter.value()[0] > previous_loss:
                for param_group in optimizer.param_groups:
                    param_group["lr"] *= opt.lr_decay

            if epoch == int(opt.epoches * 0.6) or epoch == int(opt.epoches * 0.9):
                for param_group in optimizer.param_groups:
                    param_group["lr"] *= opt.lr_decay
            previous_loss = loss_meter.value()[0]
    # save last pth
    if len(opt.gpus) > 1:
        torch.save(model.module.state_dict(), os.path.join(opt.save_folder, "last.pth"))
    else:
        torch.save(model.state_dict(), os.path.join(opt.save_folder, "last.pth"))


def val(model, valid_loader, opt):
    batch_time = AverageMeter(10)
    error_sum = 0
    n_correct = 0
    val_n = 0
    for i, (imgs, gt_chars_seg, gt_order_seg, gt_pos_seg) in enumerate(valid_loader):
        imgs = gpu(imgs, opt)
        gt_chars_seg = gpu(gt_chars_seg, opt)
        gt_order_seg = gpu(gt_order_seg, opt)
        gt_pos_seg = gpu(gt_pos_seg, opt)
        end = time.time()

        with torch.no_grad():
            chars_seg, order_seg, pos_seg = model(imgs)
        batch_time.update(time.time() - end)
        loss = get_loss(chars_seg, order_seg, pos_seg, gt_chars_seg, gt_order_seg, gt_pos_seg, opt)

        abs_error = loss
        error_sum += abs_error.item()
        val_n += 1

    print('batch time: {}'.format(batch_time.avg))
    print('###### val mean error: %f' % (error_sum / val_n))
    print('###### accuracy: %f' % (n_correct / val_n))
    return error_sum / val_n


if __name__ == "__main__":
    import fire

    fire.Fire()
