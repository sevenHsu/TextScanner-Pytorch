# /usr/bin/python
# -*- coding:utf-8 -*-
"""
    @description: test script:
    @detail:
    @copyright: Chonqqing Ainnovation Tech Co., Ltd.
    @author: Seven Hsu
    @e-mail: xushen@ainnovation.com
    @date: 2020-12-21
"""
import sys

sys.path.append('.')
import torch
import models
import numpy as np
from utils import load_state
from cfgs.config import opt
from dataset.dataset import OCRDataset
from torch.utils.data import DataLoader


def gpu(obj, opt):
    try:
        return obj.to(opt.gpus[0])
    except Exception as e:
        return obj


def manual_mat(ordmap, charseg):
    """

    :param ordmap: B,N,HW
    :param charseg: B,HW,C
    :return:
    """
    assert charseg.shape[0] == ordmap.shape[0], 'batch size should be same between order map and chars seg'

    result = torch.rand(ordmap.shape[0], ordmap.shape[1], charseg.shape[1])
    for i in range(result.shape[0]):
        for j in range(ordmap.shape[1]):
            out = charseg[i] * ordmap[i][j]
            result[i][j] = torch.sum(out, dim=[1, 2])
    return result


def word_format(inputs, t_score):
    chars_seg_map, ord_seg_map, pos_seg_map = inputs
    # B,C,h,w and softmax
    chars_seg_map = chars_seg_map.softmax(1)

    # B,N,h,w
    ord_seg_map = ord_seg_map.softmax(1)

    # B,N,h,w
    order_map = ord_seg_map * pos_seg_map

    # =========order map filter==========
    # B,N,h*w
    tmp = order_map.reshape(order_map.shape[0], order_map.shape[1], -1)
    total = torch.sum(tmp, 1).unsqueeze(1).repeat(1, tmp.shape[1], 1)
    tmp = tmp / (total + 1e-9)
    max_prob, _ = torch.max(tmp, 2)
    b, n = torch.where(max_prob < t_score)
    for b_i, n_i in zip(b, n):
        order_map[b_i, n_i] = 0.
    # ===================================

    # (chars_seg_map):B,C,h,w->B,C,h*w
    chars_seg_map = chars_seg_map.reshape((chars_seg_map.shape[0], chars_seg_map.shape[1], -1))
    # (chars_seg_map):B,h*w,C
    chars_seg_map = chars_seg_map.permute([0, 2, 1])
    # (order_map):B,N,h,w->B,N,h*w
    order_map = order_map.reshape((order_map.shape[0], order_map.shape[1], -1))
    # B,N,C
    word_format = torch.matmul(order_map, chars_seg_map)
    # calc word_format manually
    # word_format = manual_mat(order_map, chars_seg_map)

    return word_format


def test(**kwargs):
    opt.parse(kwargs)
    dataset = OCRDataset('data/images', 'data/labels', 'data/test.imglist',
                         opt.input_size, 'test', opt.chars_list, opt.max_seq)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_works)

    model = getattr(models, opt.model)(opt.basenet, opt.input_size, opt.max_seq,
                                       opt.num_classes, mode='test', attn=opt.attn)
    load_state(model, opt.load_model_path, "cuda:%d" % opt.gpus[0])
    model = gpu(model, opt)
    model.eval()
    t_score = 0.3
    match, all = 0, 0
    for inputs, text in dataloader:
        inputs = gpu(inputs, opt)
        with torch.no_grad():
            outputs = model(inputs)
        outputs = word_format(outputs, t_score)
        outputs = outputs[0].detach().cpu().numpy()
        outputs = outputs[np.where(np.max(outputs, 1) != 0)[0]]
        idx = np.argmax(outputs, 1)
        idx = idx[np.where(idx != 0)[0]]
        preds = ''.join([opt.chars_list[i] for i in idx])
        text = text[0]
        if text == preds:
            match += 1
        else:
            print('text/pred:%s,%s' % (text, preds))
        all += 1
        torch.cuda.empty_cache()
    print('match/all(%2f): %d/%d' % (match / all, match, all))


if __name__ == '__main__':
    import fire

    fire.Fire()
