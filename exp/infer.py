# /usr/bin/python
# -*- coding:utf-8 -*-
"""
    @description: test script:
    @detail:
    @copyright: xxx
    @author: Seven Hsu
    @e-mail: xxx
    @date: xxx
"""
import cv2
import sys

sys.path.append('.')
import torch
import models
import numpy as np
from utils import load_state
from cfgs.config import opt

model = None


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
    # B,N,h,w and softmax
    ord_seg_map = ord_seg_map.softmax(1)

    # ignore background
    chars_seg_map = chars_seg_map[:, 1:]
    ord_seg_map = ord_seg_map[:, 1:]

    # B,N,h,w(ignore background)
    order_map = ord_seg_map * pos_seg_map

    # =========order map filter==========
    # B,N,h*w
    tmp = order_map.reshape(order_map.shape[0], order_map.shape[1], -1)
    max_prob, _ = torch.max(tmp, 2)
    b, n = torch.where(max_prob < t_score)
    for b_i, n_i in zip(b, n):
        order_map[b_i, n_i] = 0.
    # ====================================

    # ==========show seg map==============
    pos_seg_map = pos_seg_map.detach().cpu().numpy()
    pos_seg_map = (pos_seg_map * 255).astype(np.uint8)[0, 0]

    cv2.imwrite('show/pos_seg.jpg', pos_seg_map)
    for i in range(ord_seg_map.shape[1]):
        cv2.imwrite("show/order_seg_%d.jpg" % i, (ord_seg_map[0][i].detach().cpu().numpy() * 255).astype(np.uint8))
    for i in range(order_map.shape[1]):
        cv2.imwrite("show/order_map_%d.jpg" % i, (order_map[0][i].detach().cpu().numpy() * 255).astype(np.uint8))
    for i in range(chars_seg_map.shape[1]):
        cv2.imwrite("show/chars_seg_map_%d.jpg" % i,
                    (chars_seg_map[0][i].detach().cpu().numpy() * 255).astype(np.uint8))
    # ====================================

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


def parse_opt(**kwargs):
    opt.parse(kwargs)
    assert opt.load_model_path is not None, 'model weight should not be None'
    global model
    model = getattr(models, opt.model)(opt.input_size, opt.max_seq, opt.num_classes, mode='test', attn=opt.attn)
    load_state(model, opt.load_model_path, "cuda:%d" % opt.gpus[0])
    model = gpu(model, opt)
    model.eval()


def infer(img):
    t_score = 0.5

    img = cv2.resize(img, (384, 96))
    img = np.transpose(img, [2, 0, 1])
    img = np.expand_dims(img, 0)
    inputs = torch.as_tensor(img / 255., dtype=torch.float32)
    inputs = gpu(inputs, opt)
    h0 = gpu(torch.zeros(1, inputs.shape[0], 384), opt)
    with torch.no_grad():
        outputs = model(inputs, h0)
    # get word format from model output
    outputs = word_format(outputs, t_score)[0].detach().cpu().numpy()
    outputs = outputs[np.where(np.max(outputs, 1) != 0)[0]]
    idx = np.argmax(outputs, 1)
    preds = ''.join([opt.chars_list[i + 1] for i in idx])
    return preds


if __name__ == '__main__':
    import fire
    from glob import glob

    fire.Fire(parse_opt)
    test_dir = 'data/images'
    img_list = glob(test_dir + '/*jpg') + glob(test_dir + '/*png')
    for img_path in img_list:
        img = cv2.imread(img_path)
        pred = infer(img)
        print('image:%s,pred:%s' % (img_path, pred))
