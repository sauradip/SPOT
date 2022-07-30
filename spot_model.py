# -*- coding: utf-8 -*-
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.transformer import SnippetEmbedding
import yaml
import random
# from performer_pytorch import Performer

with open("./config/anet.yaml", 'r', encoding='utf-8') as f:
        tmp = f.read()
        config = yaml.load(tmp, Loader=yaml.FullLoader)


class TemporalShift(nn.Module):
    def __init__(self, n_segment=3, n_div=8, inplace=False):
        super(TemporalShift, self).__init__()
        # self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div
        self.inplace = inplace
        self.channels_range = list(range(400))  # feature_channels
        if inplace:
            print('=> Using in-place shift...')
        # print('=> Using fold div: {}'.format(self.fold_div))

    def forward(self, x):
        # self.fold_div = n_div
        x = self.shift(x, self.n_segment, fold_div=self.fold_div, inplace=self.inplace, channels_range =self.channels_range)
        return x

    @staticmethod
    def shift(x, n_segment, fold_div=8, inplace=False, channels_range=[1,2]):
        x = x.permute(0, 2, 1)   # [B,C,T] --> [B, T, C]
        # set_trace()
        n_batch, T, c = x.size()
        # nt, c, h, w = x.size()
        # n_batch = nt // n_segment
        # x = x.view(n_batch, n_segment, c, h, w)
        # x = x.view(n_batch, T, c, h, w)
        fold = c // 2*fold_div
        # all = random.sample(channels_range, fold*2)
        # forward = sorted(all[:fold])
        # backward = sorted(all[fold:])
        # fixed = list(set(channels_range) - set(all))
        # fold = c // fold_div

        if inplace:
            # Due to some out of order error when performing parallel computing.
            # May need to write a CUDA kernel.
            raise NotImplementedError
            # out = InplaceShift.apply(x, fold)
        else:
            out = torch.zeros_like(x)
            out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
            out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
            out[:, :, 2 * fold:200] = x[:, :, 2 * fold:200]  # not shift

            out[:, :-1, 200:200+fold] = x[:, 1:, 200:200+fold]  # shift left
            out[:, 1:, 200+fold: 200+2 * fold] = x[:, :-1, 200+fold: 200+2 * fold]  # shift right
            out[:, :, 200+2 * fold:] = x[:, :, 200 + 2 * fold:]  # not shift
            # out = torch.zeros_like(x)
            # out[:, :-1, forward] = x[:, 1:, forward]  # shift left
            # out[:, 1:, backward] = x[:, :-1, backward]  # shift right
            # out[:, :, fixed] = x[:, :, fixed]  # not shift

        # return out.view(nt, c, h, w)
        return out.permute(0, 2, 1)


class TemporalShift_random(nn.Module):
    def __init__(self, n_segment=3, n_div=8, inplace=False):
        super(TemporalShift_random, self).__init__()
        # self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div
        self.inplace = inplace
        self.channels_range = list(range(400))  # feature_channels
        if inplace:
            print('=> Using in-place shift...')
        # print('=> Using fold div: {}'.format(self.fold_div))

    def forward(self, x):
        # self.fold_div = n_div
        x = self.shift(x, self.n_segment, fold_div=self.fold_div, inplace=self.inplace, channels_range =self.channels_range)
        return x

    @staticmethod
    def shift(x, n_segment, fold_div=8, inplace=False, channels_range=[1,2]):
        x = x.permute(0, 2, 1)   # [B,C,T] --> [B, T, C]
        # set_trace()
        n_batch, T, c = x.size()
        # nt, c, h, w = x.size()
        # n_batch = nt // n_segment
        # x = x.view(n_batch, n_segment, c, h, w)
        # x = x.view(n_batch, T, c, h, w)
        fold = c // fold_div
        all = random.sample(channels_range, fold*2)
        forward = sorted(all[:fold])
        backward = sorted(all[fold:])
        fixed = list(set(channels_range) - set(all))
        # fold = c // fold_div

        if inplace:
            # Due to some out of order error when performing parallel computing.
            # May need to write a CUDA kernel.
            raise NotImplementedError
            # out = InplaceShift.apply(x, fold)
        else:
            # out = torch.zeros_like(x)
            # out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
            # out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
            # out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift
            out = torch.zeros_like(x)
            out[:, :-1, forward] = x[:, 1:, forward]  # shift left
            out[:, 1:, backward] = x[:, :-1, backward]  # shift right
            out[:, :, fixed] = x[:, :, fixed]  # not shift

        # return out.view(nt, c, h, w)
        return out.permute(0, 2, 1)


class InplaceShift(torch.autograd.Function):
    # Special thanks to @raoyongming for the help to this function
    @staticmethod
    def forward(ctx, input, fold):
        # not support higher order gradient
        # input = input.detach_()
        ctx.fold_ = fold
        n, t, c, h, w = input.size()
        buffer = input.data.new(n, t, fold, h, w).zero_()
        buffer[:, :-1] = input.data[:, 1:, :fold]
        input.data[:, :, :fold] = buffer
        buffer.zero_()
        buffer[:, 1:] = input.data[:, :-1, fold: 2 * fold]
        input.data[:, :, fold: 2 * fold] = buffer
        return input

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output = grad_output.detach_()
        fold = ctx.fold_
        n, t, c, h, w = grad_output.size()
        buffer = grad_output.data.new(n, t, fold, h, w).zero_()
        buffer[:, 1:] = grad_output.data[:, :-1, :fold]
        grad_output.data[:, :, :fold] = buffer
        buffer.zero_()
        buffer[:, :-1] = grad_output.data[:, 1:, fold: 2 * fold]
        grad_output.data[:, :, fold: 2 * fold] = buffer
        return grad_output, None


class SPOT(nn.Module):
    def __init__(self):
        super(SPOT, self).__init__()
        self.len_feat = config['model']['feat_dim']
        self.temporal_scale = config['model']['temporal_scale']
        self.num_classes = config['dataset']['num_classes']+1
        self.n_heads = config['model']['embedding_head']
        self.win_softmax = nn.Softmax(dim=-1)
        # self.embedding = Performer(
        #                             dim = self.len_feat,
        #                             depth = 1,
        #                             heads = 1,
        #                             causal = True,
        #                             attn_dropout = 0.3,
        #                             dim_head = 100
        #                         )   

        self.embedding = SnippetEmbedding(self.n_heads, self.len_feat, self.len_feat, self.len_feat, 0.3)
        self.clip_trans = SnippetEmbedding(self.n_heads, self.len_feat, self.len_feat, self.len_feat , 0.1,True)

        self.classifier = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feat, out_channels=self.num_classes, kernel_size=1,
            padding=0)
        )

        self.clip_order_drop = nn.Dropout(0.5)
        self.clip_order_linear = nn.Linear(100, 2)

        self.clip_order = nn.Sequential(
            nn.Conv1d(400, 1, kernel_size=3, padding=1),  # 256
            nn.ReLU(inplace=True)
        )

        self.maxpool_1 = nn.MaxPool1d(3, stride=2)
        self.maxpool_2 = nn.MaxPool1d(3, stride=3)
        self.maxpool_3 = nn.MaxPool1d(3,stride=4)

        self.global_mask = nn.Sequential(
            nn.Conv1d(in_channels=400, out_channels=256, kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=256, out_channels=self.temporal_scale, kernel_size=1,stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def window_pool(self,snippet,pad = 10, window = 10):
        #### pad
        snip_pad = F.pad(snippet,(pad,pad),"constant", 0)
        n_b , n_f, n_t = snip_pad.size()
        win = window
        ## break into windows ##
        snip_win = snip_pad.view(n_b,n_f,n_t//win,win)
        snip_win = snip_win.permute(0,2,1,3).contiguous().view(-1,n_f,win)
        ## pool the windows ##
        snip_win_pool = self.maxpool_1(snip_win)
        n_b_new,_,pool_win = snip_win_pool.size()
        new_t = n_b_new//n_b
        n_b_new_1 = int(n_b_new/ (n_t  / pool_win / pool_win))

        snip_win_pool = snip_win_pool.view(n_b_new_1, -1,n_t // pool_win, pool_win)
        snip_win_pool = snip_win_pool.permute(0, 2,1,3).contiguous().view(-1,n_f,n_t)

        return snip_win_pool



    def forward(self, snip, recons=False, clip_order=False):

        snip_ = snip.permute(0,2,1)

        out = self.embedding(snip_,snip_,snip_)
        out = out.permute(0,2,1)
        batch_size,_,_ = out.size()
        features = out
        if clip_order:            
            new_feat = features.permute(0,2,1)           
            new_feat_order = self.clip_trans(new_feat,new_feat,new_feat)
            self_att_feat = new_feat.permute(0,2,1)
            clip_drop = self.clip_order_drop(self.clip_order(self_att_feat).view(batch_size, 100))

            return self.clip_order_linear(clip_drop)
            
        if recons:
            return features
        ### Classifier Branch ###
        top_br = self.classifier(features)
        ### Global Segmentation Mask Branch ###
        bottom_br = self.global_mask(features)

        return top_br, bottom_br, features

