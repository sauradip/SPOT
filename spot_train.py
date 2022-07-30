
import os
import torch
import torch.nn.parallel
import torch.optim as optim
from torch import autograd
import numpy as np
# from gsm_lib import opts
from spot_model import SPOT, TemporalShift, TemporalShift_random
import yaml
import spot_lib.spot_dataloader as spot_dataset
from spot_lib.loss_spot import spot_loss, spot_loss_bot, ce_loss_thresh, bottom_branch_loss, top_ce_loss, ACSL
# from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import pandas as pd
import random
import torch.nn.functional as F
from spot_lib.augmentations import TemporalHalf, TemporalReverse, TemporalCutOut , RandAugment
from utils.contrastive_refinement import easy_snippets_mining, hard_snippets_mining, SniCoLoss
import itertools,operator
from scipy import ndimage
from collections import Counter
from spot_lib.tsne import viusalize
# writer = SummaryWriter()


contrast_loss = SniCoLoss()
# writer = SummaryWriter()

acsl_loss = ACSL()

with open("./config/anet.yaml", 'r', encoding='utf-8') as f:
        tmp = f.read()
        config = yaml.load(tmp, Loader=yaml.FullLoader)

output_path=config['dataset']['training']['output_path']
num_gpu = config['training']['num_gpu']
batch_size = config['training']['batch_size']
learning_rate = config['training']['learning_rate']
decay = config['training']['weight_decay']
epoch = config['training']['max_epoch']
num_batch = config['training']['batch_size']
step_train = config['training']['step']
gamma_train = config['training']['gamma']
fix_seed = config['training']['random_seed']
use_semi = config['dataset']['training']['use_semi']
unlabel_percent = config['dataset']['training']['unlabel_percent']


################## fix everything ##################
import random
seed = fix_seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#######################################################


def top_lr_loss(target,pred):

    gt_action = target
    pred_action = pred
    topratio = 0.6
    num_classes = 200
    alpha = 10

    pmask = (gt_action == 1).float()
    nmask = (gt_action == 0).float()
    nmask = nmask 

    num_positive = 10 + torch.sum(pmask) # in case of nan
    num_entries = 10 + num_positive + torch.sum(nmask)
    ratio = num_entries / num_positive
    coef_0 = 0.5 * ratio / (ratio - 1)
    coef_1 = 0.5 * ratio
    eps = 0.000001
    pred_p = torch.log(pred_action + eps)
    pred_n = torch.log(1.0 - pred_action + eps)


    topk = int(num_classes * topratio)
    # targets = targets.cuda()
    count_pos = num_positive
    hard_neg_loss = -1.0 * (1.0-gt_action) * pred_n
    topk_neg_loss = -1.0 * hard_neg_loss.topk(topk, dim=1)[0]#topk_neg_loss with shape batchsize*topk

    loss = (gt_action * pred_p).sum() / count_pos + alpha*(topk_neg_loss.cuda()).mean()

    return -1*loss




def get_mem_usage():
    GB = 1024.0 ** 3
    output = ["device_%d = %.03fGB" % (device, torch.cuda.max_memory_allocated(torch.device('cuda:%d' % device)) / GB) for device in range(num_gpu)]
    return ' '.join(output)[:-1]


blue = lambda x: '\033[94m' + x + '\033[0m'

global_step = 0
consistency_rampup = 5
consistency = 2  # 30  # 3  # None

def temporal_crop(snip):

    ran_start = random.randint(0,100)
    if (ran_start + 10) > 100:
        ran_end = 100
    else:
        ran_end = random.randint(ran_start+10,100)
    feat_crop = snip[:,]



def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()

    return F.mse_loss(input_logits, target_logits, reduction='mean')

def TemporalCrop(input_feat,top_br):

    n_btach, feat_dim, tmp_dim = input_feat.size()
    n_batch ,_,_ = top_br.size()
    batch_start = np.zeros([n_batch])
    batch_end = np.zeros([n_batch])
    temp_mask_cls = np.zeros([100])
    
    bottom_gt = np.zeros([n_batch,100,100])
    fg_action_idx = torch.argmax(top_br[:,:200,:],dim=1)
    fg_action_idx_mode, _ = torch.mode(fg_action_idx,dim=1)
    new_mask = np.zeros([n_batch,100])
    empty_gt = np.zeros_like(top_br.detach().cpu().numpy())
    labeled_gt = np.zeros([n_batch,200])

    for p in range(100):
        new_mask[:,p] = -1 
    temp_data = torch.zeros_like(input_feat)
    for i in range(0,n_batch):
        batch_feat = input_feat[i,:,:]
        # batch_feat 
        batch_feat -= batch_feat.min(1, keepdim=True)[0]
        batch_feat /= batch_feat.max(1, keepdim=True)[0] - batch_feat.min(1, keepdim=True)[0]
       
        for p in range(0,2):
            rand_start = np.random.randint(0,94,size=1)[0]
            rand_end = np.random.randint(rand_start,100,size=1)[0]
            len_snip = rand_end - rand_start
            if len_snip < 5:
                rand_end = rand_end+5
                rand_start = rand_start-5

            if rand_start < 0:
                rand_start = 0
            if rand_end > 99:
                rand_end = 99

            if rand_start > rand_end:
                rand_start = np.random.randint(0,49,size=1)
                rand_end = np.random.randint(rand_start[0],99,size=1)
            temp_data[i,:,rand_start:rand_end] = batch_feat[:,rand_start:rand_end]
            temp_mask_cls[rand_start:rand_end] = 1
            background_mask = 1 - temp_mask_cls
            empty_gt[i,fg_action_idx_mode[i],:] = temp_mask_cls
            empty_gt[i,200,:] = background_mask
            bottom_gt[i,rand_start:rand_end,rand_start:rand_end] = 1
            new_mask[i,rand_start:rand_end] = fg_action_idx_mode[i].detach().cpu().numpy()
            labeled_gt[i,fg_action_idx_mode[i]] = 1
            for p in range(100):
                if new_mask[i,p] == -1:
                    new_mask[i,p] = 200
        

    top_gt_crop = torch.Tensor(new_mask).type(torch.LongTensor)
    bottom_gt_crop = torch.Tensor(bottom_gt)
    mask_top = torch.Tensor(empty_gt)
    label_gt = torch.Tensor(labeled_gt)

    return temp_data, top_gt_crop, bottom_gt_crop , mask_top, label_gt




    
def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()

    return F.kl_div(input_logits, target_logits, reduction='mean')


def Motion_MSEloss(output,clip_label,motion_mask=torch.ones(100).cuda()):
    z = torch.pow((output-clip_label),2)
    loss = torch.mean(motion_mask*z)
    return loss


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return consistency * sigmoid_rampup(epoch, consistency_rampup)



def pretrain(data_loader, model, optimizer):

    not_freeze_class = False
    if not_freeze_class == False:
        model.module.classifier[0].weight.requires_grad = False
        model.module.classifier[0].bias.requires_grad = False
    model.train()
    
    warmup_epoch = 30
    order_clip_criterion = nn.CrossEntropyLoss()
    for ep in range(warmup_epoch):
        for n_iter, (input_data, top_br_gt, bottom_br_gt, action_gt, label_gt, input_data_big, input_data_small, _) in enumerate(data_loader):

            input_data_tdrop = F.dropout(input_data.cuda(),0.1)
            input_data_tdrop_big = F.dropout(input_data_big.cuda(),0.1)
            input_data_tdrop_small = F.dropout(input_data_small.cuda(),0.1)
            input_data_aug = torch.stack([input_data.cuda(),input_data_tdrop],dim=0).view(-1,400,100)
            input_data_aug_b = torch.stack([input_data_big.cuda(),input_data_tdrop_big],dim=0).view(-1,400,200)
            input_data_aug_s = torch.stack([input_data_small.cuda(),input_data_tdrop_small],dim=0).view(-1,400,50)
            top_br_pred, bottom_br_pred, feat = model(input_data_aug)
            mod_input_data, top_br_gt, bottom_br_gt, action_gt, label_gt  = TemporalCrop(input_data_aug,top_br_pred)

            if not_freeze_class:
                easy_dict_label = easy_snippets_mining(top_br_pred, feat) 
                hard_dict_label = hard_snippets_mining(bottom_br_pred, feat)

            top_br_pred_crop, bottom_br_pred_crop, feat_crop = model(mod_input_data)

            if not_freeze_class:
                easy_dict_label_crop = easy_snippets_mining(top_br_pred_crop, feat_crop) 
                hard_dict_label_crop = hard_snippets_mining(bottom_br_pred_crop, feat_crop)


                c_pair_label = {
                    "EA": easy_dict_label[0],
                    "EB": easy_dict_label[1],
                    "HA": hard_dict_label[0],
                    "HB": hard_dict_label[1]
                }

                c_pair_label_crop = {
                    "EA": easy_dict_label_crop[0],
                    "EB": easy_dict_label_crop[1],
                    "HA": hard_dict_label_crop[0],
                    "HB": hard_dict_label_crop[1]
                }

                con_loss = contrast_loss(c_pair_label)
                con_loss_crop = contrast_loss(c_pair_label_crop)

            feat_loss = Motion_MSEloss(feat,input_data_aug)
            feat_loss_crop = Motion_MSEloss(feat,input_data_aug)
            

            # clip order 

            input_data_all = torch.cat([input_data_aug,mod_input_data], 0).view(-1,400,100)
            batch_size, C, T = input_data_all.size()
            idx = torch.randperm(batch_size)
            input_data_all_new = input_data_all[idx]
            forw_input = torch.cat(
                [input_data_all_new[:batch_size // 2, :, T // 2:], input_data_all_new[:batch_size // 2, :, :T // 2]], 2)
            back_input = input_data_all_new[batch_size // 2:, :, :]
            input_all = torch.cat([forw_input, back_input], 0)
            label_order = [0] * (batch_size // 2) + [1] * (batch_size - batch_size // 2)
            label_order = torch.tensor(label_order).long().cuda()
            out = model(input_all ,clip_order=True)
            loss_clip_order = order_clip_criterion(out, label_order)

            if not_freeze_class:
                loss = spot_loss(top_br_gt,top_br_pred,bottom_br_gt,bottom_br_pred, action_gt,label_gt,pretrain=False)
                loss_crop = spot_loss(top_br_gt,top_br_pred,bottom_br_gt,bottom_br_pred, action_gt,label_gt,pretrain=False)
                tot_loss = loss + feat_loss + con_loss
                tot_loss_crop = loss_crop + feat_loss_crop + con_loss_crop
            else:
                loss = spot_loss(top_br_gt,top_br_pred,bottom_br_gt,bottom_br_pred, action_gt,label_gt,pretrain=True)
                loss_crop = spot_loss(top_br_gt,top_br_pred,bottom_br_gt,bottom_br_pred, action_gt,label_gt,pretrain=True)
                tot_loss = feat_loss
                tot_loss_crop = feat_loss_crop 

            final_loss = loss + tot_loss + tot_loss_crop + loss_clip_order
            # update step
            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step()
        if not_freeze_class:    
            print("[Pretraining Epoch {0:03d}] Total-Loss {1:.2f} = T-Loss {2:.2f} + B-Loss {3:.2f} + F-Loss {4:.2f} + C-Loss {5:.2f} + Clip-Loss {6:.2f} (train)".format(
            ep, tot_loss,loss[1],loss[2], feat_loss, con_loss, loss_clip_order))
        else:
            print("[Pretraining Epoch {0:03d}] Total-Loss {1:.2f} =  F-Loss {2:.2f} + Clip-Loss {3:.2f} (train)".format(
            ep, tot_loss,feat_loss,loss_clip_order))

        state = {'epoch': ep + 1,
             'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()}

        torch.save(state, output_path + "/SPOT_pretrain_checkpoint.pth.tar")
        best_loss =  1e10
        if not_freeze_class:
            if loss[0] < best_loss:
                best_loss = loss[0]
                torch.save(state, output_path + "/SPOT_pretrain_best.pth.tar")
        else: 
            if loss < best_loss:
                best_loss = loss
                torch.save(state, output_path + "/SPOT_pretrain_best.pth.tar")



# training
def train(data_loader, model, optimizer, epoch):
    model.train()
    for n_iter, (input_data, top_br_gt, bottom_br_gt, action_gt, label_gt) in enumerate(data_loader):
        # forward pass
        top_br_pred, bottom_br_pred, feat = model(input_data.cuda())

        easy_dict_label = easy_snippets_mining(top_br_pred, feat) 
        hard_dict_label = hard_snippets_mining(bottom_br_pred, feat)

        c_pair_label = {
            "EA": easy_dict_label[0],
            "EB": easy_dict_label[1],
            "HA": hard_dict_label[0],
            "HB": hard_dict_label[1]
        }
        loss_contrast_label = contrast_loss(c_pair_label)
        loss = spot_loss(top_br_gt,top_br_pred,bottom_br_gt,bottom_br_pred, action_gt,label_gt)
        # update step
        tot_loss = loss[0] + 0
        optimizer.zero_grad()
        tot_loss.backward()
        optimizer.step()
    print("[Epoch {0:03d}] Total-Loss {1:.2f} = T-Loss {2:.2f} + B-Loss {3:.2f}  (train)".format(
    epoch, tot_loss,loss[1],loss[2]))

# validation
def test(data_loader, model, epoch, best_loss):
    model.eval()
    with torch.no_grad():
      for n_iter, (input_data, top_br_gt, bottom_br_gt, action_gt, label_gt) in enumerate(data_loader):

        # forward pass
        top_br_pred, bottom_br_pred,feat= model(input_data.cuda())
        easy_dict_label = easy_snippets_mining(top_br_pred, feat) 
        hard_dict_label = hard_snippets_mining(bottom_br_pred, feat)

        c_pair_label = {
            "EA": easy_dict_label[0],
            "EB": easy_dict_label[1],
            "HA": hard_dict_label[0],
            "HB": hard_dict_label[1]
        }
        loss_contrast_label = contrast_loss(c_pair_label)
        loss = spot_loss(top_br_gt,top_br_pred,bottom_br_gt,bottom_br_pred, action_gt,label_gt)
        # update step
        tot_loss = loss[0] + 0
    print("[Epoch {0:03d}] Total-Loss {1:.2f} = T-Loss {2:.2f} + B-Loss {3:.2f}  (val)".format(
    epoch, tot_loss,loss[1],loss[2]))

    state = {'epoch': epoch + 1,
             'state_dict': model.state_dict()}
    torch.save(state, output_path + "/SPOT_checkpoint.pth.tar")
    if tot_loss < best_loss:
        best_loss = tot_loss
        torch.save(state, output_path + "/SPOT_best.pth.tar")

    return best_loss





# semi-supervised training


def train_semi(data_loader, train_loader_unlabel, model, optimizer, epoch):
    global global_step
    model.module.classifier[0].weight.requires_grad = True
    model.module.classifier[0].bias.requires_grad = True
    model.train()
    total_loss = 0 
    top_loss = 0
    bottom_loss = 0
    consistency_loss_all = 0
    consistency_loss_ema_all = 0
    consistency_criterion = softmax_mse_loss  # softmax_kl_loss
    consistency_criterion_top = softmax_kl_loss

    
    temporal_perb = TemporalShift_random(400, 64)   
    order_clip_criterion = nn.CrossEntropyLoss()
    consistency = False
    clip_order = True
    dropout2d = True
    temporal_re = True

    unlabeled_train_iter = iter(train_loader_unlabel)
    temp = 0.6 # temperature
    u_thres = 0.95 
    lambda_3 = 1.0

    # dynamic thresholding

    selected_label = torch.ones((len(unlabeled_train_iter),), dtype=torch.long, ) * -1
    selected_label = selected_label.cuda()
    co = 0
    for n_iter, (input_data, top_br_gt, bottom_br_gt, action_gt, label_gt, input_data_big, input_data_small, f_mask) in enumerate(data_loader):
        # forward pass
        co+=1
        ## labeled data 

        ## weak augmentations -- temporal shift

        input_data_shift = temporal_perb(input_data)

        ## weak augmentations -- temporal flip

        input_data_flip = input_data.flip(2).contiguous()

        ## Temporal Crop

        if dropout2d:
            input_data_shift = F.dropout2d(input_data_shift, 0.2)
            input_data_flip = F.dropout2d(input_data_flip,0.1)
        else:
            input_data_shift = F.dropout(input_data_shift, 0.2)
            input_data_flip = F.dropout2d(input_data_flip,0.1)
        

       ## weak augmentations -- temporal shift and flip on labeled

        top_br_pred, bottom_br_pred, feat = model(input_data.cuda())
        
        loss_shift = spot_loss(top_br_gt,top_br_pred,bottom_br_gt,bottom_br_pred, action_gt,label_gt) # supervised loss - weak augmentation 1 (shift)

        loss_feat_label = Motion_MSEloss(feat,input_data.cuda())
        loss_label =  loss_shift
        ## unlabeled data 
        try:
            input_data_unlabel = unlabeled_train_iter.next()
            gt_top_br = input_data_unlabel[1].cuda()
            gt_action = input_data_unlabel[3].cuda()
            gt_label = input_data_unlabel[4].cuda()
            input_data_unlabel_big = input_data_unlabel[5].cuda()
            input_data_unlabel_small = input_data_unlabel[6].cuda()
            input_data_unlabel = input_data_unlabel[0].cuda()

        except:
            unlabeled_train_iter = iter(train_loader_unlabel)
            input_data_unlabel = unlabeled_train_iter.next()
            gt_top_br = input_data_unlabel[1].cuda()
            gt_action = input_data_unlabel[3].cuda()
            gt_label = input_data_unlabel[4].cuda()
            input_data_unlabel_big = input_data_unlabel[5].cuda()
            input_data_unlabel_small = input_data_unlabel[6].cuda()
            input_data_unlabel = input_data_unlabel[0].cuda()

        ## strong augmentations --
        top_br_pred_unlabel, bottom_br_pred_unlabel, feat_unlabel = model(input_data_unlabel)

        dynmaic_thres = False
        thresh_warmup = True
        if dynmaic_thres:
            pseudo_counter = Counter(selected_label.tolist())
            classwise_acc = torch.zeros((201,)).cuda()
            if max(pseudo_counter.values()) < len(unlabeled_train_iter):  # not all(5w) -1
                if thresh_warmup:
                    for i in range(201):
                        classwise_acc[i] = pseudo_counter[i] / max(pseudo_counter.values())
                else:
                    wo_negative_one = deepcopy(pseudo_counter)
                    if -1 in wo_negative_one.keys():
                        wo_negative_one.pop(-1)
                    for i in range(201):
                        classwise_acc[i] = pseudo_counter[i] / max(wo_negative_one.values())

            
            pseudo_label = torch.softmax(top_br_pred_unlabel, dim=1)
            T=1.01
            p_cutoff=0.7
            use_hard_labels=True
            max_probs, max_idx = torch.max(pseudo_label, dim=1)
            mask = max_probs.ge(p_cutoff * (classwise_acc[max_idx] / (2. - classwise_acc[max_idx]))).float()  # convex
            select = max_probs.ge(p_cutoff).long() # 24 x 100
            if use_hard_labels:
 
                pseudo_label = torch.softmax(top_br_pred_unlabel / T, dim=1)
                max_probs, max_idx = torch.max(pseudo_label,dim=1)
                mask_unlabel_gt = F.one_hot(max_idx,num_classes=201).permute(0,2,1)
                masked_loss = top_ce_loss(max_idx,top_br_pred_unlabel,nm=True)* mask

            pseudo_lb = max_idx.long()
            select_idx = select[select == 1]
            pred_unlabel = torch.argmax(torch.softmax(top_br_pred_unlabel,dim=1),dim=1)

            if pred_unlabel[select==1].nelement() != 0:
                    selected_label[pred_unlabel[select == 1]] = pseudo_lb[select == 1]

            unsup_loss_top = masked_loss.mean() + acsl_loss(top_br_pred_unlabel, mask_unlabel_gt)
            bottom_br_target_unlabel = torch.ge(bottom_br_pred_unlabel, 0.7).float()
            unsup_loss_bottom = bottom_branch_loss(bottom_br_target_unlabel, bottom_br_pred_unlabel)
            loss_unlabel = unsup_loss_bottom + unsup_loss_top
        else:
            # top_br_target_unlabel 

            max_probs, targets_u = torch.max(torch.softmax(top_br_pred_unlabel, dim=1),dim=1)
            mask_unlabel_gt = F.one_hot(targets_u,num_classes=201).permute(0,2,1)
            top_br_target_unlabel = targets_u
            bottom_br_target_unlabel = torch.ge(bottom_br_pred_unlabel, 0.7).float()
            loss_unlabel = spot_loss(top_br_target_unlabel,top_br_pred_unlabel, bottom_br_target_unlabel, bottom_br_pred_unlabel, mask_unlabel_gt, mask_unlabel_gt)
        
        loss_feat_unlabel = Motion_MSEloss(feat_unlabel,input_data_unlabel)
        easy_dict_unlabel = easy_snippets_mining(top_br_pred_unlabel, feat_unlabel) 
        hard_dict_unlabel = hard_snippets_mining(bottom_br_pred_unlabel, feat_unlabel)
        easy_dict_label = easy_snippets_mining(top_br_pred, feat) 
        hard_dict_label = hard_snippets_mining(bottom_br_pred, feat)

        c_pair_unlabel = {
            "EA": easy_dict_unlabel[0],
            "EB": easy_dict_unlabel[1],
            "HA": hard_dict_unlabel[0],
            "HB": hard_dict_unlabel[1]
        }

        c_pair_label = {
            "EA": easy_dict_label[0],
            "EB": easy_dict_label[1],
            "HA": hard_dict_label[0],
            "HB": hard_dict_label[1]
        }

        loss_contrast_unlabel = contrast_loss(c_pair_unlabel)
        loss_contrast_label = contrast_loss(c_pair_label)

        if dynmaic_thres:
            loss_total = loss_label[0] + 10*loss_unlabel
        else:
            loss_total = loss_label + loss_unlabel

        if temporal_re:
            input_recons = F.dropout2d(input_data.permute(0,2,1), 0.2).permute(0,2,1)
        else:
            input_recons = F.dropout2d(input_data, 0.2)

        recons_feature = model(input_recons, recons=True)

        if temporal_re:
            recons_input_student =  F.dropout2d(input_data_unlabel.permute(0,2,1), 0.2).permute(0,2,1)
        else:
            recons_input_student = F.dropout2d(input_data_unlabel, 0.2)

        recons_feature_unlabel_student = model(recons_input_student.cuda(), recons=True)

        loss_recons = 0.1 * (
            Motion_MSEloss(recons_feature, input_data.cuda()) + Motion_MSEloss(recons_feature_unlabel_student,
                                                                        input_data_unlabel))  # 0.0001
       
        if consistency:

            top_one_hot = torch.argmax(top_br_pred_teacher,dim=1)

            top_br_pred_teacher_gt = F.one_hot(top_one_hot,num_classes=201).permute(0,2,1).type(torch.cuda.FloatTensor)

            top_unlabel_onehot = torch.argmax(top_br_pred_unlabel_student,dim=1)

            top_br_pred_unlabel_student_gt = F.one_hot(top_unlabel_onehot,num_classes=201).permute(0,2,1).type(torch.cuda.FloatTensor)

            consistency_weight = get_current_consistency_weight(epoch)

            consistency_loss = consistency_weight * (
                consistency_criterion(bottom_br_pred, bottom_br_pred_teacher)+ consistency_criterion_top(top_br_pred.log_softmax(1) , top_br_pred_teacher_gt))
 
            consistency_loss_ema = consistency_weight * (
                consistency_criterion(bottom_br_pred_unlabel_teacher, bottom_br_pred_unlabel_student) + consistency_criterion_top(top_br_pred_unlabel_teacher.log_softmax(1), top_br_pred_unlabel_student_gt))

            if torch.isnan(consistency_loss_ema):
                consistency_loss_ema = torch.tensor(0.).cuda()

        consistency_loss = torch.tensor(0).cuda()
        consistency_loss_ema = torch.tensor(0).cuda()
        clip_order = False
        if clip_order:
            consistency_loss_ema_flip = torch.tensor(0).cuda()
        clip_order = True
        if clip_order:
            
            input_data_all = torch.cat([input_data.cuda(), input_data_unlabel.cuda()], 0)
            batch_size, C, T = input_data_all.size()
            idx = torch.randperm(batch_size)
            input_data_all_new = input_data_all[idx]
            forw_input = torch.cat(
                [input_data_all_new[:batch_size // 2, :, T // 2:], input_data_all_new[:batch_size // 2, :, :T // 2]], 2)
            back_input = input_data_all_new[batch_size // 2:, :, :]
            input_all = torch.cat([forw_input, back_input], 0)

            label_order = [0] * (batch_size // 2) + [1] * (batch_size - batch_size // 2)
            label_order = torch.tensor(label_order).long().cuda()
            out = model(input_all,clip_order=True)
            loss_clip_order = order_clip_criterion(out, label_order)

        if dynmaic_thres:
            loss_all = loss_total + consistency_loss + loss_feat_unlabel + loss_contrast_label + loss_contrast_unlabel
        else:
            loss_all = loss_total[0] + consistency_loss + loss_feat_unlabel + loss_contrast_label + loss_contrast_unlabel + loss_feat_label

        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()
        global_step += 1
        # update_ema_variables(model, model_ema, 0.999, float(global_step/20))   # //5  //25

        if dynmaic_thres:
            total_loss += loss_total.cpu().detach().numpy()
            top_loss += (loss_label[1]+unsup_loss_top).cpu().detach().numpy()
            bottom_loss += (loss_label[2]+unsup_loss_bottom).cpu().detach().numpy()
            consistency_loss_all += consistency_loss.cpu().detach().numpy()
            consistency_loss_ema_all += consistency_loss_ema.cpu().detach().numpy()
        else:
            total_loss += loss_total[0].cpu().detach().numpy()
            top_loss += loss_total[1].cpu().detach().numpy()
            bottom_loss += loss_total[2].cpu().detach().numpy()
            consistency_loss_all += consistency_loss.cpu().detach().numpy()
            consistency_loss_ema_all += consistency_loss_ema.cpu().detach().numpy()

        if n_iter % 10 == 0:
            print(
                "[Iteration {0:03d}] Total-Loss {1:.2f} = T-Loss {2:.2f} + B-Loss {3:.2f} + C-Loss {4:.2f} + C-EMA-Loss {5:.2f}  (train)".format(
            n_iter, total_loss/(n_iter+1), 
            top_loss/(n_iter+1), 
            bottom_loss/(n_iter+1),
            consistency_loss_all/(n_iter+1),
            consistency_loss_ema_all/(n_iter+1)
                )
            )

    print(
        blue(
           "[Epoch {0:03d}] Total-Loss {1:.2f} = T-Loss {2:.2f} + B-Loss {3:.2f} (train)".format(
            epoch, total_loss/(n_iter+1), 
            top_loss/(n_iter+1), 
            bottom_loss/(n_iter+1)
                ) 
        )
    )



def train_semi_full(data_loader, model, optimizer, epoch):
    model.train()
    for n_iter, (input_data, top_br_gt, bottom_br_gt, action_gt, label_gt) in enumerate(data_loader):
        # forward pass
        top_br_pred, bottom_br_pred = model(input_data.cuda())
        loss = spot_loss(top_br_gt,top_br_pred,bottom_br_gt,bottom_br_pred, action_gt,label_gt)
        # update step
        optimizer.zero_grad()
        loss[0].backward()
        optimizer.step()
    print("[Epoch {0:03d}] Total-Loss {1:.2f} = T-Loss {2:.2f} + B-Loss {3:.2f}  (train)".format(
    epoch, loss[0],loss[1],loss[2]))

# semi-supervised validation
def test_semi(data_loader, model, epoch, best_loss):
    model.eval()
    with torch.no_grad():
      for n_iter, (input_data, top_br_gt, bottom_br_gt, action_gt, label_gt,input_data_big, input_data_small, _) in enumerate(data_loader):

        # forward pass
        top_br_pred, bottom_br_pred, _ = model(input_data.cuda())
        loss = spot_loss(top_br_gt,top_br_pred,bottom_br_gt,bottom_br_pred, action_gt,label_gt)
    print("[Epoch {0:03d}] Total-Loss {1:.2f} = T-Loss {2:.2f} + B-Loss {3:.2f}  (val)".format(
    epoch, loss[0],loss[1],loss[2]))

    state = {'epoch': epoch + 1,
             'state_dict': model.state_dict()}
    torch.save(state, output_path + "/SPOT_checkpoint_semi.pth.tar")
    if loss[0] < best_loss:
        best_loss = loss[0]
        torch.save(state, output_path + "/SPOT_best_semi.pth.tar")

    return best_loss


def getThres(data_loader,model,optimizer):
    top_phat = 0
    bot_phat = 0
    cnt = 0 
    model.module.classifier[0].weight.requires_grad = True
    model.module.classifier[0].bias.requires_grad = True
    model.train()
    print("Starting Warmup")
    for i in range(2):
        for n_iter, (input_data, top_br_gt, bottom_br_gt, action_gt, label_gt, input_data_big, input_data_small) in enumerate(data_loader):
            top_br_pred, bottom_br_pred, feat = model(input_data.cuda())
            loss_shift = spot_loss(top_br_gt,top_br_pred,bottom_br_gt,bottom_br_pred, action_gt,label_gt)
            # top_phat+=loss_shift[1]
            # bot_phat+=loss_shift[2]
            cnt+=1
            optimizer.zero_grad()
            loss_shift[0].backward()
            optimizer.step()
        print("[Warmup Epoch "+str(i)+"] Top Loss"+str(loss_shift[1])+" Bottom Loss"+str(loss_shift[2]))
    print("Ending Warmup")
    top_phat = loss_shift[1]/(2*(n_iter+1))
    bot_phat = loss_shift[2]/(2*(n_iter+1))
    # print(top_phat,bot_phat)
    return top_phat, bot_phat

    












if __name__ == '__main__':

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    model = SPOT()
    # print(model)
    model = torch.nn.DataParallel(model, device_ids=list(range(num_gpu))).cuda()

    for param in model.parameters():
        # print(param)
        param.requires_grad = True
    # print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\nTotal Number of Learnable Paramters (in M) : ",total_params/1000000)
    print('No of Gpus using to Train :  {} '.format(num_gpu))
    print(" Saving all Checkpoints in path : "+ output_path )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                           weight_decay=decay)
    
    train_loader = torch.utils.data.DataLoader(spot_dataset.SPOTDataset(subset="train"),
                                               batch_size=num_batch, shuffle=True,
                                               num_workers=8, pin_memory=False)
    train_loader_pretrain = torch.utils.data.DataLoader(spot_dataset.SPOTDataset(subset="train"),
                                               batch_size=num_batch, shuffle=True,
                                               num_workers=8, pin_memory=False)


    if use_semi and unlabel_percent > 0.:
        train_loader_unlabel = torch.utils.data.DataLoader(spot_dataset.SPOTDatasetUnlabeled(subset="unlabel"),
                                            #    batch_size=num_batch, shuffle=True,
                                               batch_size=min(max(round(num_batch*unlabel_percent/(4*(1.-unlabel_percent)))*4, 4), 24), shuffle=True,drop_last=True,
                                               num_workers=8, pin_memory=False)
    
    
        

    test_loader = torch.utils.data.DataLoader(spot_dataset.SPOTDataset(subset="validation"),
                                              batch_size=num_batch, shuffle=False,
                                              num_workers=8, pin_memory=False)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_train, gamma=gamma_train)
    best_loss = 1e10

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    # print(model.)
    print("Pretraining Start")
    # pretrain(train_loader_pretrain,model,optimizer)
    checkpoint_pre = torch.load(output_path + "/SPOT_pretrain_best.pth.tar")
    model.load_state_dict(checkpoint_pre['state_dict'])
    optimizer.load_state_dict(checkpoint_pre['optimizer'])
    # # top_th,bot_th = getThres(train_loader,model,optimizer)
    print("Pretraining Finished")
    for epoch in range(epoch):
      with autograd.detect_anomaly():

        if use_semi:
            if unlabel_percent == 0.:
                print('use Semi !!! use all label !!!')
                train_semi_full(train_loader, model, optimizer, epoch)
                test_semi(test_loader, model, epoch, best_loss)
            else:
                print('use Semi !!!')
                train_semi(train_loader, train_loader_unlabel, model, optimizer, epoch)
                test_semi(test_loader, model, epoch, best_loss)
        else:
            print('use Fewer label !!!')
            train(train_loader, model, optimizer, epoch)
            test(test_loader, model, epoch, best_loss)

        scheduler.step()
    # writer.flush()
    end.record()
    torch.cuda.synchronize()

    print("Total Time taken for Running "+str(epoch)+" epoch is :"+ str(start.elapsed_time(end)/1000) + " secs")  # milliseconds


