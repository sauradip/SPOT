# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import yaml

with open("./config/anet.yaml", 'r', encoding='utf-8') as f:
        tmp = f.read()
        config = yaml.load(tmp, Loader=yaml.FullLoader)


ce = nn.CrossEntropyLoss()

lambda_1 = config['loss']['lambda_1']
lambda_2 = config['loss']['lambda_2']

activity_dict = {'Beer pong': 0, 'Kneeling': 1, 'Tumbling': 2, 'Sharpening knives': 3, 'Playing water polo': 4, 'Scuba diving': 5, 'Arm wrestling': 6, 'Archery': 7, 'Shaving': 8, 'Playing bagpipes': 9, 'Riding bumper cars': 10, 'Surfing': 11, 'Hopscotch': 12, 'Gargling mouthwash': 13, 'Playing violin': 14, 'Plastering': 15, 'Changing car wheel': 16, 'Horseback riding': 17, 'Playing congas': 18, 'Doing a powerbomb': 19, 'Walking the dog': 20, 'Using the pommel horse': 21, 'Rafting': 22, 'Hurling': 23, 'Removing curlers': 24, 'Windsurfing': 25, 'Playing drums': 26, 'Tug of war': 27, 'Playing badminton': 28, 'Getting a piercing': 29, 'Camel ride': 30, 'Sailing': 31, 'Wrapping presents': 32, 'Hand washing clothes': 33, 'Braiding hair': 34, 'Using the monkey bar': 35, 'Longboarding': 36, 'Doing motocross': 37, 'Cleaning shoes': 38, 'Vacuuming floor': 39, 'Blow-drying hair': 40, 'Doing fencing': 41, 'Playing harmonica': 42, 'Playing blackjack': 43, 'Discus throw': 44, 'Playing flauta': 45, 'Ice fishing': 46, 'Spread mulch': 47, 'Mowing the lawn': 48, 'Capoeira': 49, 'Preparing salad': 50, 'Beach soccer': 51, 'BMX': 52, 'Playing kickball': 53, 'Shoveling snow': 54, 'Swimming': 55, 'Cheerleading': 56, 'Removing ice from car': 57, 'Calf roping': 58, 'Breakdancing': 59, 'Mooping floor': 60, 'Powerbocking': 61, 'Kite flying': 62, 'Running a marathon': 63, 'Swinging at the playground': 64, 'Shaving legs': 65, 'Starting a campfire': 66, 'River tubing': 67, 'Zumba': 68, 'Putting on makeup': 69, 'Raking leaves': 70, 'Canoeing': 71, 'High jump': 72, 'Futsal': 73, 'Hitting a pinata': 74, 'Wakeboarding': 75, 'Playing lacrosse': 76, 'Grooming dog': 77, 'Cricket': 78, 'Getting a tattoo': 79, 'Playing saxophone': 80, 'Long jump': 81, 'Paintball': 82, 'Tango': 83, 'Throwing darts': 84, 'Ping-pong': 85, 'Tennis serve with ball bouncing': 86, 'Triple jump': 87, 'Peeling potatoes': 88, 'Doing step aerobics': 89, 'Building sandcastles': 90, 'Elliptical trainer': 91, 'Baking cookies': 92, 'Rock-paper-scissors': 93, 'Playing piano': 94, 'Croquet': 95, 'Playing squash': 96, 'Playing ten pins': 97, 'Using parallel bars': 98, 'Snowboarding': 99, 'Preparing pasta': 100, 'Trimming branches or hedges': 101, 'Playing guitarra': 102, 'Cleaning windows': 103, 'Playing field hockey': 104, 'Skateboarding': 105, 'Rollerblading': 106, 'Polishing shoes': 107, 'Fun sliding down': 108, 'Smoking a cigarette': 109, 'Spinning': 110, 'Disc dog': 111, 'Installing carpet': 112, 'Using the balance beam': 113, 'Drum corps': 114, 'Playing polo': 115, 'Doing karate': 116, 'Hammer throw': 117, 'Baton twirling': 118, 'Tai chi': 119, 'Kayaking': 120, 'Grooming horse': 121, 'Washing face': 122, 'Bungee jumping': 123, 'Clipping cat claws': 124, 'Putting in contact lenses': 125, 'Playing ice hockey': 126, 'Brushing hair': 127, 'Welding': 128, 'Mixing drinks': 129, 'Smoking hookah': 130, 'Having an ice cream': 131, 'Chopping wood': 132, 'Plataform diving': 133, 'Dodgeball': 134, 'Clean and jerk': 135, 'Snow tubing': 136, 'Decorating the Christmas tree': 137, 'Rope skipping': 138, 'Hand car wash': 139, 'Doing kickboxing': 140, 'Fixing the roof': 141, 'Playing pool': 142, 'Assembling bicycle': 143, 'Making a sandwich': 144, 'Shuffleboard': 145, 'Curling': 146, 'Brushing teeth': 147, 'Fixing bicycle': 148, 'Javelin throw': 149, 'Pole vault': 150, 'Playing accordion': 151, 'Bathing dog': 152, 'Washing dishes': 153, 'Skiing': 154, 'Playing racquetball': 155, 'Shot put': 156, 'Drinking coffee': 157, 'Hanging wallpaper': 158, 'Layup drill in basketball': 159, 'Springboard diving': 160, 'Volleyball': 161, 'Ballet': 162, 'Rock climbing': 163, 'Ironing clothes': 164, 'Snatch': 165, 'Drinking beer': 166, 'Roof shingle removal': 167, 'Blowing leaves': 168, 'Cumbia': 169, 'Hula hoop': 170, 'Waterskiing': 171, 'Carving jack-o-lanterns': 172, 'Cutting the grass': 173, 'Sumo': 174, 'Making a cake': 175, 'Painting fence': 176, 'Doing crunches': 177, 'Making a lemonade': 178, 'Applying sunscreen': 179, 'Painting furniture': 180, 'Washing hands': 181, 'Painting': 182, 'Putting on shoes': 183, 'Knitting': 184, 'Doing nails': 185, 'Getting a haircut': 186, 'Using the rowing machine': 187, 'Polishing forniture': 188, 'Using uneven bars': 189, 'Playing beach volleyball': 190, 'Cleaning sink': 191, 'Slacklining': 192, 'Bullfighting': 193, 'Table soccer': 194, 'Waxing skis': 195, 'Playing rubik cube': 196, 'Belly dance': 197, 'Making an omelette': 198, 'Laying tile': 199}

easy_class = ['Windsurfing','Using the pommel horse','Using the monkey bar','Tango','Table soccer','Swinging at the playground','Surfing','Springboard diving','Snowboarding','Snow tubing','Slacklining','Skiing','Shoveling snow','Sailing','Rock climbing','River tubing','Riding bumper cars','Raking leaves','Rafting','Putting in contact lenses','Preparing pasta','Pole vault','Volleyball','Playing pool','Playing field hockey','Playing blackjack','Playing beach volleyball','Playing accordion','Plataform diving','Plastering','Mixing drinks','Making an omelette','Longboarding','Hurling','Horseback riding','Hitting a pinata','Hanging wallpaper','Hammer throw','Grooming dog','Getting a piercing','Elliptical trainer','Drum corps','Doing motocross','Decorating the Christmas tree','Curling','Croquet','Cleaning sink','Clean and jerk','Carving jack-o-lanterns','Camel ride']
hard_class = ['Drinking coffee','Doing a powerbomb','Polishing forniture','Putting on shoes','Removing curlers','Rock-paper-scissors','Gargling mouthwash','Having an ice cream','Polishing shoes','Smoking a cigarette','Applying sunscreen','Drinking beer','Washing face','Doing nails','Brushing hair','Playing harmonica','Painting furniture','Peeling potatoes','Cumbia','Cleaning shoes','Doing karate','Chopping wood','Hand washing clothes','Painting','Shaving legs','Using parallel bars','Baking cookies','Playing drums','Bathing dog','Kneeling','Hopscotch','Playing kickball','Doing crunches','Playing saxophone','Roof shingle removal','Shot put','Playing flauta','Swimming','Preparing salad','Washing dishes','Getting a tattoo','Getting a haircut','Fixing bicycle','Playing guitarra','Tai chi','Washing hands','Vacuuming floor','Waxing skis','Doing step aerobics','Putting on makeup']
easy_and_hard = easy_class + hard_class
common_class = set(easy_and_hard) ^ set(activity_dict.keys())
common_class_list = list(common_class)

rare_id = [activity_dict[hard_class[i]] for i in range(len(hard_class))]
common_id = [activity_dict[easy_class[i]] for i in range(len(easy_class))]
freq_id = [activity_dict[common_class_list[i]] for i in range(len(common_class_list))]

freq_dict = {'rare': rare_id, 'common': freq_id, 'freq': common_id}

class ACSL(nn.Module):

    def __init__(self, score_thr=0.3, loss_weight=1.0):

        super(ACSL, self).__init__()

        self.score_thr = score_thr
        assert self.score_thr > 0 and self.score_thr < 1
        self.loss_weight = loss_weight

        self.freq_group = freq_dict

    def forward(self, cls_logits_, labels_, weight=None, avg_factor=None, reduction_override=None, **kwargs):

        device = cls_logits_.device
        
        self.n_i, self.n_c, _ = cls_logits_.size()
        cls_loss = 0
        for snip_id in range(100):
            cls_logits = cls_logits_[:,:,snip_id]  # batch x class
            labels = labels_[:,:,snip_id]
            
            # expand the labels to all their parent nodes
            target = cls_logits.new_zeros(self.n_i, self.n_c)
         
            labels = torch.argmax(labels,dim=1)
            unique_label = torch.unique(labels)
            
            # print(unique_label)
            with torch.no_grad():
                sigmoid_cls_logits = torch.sigmoid(cls_logits)
            # for each sample, if its score on unrealated class hight than score_thr, their gradient should not be ignored
            # this is also applied to negative samples
            high_score_inds = torch.nonzero(sigmoid_cls_logits>=self.score_thr)
            weight_mask = torch.sparse_coo_tensor(high_score_inds.t(), cls_logits.new_ones(high_score_inds.shape[0]), size=(self.n_i, self.n_c), device=device).to_dense()
            # print(weight_mask.size())
            for cls in unique_label:
                cls = cls.item()
                # print(cls)
                cls_inds = torch.nonzero(labels == cls).squeeze(1)
                # print(cls_inds.size())
                if cls == 200:
                    # construct target vector for background samples
                    target[cls_inds, 200] = 1
                    # for bg, set the weight of all classes to 1
                    weight_mask[cls_inds] = 0

                    cls_inds_cpu = cls_inds.cpu()

                    # Solve the rare categories, random choost 1/3 bg samples to suppress rare categories
                    rare_cats = self.freq_group['rare']
                    rare_cats = torch.tensor(rare_cats, device=cls_logits.device)
                    choose_bg_num = int(len(cls_inds) * 0.01)
                    choose_bg_inds = torch.tensor(np.random.choice(cls_inds_cpu, size=(choose_bg_num), replace=False), device=device)

                    tmp_weight_mask = weight_mask[choose_bg_inds]
                    tmp_weight_mask[:, rare_cats] = 1

                    weight_mask[choose_bg_inds] = tmp_weight_mask

                    # Solve the common categories, random choost 2/3 bg samples to suppress rare categories
                    common_cats = self.freq_group['common']
                    common_cats = torch.tensor(common_cats, device=cls_logits.device)
                    choose_bg_num = int(len(cls_inds) * 0.1)
                    choose_bg_inds = torch.tensor(np.random.choice(cls_inds_cpu, size=(choose_bg_num), replace=False), device=device)

                    tmp_weight_mask = weight_mask[choose_bg_inds]
                    tmp_weight_mask[:, common_cats] = 1

                    weight_mask[choose_bg_inds] = tmp_weight_mask
                    
                    # Solve the frequent categories, random choost all bg samples to suppress rare categories
                    freq_cats = self.freq_group['freq']
                    freq_cats = torch.tensor(freq_cats, device=cls_logits.device)
                    choose_bg_num = int(len(cls_inds) * 1.0)
                    choose_bg_inds = torch.tensor(np.random.choice(cls_inds_cpu, size=(choose_bg_num), replace=False), device=device)

                    tmp_weight_mask = weight_mask[choose_bg_inds]
                    tmp_weight_mask[:, freq_cats] = 1

                    weight_mask[choose_bg_inds] = tmp_weight_mask

                    # Set the weight for bg to 1
                    weight_mask[cls_inds, 200] = 1
                    
                else:
                    # construct target vector for foreground samples
                    cur_labels = [cls]
                    cur_labels = torch.tensor(cur_labels, device=cls_logits.device)
                    tmp_label_vec = cls_logits.new_zeros(self.n_c)
                    tmp_label_vec[cur_labels] = 1
                    tmp_label_vec = tmp_label_vec.expand(cls_inds.numel(), self.n_c)
                    target[cls_inds] = tmp_label_vec
                    # construct weight mask for fg samples
                    tmp_weight_mask_vec = weight_mask[cls_inds]
                    # set the weight for ground truth category
                    tmp_weight_mask_vec[:, cur_labels] = 1

                    weight_mask[cls_inds] = tmp_weight_mask_vec

            cls_loss+= F.binary_cross_entropy_with_logits(cls_logits, target.float(), reduction='none')

        return torch.sum(weight_mask * cls_loss) / (self.n_i*100)





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


class BinaryDiceLoss(nn.Module):
   
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

dice = BinaryDiceLoss()
acsl = ACSL()


def top_ce_loss(gt_cls, pred_cls, nm=False):

    ce_loss = F.cross_entropy(pred_cls,gt_cls)
    pt = torch.exp(-ce_loss)
    if nm:
        focal_loss = ((1 - pt) **2 * ce_loss)
    else:
        focal_loss = ((1 - pt) **2 * ce_loss).mean()
    loss = focal_loss.mean() 

    return loss


def bottom_branch_loss(gt_action, pred_action):

    pmask = (gt_action == 1).float()
    nmask = (gt_action == 0).float()
    nmask = nmask 
    num_positive = 10 + torch.sum(pmask) # in case of nan
    num_entries = 10 + num_positive + torch.sum(nmask)
    ratio = num_entries / num_positive
    coef_0 = 0.5 * ratio / (ratio - 1)
    coef_1 = 0.5 * ratio
    epsilon = 0.000001
    loss_pos = coef_1 * torch.log(pred_action + epsilon) * pmask
    loss_neg = coef_0 * torch.log(1.0 - pred_action + epsilon) * nmask
    w_bce_loss = -1 * torch.sum(loss_pos + loss_neg) / num_entries
    BCE_loss = F.binary_cross_entropy(pred_action,gt_action,reduce=False)
    pt = torch.exp(-BCE_loss)
    # F_loss = 0.4*loss2 + 0.6*dice(pred_action,gt_action)
    F_loss = lambda_2*w_bce_loss + (1 - lambda_2)*dice(pred_action,gt_action)
    
    return F_loss

def top_branch_loss(gt_cls, pred_cls, mask_gt):

    loss = lambda_1*top_ce_loss(gt_cls.cuda(), pred_cls) 
    return loss

def spot_loss(gt_cls, pred_cls ,gt_action , pred_action, mask_gt , label_gt, pretrain=False):
    

    if pretrain:
        bottom_loss = bottom_branch_loss(gt_action.cuda(), pred_action)
        return bottom_loss
    else:
        top_loss = top_branch_loss(gt_cls, pred_cls, mask_gt)
        bottom_loss = bottom_branch_loss(gt_action.cuda(), pred_action) 
        tot_loss = top_loss + bottom_loss
        return tot_loss, top_loss, bottom_loss


# def dynamic_thres()

def spot_loss_bot(gt_cls, pred_cls ,gt_action , pred_action, mask_gt , label_gt):

    
    top_loss = top_branch_loss(gt_cls, pred_cls, mask_gt)
    bottom_loss = bottom_branch_loss(gt_action.cuda(), pred_action) 

    tot_loss = bottom_loss 
    top_loss = 0

    return tot_loss, top_loss, bottom_loss


def ce_loss_thresh(logits, targets, use_hard_labels=True, reduction='none'):
    """
    wrapper for cross entropy loss in pytorch.
    
    Args
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
    """
    if use_hard_labels:
        log_pred = F.log_softmax(logits, dim=1)
        return F.nll_loss(log_pred, targets, reduction=reduction)
        # return F.cross_entropy(logits, targets, reduction=reduction) this is unstable
    else:
        assert logits.shape == targets.shape
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        return nll_loss
