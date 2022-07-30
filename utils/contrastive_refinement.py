import numpy as np
import torch
import torch.nn.functional as F 
import torch.nn as nn
from scipy import ndimage
import kornia
from torchvision.utils import save_image

class SniCoLoss(nn.Module):
    def __init__(self):
        super(SniCoLoss, self).__init__()
        self.ce_criterion = nn.CrossEntropyLoss()

    def NCE(self, q, k, neg,  T=0.07):
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        neg = neg.permute(0,2,1)
        neg = nn.functional.normalize(neg, dim=1)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,nck->nk', [q, neg])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        loss = self.ce_criterion(logits, labels)
        # print(loss)



        return loss

    def forward(self, contrast_pairs):

        HA_refinement = self.NCE(
            torch.mean(contrast_pairs['HA'], 1), 
            torch.mean(contrast_pairs['EA'], 1), 
            contrast_pairs['EB'],
        )

        HB_refinement = self.NCE(
            torch.mean(contrast_pairs['HB'],1), 
            torch.mean(contrast_pairs['EB'], 1), 
            contrast_pairs['EA']
        )

        loss = HA_refinement + HB_refinement
        return loss


def select_topk_embeddings_bottom(scores, embeddings, k):

    _, idx_DESC = scores.sort(descending=True, dim=1)
    idx_topk = idx_DESC[:,:k, :]
    selected_embeddings = torch.gather(embeddings, 1, idx_topk)
    return selected_embeddings

def select_topk_embeddings_top(scores, embeddings, k):

    _, idx_DESC = scores.sort(descending=True, dim=1)
    idx_topk = idx_DESC[:, :k]
    idx_topk = idx_topk.unsqueeze(2).expand([-1, -1, embeddings.shape[2]])
    selected_embeddings = torch.gather(embeddings, 1, idx_topk)
    return selected_embeddings


def easy_snippets_mining(actionness, embeddings, k_easy=10):

    select_idx = torch.ones((actionness.size(0),actionness.size(2))).cuda()
    select_idx = F.dropout(select_idx,0.3)
    soft_action = actionness
    fg_action_idx = torch.argmax(soft_action[:,:200,:],dim=1)
    fg_action_idx_mode, _ = torch.mode(fg_action_idx,dim=1)
    fg_action = torch.stack([actionness[i,fg_action_idx_mode[i],:] for i in range(actionness.size(0))],dim=0)
    bg_action = actionness[:,200,:] 
    fg_action_drop = fg_action * select_idx
    bg_action_drop = bg_action * select_idx
    easy_act = select_topk_embeddings_top(fg_action_drop, embeddings, k_easy)
    easy_bkg = select_topk_embeddings_top(bg_action_drop, embeddings, k_easy)

    return easy_act, easy_bkg

def hard_snippets_mining(actionness, embeddings, k_hard=20):
    m = 4
    M = 8
    aness_median = torch.median(actionness)
    aness_bin = torch.ge(actionness,aness_median).float()

    kernel_M = torch.ones((M,M),dtype=torch.float64)
    erosion_M = kornia.morphology.erosion(aness_bin.unsqueeze(1).type(torch.cuda.DoubleTensor), kernel_M.type(torch.cuda.DoubleTensor),border_type="constant")
    kernel_m = torch.ones((m,m),dtype=torch.float64)
    erosion_m = kornia.morphology.erosion(aness_bin.unsqueeze(1).type(torch.cuda.DoubleTensor), kernel_m.type(torch.cuda.DoubleTensor),border_type="constant")
    idx_region_inner = erosion_m - erosion_M
    aness_region_inner = actionness.unsqueeze(1) * idx_region_inner
    aness_region_inner = aness_region_inner.squeeze(1)
    hard_act = select_topk_embeddings_bottom(aness_region_inner, embeddings, k_hard)


    kernel_M = torch.ones((M,M),dtype=torch.float64)
    dilation_M = kornia.morphology.dilation(aness_bin.unsqueeze(1).type(torch.cuda.DoubleTensor), kernel_M.type(torch.cuda.DoubleTensor),border_type="constant")
    kernel_m = torch.ones((m,m),dtype=torch.float64)
    dilation_m = kornia.morphology.dilation(aness_bin.unsqueeze(1).type(torch.cuda.DoubleTensor), kernel_m.type(torch.cuda.DoubleTensor),border_type="constant")
    idx_region_outer = dilation_M - dilation_m
    aness_region_outer = actionness.unsqueeze(1) * idx_region_outer
    aness_region_outer = aness_region_outer.squeeze(1)
    hard_bkg = select_topk_embeddings_bottom(aness_region_outer, embeddings, k_hard)

    return hard_act, hard_bkg




