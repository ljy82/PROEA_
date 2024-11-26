import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from tqdm import tqdm
import random
import os
import scipy.spatial.distance
import torch.nn.functional as F


class icl_loss(nn.Module):
    def __init__(self, tau=0.05, ab_weight=0.5, n_view=2, intra_weight=1.0, inversion=False):
        super(icl_loss, self).__init__()
        self.tau = tau
        self.weight = ab_weight
        self.n_view = n_view
        self.intra_weight = intra_weight
        self.inversion = inversion

    def softXEnt(self, target, logits):
        logprobs = F.log_softmax(logits, dim=1)
        loss = -(target * logprobs).sum() / logits.shape[0]
        return loss

    def forward(self, emb, train_links, emb2=None, norm=True):
        if norm:
            emb = F.normalize(emb, dim=1)
            if emb2 is not None:
                emb2 = F.normalize(emb2, dim=1)

        train_links = torch.tensor(train_links).to(emb.device)
        num_ent = emb.shape[0]
        tt=train_links[:, 0]
        zis = emb[tt]
        if emb2 is not None:
            zjs = emb2[train_links[:, 1]]
        else:
            zjs = emb[train_links[:, 1]]

        temperature = self.tau
        alpha = self.weight
        n_view = self.n_view

        LARGE_NUM = 1e9

        hidden1, hidden2 = zis, zjs
        batch_size = hidden1.shape[0]

        hidden1_large = hidden1
        hidden2_large = hidden2
        num_classes = batch_size * n_view

        labels = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=num_classes).float().to(emb.device)
        masks = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=batch_size).float().to(emb.device)

        # labels = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=num_classes).float()

        # masks = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=batch_size)
        masks = masks.float()
        logits_aa = torch.matmul(hidden1, torch.transpose(hidden1_large, 0, 1)) / temperature
        logits_aa = logits_aa - masks * LARGE_NUM
        logits_bb = torch.matmul(hidden2, torch.transpose(hidden2_large, 0, 1)) / temperature
        logits_bb = logits_bb - masks * LARGE_NUM
        logits_ab = torch.matmul(hidden1, torch.transpose(hidden2_large, 0, 1)) / temperature
        logits_ba = torch.matmul(hidden2, torch.transpose(hidden1_large, 0, 1)) / temperature

        if self.inversion:
            logits_a = torch.cat([logits_ab, logits_bb], dim=1)
            logits_b = torch.cat([logits_ba, logits_aa], dim=1)
        else:
            logits_a = torch.cat([logits_ab, logits_aa], dim=1)
            logits_b = torch.cat([logits_ba, logits_bb], dim=1)

        loss_a = self.softXEnt(labels, logits_a)
        loss_b = self.softXEnt(labels, logits_b)

        return alpha * loss_a + (1 - alpha) * loss_b

class ial_loss(nn.Module):
    def __init__(self, tau=0.05, ab_weight=0.5, zoom=0.1, n_view=2, inversion=False, reduction="mean", detach=False):
        super(ial_loss, self).__init__()
        self.tau = tau
        self.weight = ab_weight
        self.zoom = zoom
        self.n_view = n_view
        self.inversion = inversion
        self.reduction = reduction
        self.detach = detach

    def forward(self, src_emb, tar_emb, train_links, norm=True):
        if norm:
            src_emb = F.normalize(src_emb, dim=1)
            tar_emb = F.normalize(tar_emb, dim=1)

        train_links = torch.tensor(train_links).to(src_emb.device)

        src_zis = src_emb[train_links[:, 0]]
        src_zjs = src_emb[train_links[:, 1]]
        tar_zis = tar_emb[train_links[:, 0]]
        tar_zjs = tar_emb[train_links[:, 1]]

        temperature = self.tau
        alpha = self.weight

        assert src_zis.shape[0] == tar_zjs.shape[0]
        batch_size = src_zis.shape[0]
        LARGE_NUM = 1e9
        masks = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=batch_size).float().to(src_emb.device)
        # masks = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=batch_size)
        masks = masks.float()
        p_ab = torch.matmul(src_zis, torch.transpose(src_zjs, 0, 1)) / temperature
        p_ba = torch.matmul(src_zjs, torch.transpose(src_zis, 0, 1)) / temperature
        q_ab = torch.matmul(tar_zis, torch.transpose(tar_zjs, 0, 1)) / temperature
        q_ba = torch.matmul(tar_zjs, torch.transpose(tar_zis, 0, 1)) / temperature
        p_aa = torch.matmul(src_zis, torch.transpose(src_zis, 0, 1)) / temperature
        p_bb = torch.matmul(src_zjs, torch.transpose(src_zjs, 0, 1)) / temperature
        q_aa = torch.matmul(tar_zis, torch.transpose(tar_zis, 0, 1)) / temperature
        q_bb = torch.matmul(tar_zjs, torch.transpose(tar_zjs, 0, 1)) / temperature
        p_aa = p_aa - masks * LARGE_NUM
        p_bb = p_bb - masks * LARGE_NUM
        q_aa = q_aa - masks * LARGE_NUM
        q_bb = q_bb - masks * LARGE_NUM

        if self.inversion:
            p_ab = torch.cat([p_ab, p_bb], dim=1)
            p_ba = torch.cat([p_ba, p_aa], dim=1)
            q_ab = torch.cat([q_ab, q_bb], dim=1)
            q_ba = torch.cat([q_ba, q_aa], dim=1)
        else:
            p_ab = torch.cat([p_ab, p_aa], dim=1)
            p_ba = torch.cat([p_ba, p_bb], dim=1)
            q_ab = torch.cat([q_ab, q_aa], dim=1)
            q_ba = torch.cat([q_ba, q_bb], dim=1)

        loss_a = F.kl_div(F.log_softmax(p_ab, dim=1), F.softmax(q_ab.detach(), dim=1), reduction="none")
        loss_b = F.kl_div(F.log_softmax(p_ba, dim=1), F.softmax(q_ba.detach(), dim=1), reduction="none")

        if self.reduction == "mean":
            loss_a = loss_a.mean()
            loss_b = loss_b.mean()
        elif self.reduction == "sum":
            loss_a = loss_a.sum()
            loss_b = loss_b.sum()

        return self.zoom * (alpha * loss_a + (1 - alpha) * loss_b)