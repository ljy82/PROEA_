import numpy as np
import torch.nn as nn
import torch
from torch.nn.functional import relu
from boxes import Boxes
import os
# from model.loaded_models import BoxSqELLoadedModel
import pickle

class BoxSquaredEL(nn.Module):
    def __init__(self, device, embedding_dim, num_classes, margin=0, 
                 reg_factor=0.05, num_neg=2, batch_size=512):
        super(BoxSquaredEL, self).__init__()

        self.name = 'boxsqel'
        self.device = device
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.margin = margin
        self.reg_factor = reg_factor
        self.num_neg = num_neg
        self.batch_size = batch_size

        self.negative_sampling = True

        self.class_embeds = self.init_embeddings(self.num_classes, embedding_dim * 2)
        self.bumps = self.init_embeddings(self.num_classes, embedding_dim)

    def init_embeddings(self, num_embeddings, dim, min=-1, max=1, normalise=True):
        if num_embeddings == 0:
            return None
        embeddings = nn.Embedding(num_embeddings, dim)
        nn.init.uniform_(embeddings.weight, a=min, b=max)
        if normalise:
            embeddings.weight.data /= torch.linalg.norm(embeddings.weight.data, axis=1).reshape(-1, 1)
        return embeddings

    def get_boxes(self, embedding):
        return Boxes(embedding[:, :self.embedding_dim], torch.abs(embedding[:, self.embedding_dim:]))

    def get_class_boxes(self, nf_data, *indices):
        return (self.get_boxes(self.class_embeds(nf_data[:, i])) for i in indices)


    def inclusion_loss(self, boxes1, boxes2):
        diffs = torch.abs(boxes1.centers - boxes2.centers)
        dist = torch.reshape(torch.linalg.norm(relu(diffs + boxes1.offsets - boxes2.offsets - self.margin), axis=1),
                             [-1, 1])
        return dist

    def disjoint_loss(self, boxes1, boxes2):
        diffs = torch.abs(boxes1.centers - boxes2.centers)
        dist = torch.linalg.norm(relu(-diffs + boxes1.offsets + boxes2.offsets - self.margin), axis=1).reshape([-1, 1])
        return dist

    def neg_loss(self, boxes1, boxes2):
        diffs = torch.abs(boxes1.centers - boxes2.centers)
        dist = torch.reshape(torch.linalg.norm(relu(diffs - boxes1.offsets - boxes2.offsets + self.margin), axis=1),
                             [-1, 1])
        return dist

# subclass
    def nf1_loss(self, nf1_data):
        c_boxes, d_boxes = self.get_class_boxes(nf1_data, 0, 1)
        return self.inclusion_loss(c_boxes, d_boxes)

    def nf2_disjoint_loss(self, disjoint_data):
        c_boxes, d_boxes = self.get_class_boxes(disjoint_data, 0, 1)
        return self.disjoint_loss(c_boxes, d_boxes)

    def get_data_batch(self, train_data, key):
        if len(train_data[key]) <= self.batch_size:
            return train_data[key].to(self.device)
        else:
            rand_index = np.random.choice(len(train_data[key]), size=self.batch_size)
            return train_data[key][rand_index].to(self.device)

    def get_negative_sample_batch(self, train_data, key):
        rand_index = np.random.choice(len(train_data[f'{key}0']), size=self.batch_size)
        neg_data = train_data[f'{key}0'][rand_index]
        for i in range(1, self.num_neg):
            neg_data2 = train_data[f'{key}{i}'][rand_index]
            neg_data = torch.cat([neg_data, neg_data2], dim=0)
        return neg_data.to(self.device)

    def forward(self, train_data):
        loss = 0

        # 处理子类数据
        if 'nf1' in train_data:
            nf1_data = self.get_data_batch(train_data, 'nf1')
            loss += self.nf1_loss(nf1_data).square().mean()
            # print("ok")

        # 处理类冲突数据
        if 'disjoint' in train_data:
            disjoint_data = self.get_data_batch(train_data, 'disjoint')
            loss += self.nf2_disjoint_loss(disjoint_data).square().mean()

        # 正则化损失
        class_reg = self.reg_factor * torch.linalg.norm(self.bumps.weight, dim=1).reshape(-1, 1).mean()
        loss += class_reg
        return loss

    # def to_loaded_model(self):
    #     model = BoxSqELLoadedModel()
    #     model.embedding_size = self.embedding_dim
    #     model.class_embeds = self.class_embeds.weight.detach()
    #     model.bumps = self.bumps.weight.detach()
    #     return model

    def save(self, folder, class_to_index, best=False):
        if not os.path.exists(folder):
            os.makedirs(folder)
        suffix = '_best' if best else ''

        # 创建一个字典，键为本体名，值为训练好的嵌入向量
        embeddings_dict = {
            class_name: self.class_embeds.weight[index].detach().cpu().numpy()
            for class_name, index in class_to_index.items()
        }

        # 保存为 pkl 文件
        with open(f'{folder}/ja_class_embeds{suffix}.pkl', 'wb') as f:
            pickle.dump(embeddings_dict, f)