import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, AutoConfig
from torch.optim.lr_scheduler import LRScheduler
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from typing import Tuple
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from src.eval import biological_distance, BiologicalTree

class WarmupStepLR(LRScheduler):
    def __init__(self, optimizer, warmup_steps, step_size, gamma=0.1, base_lr=1e-3, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.step_size = step_size
        self.gamma = gamma
        self.base_lr = base_lr
        super(WarmupStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Warmup phase
            return [self.base_lr * (self.last_epoch + 1) / self.warmup_steps for _ in self.base_lrs]
        else:
            # StepLR phase
            steps_since_warmup = self.last_epoch - self.warmup_steps
            factor = self.gamma ** (steps_since_warmup // self.step_size)
            return [base_lr * factor for base_lr in self.base_lrs]

class MLP_ProjModel(torch.nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)

class ConcatEmb_ProjModel(torch.nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class classifier(torch.nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)

class hierarchical_classifier(torch.nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim_list, dropout=0.):
        super(hierarchical_classifier, self).__init__()
        self.h_modules = nn.ModuleList([
            classifier(in_dim, hidden_dim, out_dim, dropout)
            for out_dim in out_dim_list
        ])

    def forward(self, x):
        out = []
        for module in self.h_modules:
            out.append(module(x))
        return out

class AttentionLayer(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads=4, dropout=0.0):
        super(AttentionLayer, self).__init__()
        self.query_proj = nn.Linear(input_dim, embed_dim)
        self.key_proj = nn.Linear(input_dim, embed_dim)
        self.value_proj = nn.Linear(input_dim, embed_dim)

        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x, kv, mask=None):
        # (B, N), bool
        # 2. Projection
        q = self.query_proj(x)
        k = self.key_proj(kv)
        v = self.value_proj(kv)
        # 3. Attention with masking
        attn_output, attn_weights = self.attn(q, k, v, key_padding_mask=mask)
        # 4. Residual + MLP
        x = self.norm1(attn_output + q)
        mlp_output = self.mlp(x)
        x = self.norm2(mlp_output + x)
        return x, attn_weights


class MultiLayerAttentionModel(nn.Module):
    def __init__(self, query_dim, embed_dim, num_heads=4, num_blocks=3, dropout=0.0):
        super(MultiLayerAttentionModel, self).__init__()

        self.layers = nn.ModuleList([
            AttentionLayer(query_dim if i == 0 else embed_dim, embed_dim, num_heads, dropout)
            for i in range(num_blocks)
        ])

    def forward(self, q, kv, mask=None):
        for layer in self.layers:
            q, attn_weights = layer(q, kv, mask)
        return q

class FathomnetModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)

        # assert self.hparams.temperature > 0.0, "The temperature must be a positive float!"
        self.img_vit_region_encoders = nn.ModuleDict()
        for scales in self.hparams.img_encoder_size:
            for crop_scale in self.hparams.env_img_crop_scale_list:
                self.img_vit_region_encoders[str(scales[0])+'_'+str(crop_scale)] = AutoModel.from_pretrained(self.hparams.img_vit_encoder_path)

        self.obj_vit_region_encoder  = AutoModel.from_pretrained(self.hparams.obj_vit_encoder_path)

        self.classifier = classifier(in_dim=self.hparams.feature_dim,
                                               hidden_dim=self.hparams.feature_dim,
                                               out_dim=self.hparams.nclass, dropout=0.3)

        n_concat = 1
        if self.hparams.intra_env_attn:
            self.intra_env_attn_module = nn.ModuleDict()
            self.obj_proj_module = nn.ModuleDict()
            for scales in self.hparams.img_encoder_size:
                for crop_scale in self.hparams.env_img_crop_scale_list:
                    n_concat += 1
                    _name = str(scales[0]) + '_' + str(crop_scale)
                    self.intra_env_attn_module[_name] = MultiLayerAttentionModel(query_dim=self.hparams.feature_dim,
                                                                               embed_dim=self.hparams.feature_dim,
                                                                               num_heads=8,
                                                                               num_blocks=4,
                                                                               dropout=0.3).cuda()

                    self.obj_proj_module[_name] = MLP_ProjModel(in_dim=self.hparams.feature_dim,
                                                          hidden_dim=self.hparams.feature_dim,
                                                          out_dim=self.hparams.feature_dim, dropout=0.3)



        if self.hparams.hierarchical_loss:
            self.hierarchical_target = pd.read_csv(self.hparams.hierarchical_label_path, index_col=0)
            with open(self.hparams.hierachical_labelencoder_path, 'rb') as f:
                self.hierachical_labelencoder = pickle.load(f)

            self.rank = self.hparams.hierarchical_node_rank
            self.hierarchical_classifier = hierarchical_classifier(in_dim=self.hparams.feature_dim,
                                                 hidden_dim=self.hparams.feature_dim,
                                                 out_dim_list=self.hparams.hierarchical_node_cnt, dropout=0.3)

        self.concat_proj = ConcatEmb_ProjModel(in_dim=self.hparams.feature_dim*n_concat,
                                               hidden_dim=self.hparams.feature_dim,
                                               out_dim=self.hparams.feature_dim, dropout=0.3)


        # 거리 행렬 생성 (C x C)
        category_names = list(self.hparams.category_name2id.keys())
        self.label_distance = pd.read_csv(self.hparams.categories_path, index_col=0)
        distance_matrix = self.label_distance.loc[category_names, category_names].values  # np.ndarray
        self.label_distance_tensor = torch.tensor(distance_matrix, dtype=torch.float32)  # to torch.Tensor

        self.crossentropy = nn.CrossEntropyLoss()


    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)

        #
        lr_scheduler = WarmupStepLR(optimizer=optimizer,
                                    warmup_steps=self.hparams.scheduler_t_up,
                                    step_size=self.hparams.scheduler_step_size,
                                    gamma=self.hparams.scheduler_gamma,
                                    base_lr=self.hparams.scheduler_eta_max)

        return [optimizer], [lr_scheduler]

    def crossentropy_loss(self, logits, labels, type, mode):
        """
        Args:
            logits: Tensor of shape [batch_size, num_classes]
            labels: Tensor of shape [batch_size] with class indices (0 to num_classes - 1)

        Returns:
            loss: Cross-entropy loss (scalar tensor)
            error: Classification error rate (float, 0.0 to 1.0)
        """
        loss = self.crossentropy(logits, labels)
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            correct = (preds == labels).sum().item()
            total = labels.size(0)
            acc = (correct / total)
        # Logging
        if not self.hparams.disable_logger:
            self.log(f"{mode}_{type}_celoss", loss)
            self.log(f"{mode}_{type}_acc", acc)
        return loss

    def sub_h_crossentropy_loss(self, hierarchical_logits_list, target, mode):
        h_target = self.hierarchical_target.loc[target.cpu().numpy(), :]
        loss_list = []
        for i in range(len(self.rank)):
            level = self.rank[i]
            _level_logit = hierarchical_logits_list[i].squeeze()
            _level_target = torch.tensor(h_target.loc[:, level].values).to(_level_logit.device)
            _loss = self.crossentropy(_level_logit, _level_target)
            loss_list.append(_loss)
            if not self.hparams.disable_logger:
                with torch.no_grad():
                    preds = torch.argmax(_level_logit, dim=1)
                    correct = (preds == _level_target).sum().item()
                    total = _level_target.size(0)
                    acc = (correct / total)
                self.log(f"{mode}_{level}_acc", acc)
        return loss_list

    def hierarchical_distance(self, logits, target, mode):
        # 확률화
        probs = torch.softmax(logits, dim=1)  # (B, C)
        # target index → distance vector (C,)
        target_idx = [self.hparams.category_name2id[self.hparams.category_id2name[int(i)]] for i in target.cpu()]
        target_idx = torch.tensor(target_idx)
        # 거리 행렬 중에서 target column만 추출 (B, C)
        distance_targets = self.label_distance_tensor[target_idx, : ].to(logits.device) # (B, C)
        # soft expectation: 각 샘플에 대해 확률 * 거리
        loss_vec = (probs * distance_targets).sum(dim=1)  # (B,)
        mean_h_score = torch.mean(loss_vec)
        if not self.hparams.disable_logger:
            self.log(f"{mode}_h_score", mean_h_score)
        return mean_h_score

    def run_step(self, batch, step_mode):

        obj_processed_imgs = batch['obj_processed_img']

        global_processed_imgs = {}
        for scales in self.hparams.img_encoder_size:
            scale = scales[0]
            for crop_scale in self.hparams.env_img_crop_scale_list:
                _name = str(scale) + '_' + str(crop_scale)
                global_processed_imgs[_name] = batch['global_processed_img' + _name]
        # obj_masks = batch['obj_mask']
        target = batch['target']

        obj_vit_enc_out = self.obj_vit_region_encoder(obj_processed_imgs)
        obj_vit_embeddings = obj_vit_enc_out.last_hidden_state[:, :1, :]

        batch_size, _, _ = obj_vit_embeddings.shape
        concat_embs = obj_vit_embeddings.view(batch_size, -1)

        if self.hparams.intra_env_attn:
            # img_vit_g_embeddings = {}
            img_vit_p_embeddings = {}
            for scales in self.hparams.img_encoder_size:
                for crop_scale in self.hparams.env_img_crop_scale_list:
                    _name = str(scales[0])+'_'+str(crop_scale)
                    img_vit_enc_out = self.img_vit_region_encoders[_name](global_processed_imgs[_name])
                    img_vit_p_embeddings[_name] = img_vit_enc_out.last_hidden_state[:, 1:, :]

            intra_env_embs_dict = {}
            for scales in self.hparams.img_encoder_size:
                for crop_scale in self.hparams.env_img_crop_scale_list:
                    _name = str(scales[0])+'_'+str(crop_scale)
                    obj_vit_embs = obj_vit_embeddings
                    intra_env_embs_dict[_name] = self.intra_env_attn_module[_name](obj_vit_embs, img_vit_p_embeddings[_name]).view(batch_size, -1)
            intra_env_embs = torch.concat(list(intra_env_embs_dict.values()), 1)
            concat_embs = torch.concat((concat_embs, intra_env_embs), dim=-1)

        embs = self.concat_proj(concat_embs)

        sub_h_loss = 0
        if self.hparams.hierarchical_loss:
            hierarchical_logits_list = self.hierarchical_classifier(embs)
            h_loss_list = self.sub_h_crossentropy_loss(hierarchical_logits_list, target, step_mode)
            h_loss_arr = torch.stack(h_loss_list)
            sub_h_loss = torch.mean(h_loss_arr)

        logits = self.classifier(embs).squeeze()
        ce_loss = self.crossentropy_loss(logits, target, 'class', step_mode)
        h_loss = self.hierarchical_distance(logits, target, step_mode)
        total_loss = (self.hparams.lambda_ce * ce_loss +
                      self.hparams.lambda_sub_h * sub_h_loss
                      )
        if not self.hparams.disable_logger:
            self.log(step_mode + "_loss", total_loss.float().mean())

        return total_loss

    def training_step(self, batch, batch_idx):
        step_mode = 'train'
        total_loss = self.run_step(batch, step_mode)
        return total_loss

    def validation_step(self, batch, batch_idx):
        step_mode = 'val'
        total_loss = self.run_step(batch, step_mode)
        return total_loss