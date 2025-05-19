import glob
import os
import argparse
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from src.datautils import Fathomnet_Dataset
from src.model import FathomnetModel
import nibabel as nib
import random
import json
from sklearn.model_selection import KFold
import numpy as np
import torch
import wandb
from sklearn.model_selection import GroupKFold

def init_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_logger(config):
    # time_now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    foldnum = f'Fold-{config.current_fold}'
    run_id = f"{config.project_name}-fold{config.current_fold}"
    log_dir = f"logs/{config.project_name}/{foldnum}/"
    os.makedirs(log_dir, exist_ok=True)
    if not config.disable_logger:
        if config.logger_id:
            logger = WandbLogger(name=foldnum,
                                 project=config.project_name,
                                 log_model=False,
                                 save_dir=log_dir,
                                 id=config.logger_id,
                                 resume="allow",
                                 reinit=True
                                 )
        else:
            logger = WandbLogger(name=foldnum,
                                 project=config.project_name,
                                 id=run_id,
                                 reinit=True,
                                 log_model=False,
                                 resume=False,
                                 save_dir=log_dir,
                                 )
    else:
        logger = None

    ckpt_cb = ModelCheckpoint(dirpath=log_dir,
                              monitor='val_loss',
                              mode="min",
                              save_top_k=0,
                              save_last=True)

    return logger, ckpt_cb

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='./config/experiment28.yaml', help='Path to config file')
    parser.add_argument("--mode", type=str, default=None)
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=str, default=None)
    return parser.parse_args()

args = get_parser()
config = OmegaConf.load(args.config)
pl.seed_everything(config.seed_number)
init_random_seed(config.seed_number)

train_anno_path = './dataset/fathomnet-2025/dataset_train.json'
with open(train_anno_path, 'r', encoding='utf-8') as f:
    dataset = json.load(f)
anno_data = dataset['annotations']

len_cate = len(dataset['categories'])
cate_name2id = {}
cate_id2name = {}
for i in range(len_cate):
    cate_id = dataset['categories'][i]['id'] -1 # 0~79
    dataset['categories'][i]['id'] = cate_id
    cate_name = dataset['categories'][i]['name']
    cate_name2id[cate_name] = cate_id
    cate_id2name[cate_id] = cate_name
config.category_name2id = cate_name2id
config.category_id2name = cate_id2name

len_anno = len(anno_data)
for i in range(len_anno):
    cate_id = anno_data[i]['category_id'] -1 # 0~79
    anno_data[i]['category_id']  = cate_id

if config.kfold:
    # groups = [ann['image_id'] for ann in anno_data]  # 각 annotation에 대한 image_id
    # np.random.seed(777)
    # np.random.shuffle(groups)
    # kf = GroupKFold(n_splits=config.kfold_nsplits)
    # fold_indices = []
    # for fold, (train_idx, val_idx) in enumerate(kf.split(anno_data, groups=groups)):
    #     fold_indices.append({
    #         'train': train_idx.tolist(),
    #         'val': val_idx.tolist()
    #     })
    data_len = len(anno_data)
    kf = KFold(n_splits=config.kfold_nsplits, shuffle=True, random_state=777)  # 재현성을 위한 random_state
    fold_indices = []
    for fold, (train_index, val_index) in enumerate(kf.split(range(data_len))):
        fold_indices.append({
            'train': train_index.tolist(),
            'val': val_index.tolist()
        })
else:
    fold_indices = []
    data_len = len(anno_data)
    sampled_val = np.random.choice(list(range(data_len)), 100).tolist()
    fold_indices.append({'train' : list(range(data_len)),
                         'val': sampled_val})

print('Run Fold Training')
n_fold = len(fold_indices)
config.current_fold = 0
for current_fold in range(n_fold):
    print(current_fold)

    if wandb.run is not None:
        wandb.finish()

    config.current_fold = current_fold
    logger, ckpt_cb = load_logger(config)
    train_anno_ids = fold_indices[current_fold]['train']
    valid_anno_ids = fold_indices[current_fold]['val']
    train_anno = [anno_data[i] for i in train_anno_ids]
    valid_anno = [anno_data[i] for i in valid_anno_ids]

    traindataset = Fathomnet_Dataset(train_anno[:],
                phase='train',
                args=config
                )
    # #sync_dsrv.sh
    train_dataloader = DataLoader(
        traindataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True)

    validdataset = Fathomnet_Dataset(valid_anno[:],
                phase='valid',
                args=config
                )

    valid_dataloader = DataLoader(
        validdataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        drop_last=True,
        pin_memory=True)


    lr_monitor_cb = LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=config.device_num,
        max_epochs=config.max_epochs,
        accumulate_grad_batches=config.accumulate_grad_batches,
        callbacks=[lr_monitor_cb, ckpt_cb],
        logger=logger, precision="16-mixed")  # precision="16-mixed"

    # Setting the seed
    # Check whether pretrained model exists. If yes, load it and skip training
    if config.trained_model_name:
        ckpt_path = f'./logs/{config.project_name}/{config.trained_model_name}/last.ckpt'
        model = FathomnetModel(config)
        trainer.fit(model, train_dataloader, valid_dataloader, ckpt_path=ckpt_path)
    else:
        model = FathomnetModel(config)
        trainer.fit(model, train_dataloader, valid_dataloader)