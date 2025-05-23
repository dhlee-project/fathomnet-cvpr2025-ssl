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
import tqdm
import pandas as pd
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
    parser.add_argument("--config", type=str, default='./config/experiment54.yaml', help='Path to config file')
    parser.add_argument("--mode", type=str, default=None)
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=str, default=None)
    return parser.parse_args()

args = get_parser()
config = OmegaConf.load(args.config)
pl.seed_everything(config.seed_number)
init_random_seed(config.seed_number)
config.kfold = False
config.transform = False
config.imgxaug = False
config.kfold_nsplits = 1
batchsize = 2
device = 'cuda'
test_anno_path = './dataset/fathomnet-2025/dataset_test.json'
with open(test_anno_path, 'r', encoding='utf-8') as f:
    dataset = json.load(f)
anno_data = dataset['annotations']
len_cate = len(dataset['categories'])
cate_name2id = {}
cate_id2name = {}
for i in range(len_cate):
    cate_id = dataset['categories'][i]['id'] -1
    dataset['categories'][i]['id'] = cate_id
    cate_name = dataset['categories'][i]['name']
    cate_name2id[cate_name] = cate_id
    cate_id2name[cate_id] = cate_name
config.category_name2id = cate_name2id
config.category_id2name = cate_id2name

fold_indices = []
data_len = len(anno_data)
sampled_val = np.random.choice(list(range(data_len)), 100).tolist()
fold_indices.append({'train' : list(range(data_len)[:]),
                     'val': sampled_val})

print('Run Testing')
config.current_fold = 0
results = []
n_fold = config.kfold_nsplits
for current_fold in range(n_fold):
    model_path = f'~/Project/cvprcom/logs/{config.project_name}/Fold-{current_fold}/last.ckpt'
    Fathomnet_model = FathomnetModel.load_from_checkpoint(model_path, inter_env_attn=False).to(device)
    Fathomnet_model.eval()

    config.current_fold = current_fold
    train_anno_ids = fold_indices[0]['train']
    train_anno = [anno_data[i] for i in train_anno_ids]

    testdataset = Fathomnet_Dataset(train_anno[:],
                phase='test',
                args=config
                )
    # #sync_dsrv.sh
    test_dataloader = DataLoader(
        testdataset,
        batch_size=batchsize,
        num_workers=config.num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=True)

    with torch.no_grad():
        for batch in tqdm.tqdm(test_dataloader):
            # print(volume_images.shape, seg_images.shape)  # e.g., [1, H, W, D]
            obj_processed_imgs = batch['obj_processed_img'].to(device)
            # global_processed_imgs = batch['global_processed_img'].to(device)
            target = batch['target']
            obj_anno = batch['obj_anno']

            global_processed_imgs = {}
            for scales in Fathomnet_model.hparams.img_encoder_size:
                for crop_scale in Fathomnet_model.hparams.env_img_crop_scale_list:
                    _name = str(scales[0]) + '_' + str(crop_scale)
                    global_processed_imgs[_name] = batch[f'global_processed_img{_name}'].to(device)
            # obj_masks = batch['obj_mask']


            batch_size, _, _, _ = obj_processed_imgs.shape
            obj_vit_enc_out = Fathomnet_model.obj_vit_region_encoder(obj_processed_imgs)
            obj_vit_embeddings = obj_vit_enc_out.last_hidden_state[:, :1, :]

            img_vit_g_embeddings = {}
            img_vit_p_embeddings = {}
            for scales in Fathomnet_model.hparams.img_encoder_size:
                for crop_scale in Fathomnet_model.hparams.env_img_crop_scale_list:
                    _name = str(scales[0]) + '_' + str(crop_scale)
                    img_vit_enc_out = Fathomnet_model.img_vit_region_encoders[_name](global_processed_imgs[_name])
                    img_vit_g_embeddings[_name] = img_vit_enc_out.last_hidden_state[:, :1, :]
                    img_vit_p_embeddings[_name] = img_vit_enc_out.last_hidden_state[:, 1:, :]

            concat_embs = obj_vit_embeddings.view(obj_vit_embeddings.shape[0], -1)
            if Fathomnet_model.hparams.intra_env_attn:
                intra_env_embs_dcit = {}
                for scales in Fathomnet_model.hparams.img_encoder_size:
                    for crop_scale in Fathomnet_model.hparams.env_img_crop_scale_list:
                        _name = str(scales[0]) + '_' + str(crop_scale)
                        intra_env_embs_dcit[_name] = Fathomnet_model.intra_env_attn_module[_name](obj_vit_embeddings, img_vit_p_embeddings[_name]).view(batch_size, -1)
                intra_env_embs = torch.concat(list(intra_env_embs_dcit.values()), -1)
                concat_embs = torch.concat((concat_embs, intra_env_embs), dim=-1)
                # concat_embs = Fathomnet_model.combiner(obj_vit_embeddings.view(batch_size, -1), intra_env_embs)

            embs = Fathomnet_model.concat_proj(concat_embs)
            logits = Fathomnet_model.classifier(embs).squeeze()
            preds = torch.argmax(logits, dim=1).cpu().numpy().astype(float)
            preds_class = [Fathomnet_model.hparams.category_id2name[preds[i]] for i in range(len(preds))]
            obj_annos = obj_anno['id'].cpu().numpy()
            for i in range(len(obj_annos)):
                results.append([current_fold, obj_annos[i], preds_class[i]])

submission = pd.DataFrame(results)
submission.columns = ['fold', 'annotation_id', 'concept_name']
# submission.to_csv(f"./results/submission_allfold_{config.project_name}_0513_02.csv", index=False)
submission = submission[['annotation_id', 'concept_name']]
voted_submission = (
    submission.groupby(['annotation_id'])['concept_name']
    .agg(lambda x: x.mode().iloc[0])  # 최빈값 (복수일 경우 첫 번째 선택)
    .reset_index()
)
voted_submission.to_csv(f"./results/submission_{config.project_name}_0522_01.csv", index=False)