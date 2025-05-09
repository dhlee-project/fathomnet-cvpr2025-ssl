import os
from PIL import Image
import pickle
import torch
from torch.utils.data import Dataset
from transformers import AutoImageProcessor
from torchvision.transforms import InterpolationMode
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np


def pad_to_square(img, pad_value=0):
    h, w = img.shape[:2]
    size = max(h, w)

    # 위아래 또는 좌우에 얼마나 패딩할지 계산
    pad_top = (size - h) // 2
    pad_bottom = size - h - pad_top
    pad_left = (size - w) // 2
    pad_right = size - w - pad_left

    # np.pad은 (top, bottom), (left, right), (채널은 0)
    padded_img = np.pad(
        img,
        pad_width=((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode='constant',
        constant_values=pad_value
    )
    return padded_img

class Fathomnet_Dataset(Dataset):
    def __init__(
            self,
            annodata=None,
            phase='train',
            args=None
    ):
        self.annodata = annodata
        self.phase = phase

        self.obj_enc_processor = AutoImageProcessor.from_pretrained(args.obj_encoder_path)  # 앞으로 빼기
        self.obj_enc_processor.size['shortest_edge'] = args.obj_encoder_size[0]
        self.obj_enc_processor.do_center_crop = False

        self.img_enc_processor = AutoImageProcessor.from_pretrained(args.img_encoder_path)  # 앞으로 빼기
        self.img_enc_processor.size['shortest_edge'] = args.img_encoder_size[0]
        self.img_enc_processor.do_center_crop = False
        self.args = args

    def __len__(self):
        return len(self.annodata)

    def __getitem__(self, idx):
        anno = self.annodata[idx]
        img_id = anno['image_id']
        category_id = anno['category_id']
        bbox = anno['bbox']

        if self.phase == 'train' or self.phase == 'valid':
            image_path = os.path.join('./dataset/fathomnet-2025/train_data/images',str(img_id)+'.png')
        else:
            image_path = os.path.join('./dataset/fathomnet-2025/test_data/images',str(img_id)+'.png')

        image = Image.open(image_path)
        if image.mode == 'L':
            image = image.convert('RGB')
        self.args.object_crop = 'square'
        init_x, init_y, w, h = bbox
        center_x = int(init_x + w // 2)
        center_y = int(init_y + h // 2)
        if self.args.object_crop == 'narrow':
            obj_img = np.array(image)[int(init_y):int(init_y + h), int(init_x):int(init_x + w), :]
        elif self.args.object_crop == 'padding':
            obj_img = np.array(image)[int(init_y):int(init_y + h), int(init_x):int(init_x + w), :]
            obj_img = pad_to_square(obj_img, pad_value=0)
        elif self.args.object_crop == 'square':
            norm_half_size = max(w, h) // 2
            obj_img = np.array(image)[int(center_y - norm_half_size):int(center_y + norm_half_size),
                      int(center_x - norm_half_size):int(center_x + norm_half_size), :]
        obj_img = Image.fromarray(obj_img)

        obj_processed_img = (self.obj_enc_processor(images=obj_img.resize(self.args.obj_encoder_size),return_tensors="pt").pixel_values).squeeze(dim=0)  ###224
        img_processed_img = (self.img_enc_processor(images=image.resize(self.args.img_encoder_size),return_tensors="pt").pixel_values).squeeze(dim=0)

        if str(category_id) == 'None':
            category_id = -1

        target = category_id
        anno['category_id'] = category_id
        anno['filepath'] = image_path
        return {
            "obj_processed_img": obj_processed_img,
            "global_processed_img": img_processed_img,
            "target" : target,
            "obj_anno": anno
        }