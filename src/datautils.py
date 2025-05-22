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
import random
import cv2

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

        self.obj_enc_processor = AutoImageProcessor.from_pretrained(args.obj_vit_encoder_path)  # 앞으로 빼기
        self.obj_enc_processor.size['shortest_edge'] = args.obj_encoder_size[0]
        self.obj_enc_processor.do_center_crop = False

        self.img_enc_processor = {}
        for scales in args.img_encoder_size:
            for crop_scale in args.env_img_crop_scale_list:
                _name = str(scales[0])+'_'+str(crop_scale)
                self.img_enc_processor[_name] = AutoImageProcessor.from_pretrained(args.img_vit_encoder_path)  # 앞으로 빼기
                self.img_enc_processor[_name].size['shortest_edge'] = scales[0]
                self.img_enc_processor[_name].do_center_crop = False

        self.colorjitter_aug = transforms.Compose([
            # transforms.RandomResizedCrop(size=args.img_encoder_size, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)
            ], p=0.99),
            transforms.RandomRotation(degrees=15, interpolation=InterpolationMode.BILINEAR, expand=False),
            # transforms.RandomGrayscale(p=0.1),
            # transforms.GaussianBlur(kernel_size=(1, 5), sigma=(0.01, 5)),
            # transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.resize = transforms.Resize((18, 18),
                                   interpolation=InterpolationMode.NEAREST_EXACT,
                                   antialias=True)

        self.args = args

        # if self.phase == 'train' or self.phase == 'valid':
        #     self.cate_img_dict = {}
        #     for k in range(len(self.annodata)):
        #         _dat = self.annodata[k]
        #         _img_id = _dat['image_id']
        #         _cate_id = _dat['category_id']
        #         if _cate_id not in self.cate_img_dict.keys():
        #             self.cate_img_dict[_cate_id] = []
        #         self.cate_img_dict[_cate_id].append(_img_id)

    def __len__(self):
        return len(self.annodata)

    def __getitem__(self, idx):
        anno = self.annodata[idx]
        img_id = anno['image_id']
        category_id = anno['category_id']
        bbox = anno['bbox']

        if self.phase == 'train' or self.phase == 'valid':
            image_path = os.path.join('./dataset/fathomnet-2025/train_data/images',str(img_id)+'.png')
            # mask_path = os.path.join('./dataset/fathomnet-2025/train_data/masks',str(img_id)+'.npy')
            # img_mask = torch.tensor(np.load(mask_path))[None,:,:]
            # img_mask = self.resize(img_mask)
        else:
            image_path = os.path.join('./dataset/fathomnet-2025/test_data/images',str(img_id)+'.png')
            # img_mask = 0

        # if self.args.imgxaug:
        #     env_img_id = random.choice(self.cate_img_dict[category_id])
        #     env_image_path = os.path.join('./dataset/fathomnet-2025/train_data/images', str(env_img_id) + '.png')
        #
        #     image = Image.open(image_path)
        #     env_image = Image.open(env_image_path)
        # else:
        image = Image.open(image_path)
        env_image = image

        if image.mode == 'L':
            image = image.convert('RGB')
        img_w, img_h = image.size
        init_x, init_y, w, h = bbox
        center_x = int(init_x + w // 2)
        center_y = int(init_y + h // 2)

        env_image_dict = {}
        for crop_scale in self.args.env_img_crop_scale_list:
            if crop_scale == -1: # -1인 경우 전체
                crop_size = max(img_h, img_w)
            else:
                crop_size = max(w, h)//2 * crop_scale
            y1 = int(max(center_y - crop_size, 0))
            y2 = int(min(center_y + crop_size, img_h))
            x1 = int(max(center_x - crop_size, 0))
            x2 = int(min(center_x + crop_size, img_w))
            w_img = np.array(image)[y1:y2, x1:x2, :]
            # w_img = pad_to_square(w_img, pad_value=0)
            env_image_dict[crop_scale] = Image.fromarray(w_img)

        if self.phase == 'train' and self.args.transform:
            object_crop = random.choice(['narrow', 'padding', 'square'])
            self.args.object_crop = object_crop

        if self.args.object_crop == 'narrow':
            y1 = max(int(init_y) - 1, 0)
            y2 = min(int(init_y + h + 1), img_h)
            x1 = max(int(init_x) - 1, 0)
            x2 = min(int(init_x + w + 1), img_w)
            obj_img = np.array(image)[y1:y2, x1:x2, :]
        elif self.args.object_crop == 'padding':
            y1 = max(int(init_y) - 1, 0)
            y2 = min(int(init_y + h + 1), img_h)
            x1 = max(int(init_x) - 1, 0)
            x2 = min(int(init_x + w + 1), img_w)
            obj_img = np.array(image)[y1:y2, x1:x2, :]
            obj_img = pad_to_square(obj_img, pad_value=0)
        elif self.args.object_crop == 'square':
            norm_half_size = max(w, h) // 2
            y1 = max(int(center_y - norm_half_size - 1), 0)
            y2 = min(int(center_y + norm_half_size + 1), img_h)
            x1 = max(int(center_x - norm_half_size - 1), 0)
            x2 = min(int(center_x + norm_half_size + 1), img_w)
            obj_img = np.array(image)[y1:y2, x1:x2, :]


        if self.phase == 'train'  and self.args.transform:
            # oh, ow, oc = obj_img.shape
            # if random.random() > 0.9 and (oh > 10 and ow > 10):
            #     resize_factor = random.uniform(0.1, 0.9)
            #     new_width = int(obj_img.shape[1] * resize_factor)
            #     new_height = int(obj_img.shape[0] * resize_factor)
            #     obj_img = cv2.resize(obj_img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            obj_img = self.colorjitter_aug(Image.fromarray(obj_img))
            for k in env_image_dict:
                env_image_dict[k] = self.colorjitter_aug(env_image_dict[k])
        else:
            obj_img = Image.fromarray(obj_img)

        obj_processed_img = (self.obj_enc_processor(images=obj_img.resize(self.args.obj_encoder_size),return_tensors="pt").pixel_values).squeeze(dim=0)  ###224

        img_processed = {}
        for scales in self.args.img_encoder_size:
            for crop_scale in self.args.env_img_crop_scale_list:
                scale = scales[0]
                _name = str(scale)+'_'+str(crop_scale)
                img_processed['global_processed_img'+_name] = (self.img_enc_processor[_name](images=env_image_dict[crop_scale].resize(scales),
                                                    return_tensors="pt").pixel_values).squeeze(dim=0)

        if str(category_id) == 'None':
            category_id = -1

        target = category_id
        anno['category_id'] = category_id
        anno['filepath'] = image_path

        res_dat = {
            "obj_processed_img": obj_processed_img,
            # "obj_mask" : img_mask,
            "target" : target,
            "obj_anno": anno}
        res_dat.update(img_processed)

        return res_dat