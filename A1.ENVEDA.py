import glob
import os
import json
import pickle
import argparse
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from datetime import datetime
import pytorch_lightning as pl
import torch
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoConfig
from transformers import AutoImageProcessor, ViTModel
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from sklearn.manifold import TSNE
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np

### 환경하고 연관이 있는가?
### 고려하지 않으면 분류하는데 문제가 있을까?
### train/test간 사이즈 차이가 있나?
class EDADataset(Dataset):
    def __init__(
            self,
            image_paths,
    ):
        self.image_paths = image_paths

        self.patch_enc_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')  # 앞으로 빼기
        self.patch_enc_processor.size['shortest_edge'] = (518, 518)
        self.patch_enc_processor.do_center_crop = False
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        patch_processed_img = (self.patch_enc_processor(images=image, return_tensors="pt").pixel_values).squeeze(dim=0)
        return patch_processed_img, img_path


#
train_anno_path = './dataset/fathomnet-2025/dataset_train.json'
with open(train_anno_path, 'r', encoding='utf-8') as f:
    train_anno_data = json.load(f)
train_anno_data.keys()#['info', 'images', 'licenses', 'annotations', 'categories']
anno_data = train_anno_data

len_cate = len(anno_data['categories'])
cate_name2id = {}
cate_id2name = {}
for i in range(len_cate):
    cate_id = anno_data['categories'][i]['id']
    cate_name = anno_data['categories'][i]['name']
    cate_name2id[cate_name] = cate_id
    cate_id2name[cate_id] = cate_name
for i in range(len(anno_data['annotations'])):
    anno_data['annotations'][i]['category_name'] = cate_id2name[anno_data['annotations'][i]['category_id']]
anno_data['cate_name2id'] = cate_name2id
anno_data['cate_id2name'] = cate_id2name

len(anno_data['annotations'])


image_obj_dict = {}
for i in range(len(anno_data['annotations'])):
    img_id = anno_data['annotations'][i]['image_id']
    category_name = anno_data['annotations'][i]['category_name']
    # if img_id == 7352:
    #     print(anno_data['annotations'][i])
    if img_id not in list(image_obj_dict.keys()):
        image_obj_dict[img_id] = []
    image_obj_dict[img_id].append(category_name)

len(image_obj_dict)
len(anno_data['images'])
cnt_v = np.array([len(v) for v in image_obj_dict.values()])
print(f'image 갯수 : {len(cnt_v)}')
print(f'image 당 mean 객체 갯수 : {np.mean(cnt_v)}')
print(f'image 당 median 객체 갯수 : {np.median(cnt_v)}')
print(f'image 당 std 객체 갯수 : {np.std(cnt_v)}')
print(f'image 당 max 객체 갯수 : {np.max(cnt_v)}')
print(f'image 당 min 객체 갯수 : {np.min(cnt_v)}')
plt.boxplot(cnt_v); plt.show()
plt.hist(cnt_v, bins=100); plt.show()
# anno_data['info']
# anno_data['images']
# anno_data['annotations'][0]
# anno_data['categories']
# anno_data['images'][93]
# len_cate = len(anno_data['categories'])
# cate_dict = {}
# for i in range(len_cate):
#     cate_id = anno_data['categories'][i]['id']
#     cate_name = anno_data['categories'][i]['name']
#     cate_dict[cate_id] = cate_name
# anno_data['annotations'][0]
# anno_data['annotations'][0]['category_id'] = cate_dict[anno_data['annotations'][0]['category_id']]

# anno = train_anno_data['annotations'][0]



idx = 401
img_id = train_anno_data['annotations'][idx]['image_id']
rois_id = train_anno_data['annotations'][idx]['id']
init_x, init_y, w, h = train_anno_data['annotations'][idx]['bbox']
center_x = int(init_x+w//2)
center_y = int(init_y+h//2)
norm_half_size = max(w, h) // 2

img = plt.imread(f'./dataset/fathomnet-2025/train_data/images/{img_id}.png')
plt.imshow(img); plt.show()
# plt.imshow(img[int(cx-w//2):int(cx+w//2),int(cy-h//2):int(cy+h//2),:]); plt.show()
plt.imshow(img[int(init_y):int(init_y+h),int(init_x):int(init_x+w),:]); plt.show()
plt.imshow(img[int(center_y-norm_half_size):int(center_y+norm_half_size),int(center_x-norm_half_size):int(center_x+norm_half_size),:]); plt.show()

img = plt.imread(f'./dataset/fathomnet-2025/train_data/rois/{img_id}_{rois_id}.png')
plt.imshow(img); plt.show()
anno_path = './dataset/fathomnet-2025/train_data/annotations.csv'
anno_data = pd.read_csv(anno_path)

# load dataset
load_embs = True
if load_embs:
    with open('./results/global_image_data.pkl', 'rb') as f:
        global_image_data = pickle.load(f)
    g_id_arr = global_image_data['ids']
    g_embs_arr = global_image_data['embs']
else:
    img_region_encoder = AutoModel.from_pretrained('facebook/dinov2-base').cuda().eval()
    img_list = glob.glob(f'./dataset/fathomnet-2025/train_data/images/*.png')
    image_length = len(img_list)
    dataset = EDADataset(img_list)
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        num_workers=10,
        shuffle=False,
        drop_last=False,
        pin_memory=True)
    g_embs_list = []; i =0
    g_embs_arr = torch.zeros((image_length,768))
    g_id_arr = np.zeros((image_length)).astype(object)
    with torch.no_grad():
        for batch in dataloader:
            print(f'{i}/{image_length}')
            batch_image, batch_id = batch
            enc_out = img_region_encoder(batch_image.cuda())
            g_embs = enc_out.last_hidden_state[:,0,:].detach().cpu()
            g_embs_list.append(g_embs)
            for j in range(len(g_embs)):
                g_embs_arr[i] = g_embs[j]
                g_id_arr[i] = batch_id[j]
                i+=1

    g_embs_arr = g_embs_arr.numpy()
    g_id_arr = g_id_arr
    global_image_data = {'embs' : g_embs_arr, 'ids': g_id_arr}
    with open('./results/global_image_data.pkl', 'wb') as f:
        pickle.dump(global_image_data, f)


# class - tsne
g_id_list = np.array([int(os.path.splitext(os.path.basename(g_id_arr[i]))[0]) for i in range(len(g_id_arr))])
g_id_list2 = g_id_list[g_id_list!=7353]
selected_obj_per_image = np.array([cate_name2id[image_obj_dict[v][0]] for v in g_id_list2]) # 1개만 추출
X = g_embs_arr[g_id_list!=7353]
y = selected_obj_per_image[:]

# t-SNE(64 to 2)
tsne = TSNE(n_components=2, random_state=1)
X_tsne = tsne.fit_transform(X)
# plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y)
# plt.show()
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='nipy_spectral', s=10, alpha=0.7)
plt.colorbar(scatter, label='Class ID')  # colorbar로 범례 대체
plt.title("t-SNE Visualization")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.grid(True)
plt.tight_layout()
plt.show()

### kmeans  - tsne
n_clusters = 256  # Set number of clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
kmeans.fit(X)

# Get cluster assignments
labels = kmeans.labels_

# Get cluster centers
centers = kmeans.cluster_centers_


XX = np.concatenate((X, centers), axis=0)
tsne = TSNE(n_components=2, random_state=1)
X_tsne2 = tsne.fit_transform(XX)
# labels2 = np.concatenate((labels, np.arange(n_clusters)))
labels2 = np.concatenate((labels+1, np.array([0]*n_clusters)))

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne2[:, 0], X_tsne2[:, 1], c= labels2, cmap='nipy_spectral', s=10, alpha=0.7)
plt.colorbar(scatter, label='Class ID')  # colorbar로 범례 대체
plt.title("t-SNE Visualization")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.grid(True)
plt.tight_layout()
plt.show()

# with open('./results/centroid_embs_256.pkl', 'wb') as f:
#     pickle.dump(centers, f)

# X: 원본 데이터 (N, D), dtype=float32
# 차원 축소 (예: 2D 또는 10D)
# n_components = 10  # 또는 2로 설정하면 시각화도 가능
# X_pca = PCA(n_components=n_components, random_state=42).fit_transform(X)
#
# # 실루엣 점수 계산
# range_n_clusters = range(10, 200, 10)
# silhouette_scores = []
#
# for n_clusters in range_n_clusters:
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
#     cluster_labels = kmeans.fit_predict(X_pca)
#
#     score = silhouette_score(X_pca, cluster_labels)
#     silhouette_scores.append(score)
#     print(f"PCA-k = {n_clusters}, silhouette score = {score:.4f}")
#
# # 시각화
# plt.plot(range_n_clusters, silhouette_scores, marker='o')
# plt.xlabel("Number of clusters (k)")
# plt.ylabel("Silhouette Score (after PCA)")
# plt.title(f"Silhouette Analysis (PCA-{n_components}D)")
# plt.grid(True)
# plt.show()

class BiologicalTree:
    def __init__(self):
        self.tree = {}

    def _clean_taxonomy(self, taxonomy):
        # np.nan 또는 None 제거 + 순서 root → leaf
        return [x for x in reversed(taxonomy) if isinstance(x, str)]

    def add_taxonomy_path(self, taxonomy):
        path = self._clean_taxonomy(taxonomy)
        node = self.tree
        for taxon in path:
            if taxon not in node:
                node[taxon] = {}
            node = node[taxon]

    def distance(self, src, dst):
        """src와 dst 노드 간 거리 반환"""
        path1 = self.find_path(src)
        path2 = self.find_path(dst)
        if path1 is None or path2 is None:
            raise ValueError(f"경로를 찾을 수 없습니다: {src} 또는 {dst}")

        # 최소 공통 조상까지의 거리 계산
        min_len = min(len(path1), len(path2))
        lca_index = 0
        for i in range(min_len):
            if path1[i] != path2[i]:
                break
            lca_index = i + 1

        # 거리 = src까지 거리 + dst까지 거리 - 2 * 공통 부분
        return (len(path1) - lca_index) + (len(path2) - lca_index)

    def find_path(self, target_name, node=None, path=None):
        """특정 생물 이름의 분류 경로를 반환"""
        if node is None:
            node = self.tree
        if path is None:
            path = []

        for key, child in node.items():
            new_path = path + [key]
            if key == target_name:
                return new_path
            result = self.find_path(target_name, child, new_path)
            if result is not None:
                return result
        return None

    def print_tree(self, node=None, indent=0):
        if node is None:
            node = self.tree
        for key in node:
            print("  " * indent + f"- {key}")
            self.print_tree(node[key], indent + 1)


import pandas as pd
biological_category = pd.read_csv('./results/categories.csv')
bio_tree = BiologicalTree()

for i in range(len(biological_category)):
    single_path = biological_category.iloc[i,:].values
    bio_tree.add_taxonomy_path(single_path)
bio_tree.print_tree()
bio_tree.find_path('Terebellida')
bio_tree.find_path('Serpulidae')
bio_tree.distance('Serpulidae', 'Serpulidae')
bio_tree.distance('Sabellida', 'Terebellida')