import numpy as np
import pandas as pd
from tqdm import tqdm
from fathomnet.api import worms
from sklearn.preprocessing import LabelEncoder
import json
import pickle
import argparse

parser = argparse.ArgumentParser(description="FathomNet 2025 preprocessing")
parser.add_argument('--data_path', type=str, default='../../dataset/fathomnet-2025/dataset_train.json',
                    help='Path to dataset_train.json')
args = parser.parse_args()
data_path = args.data_path

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

def recursive_child_snatcher(anc):
    children = [x.name for x in anc.children]
    ranks = [x.rank for x in anc.children]

    assert len(children) == 1  # 단일 경로만 가정
    if len(anc.children[0].children) > 0:
        child_names, child_ranks = recursive_child_snatcher(anc.children[0])
        return children + child_names, ranks + child_ranks
    else:
        return children, ranks

with open(data_path, 'r') as f:
    data = json.load(f)
classes = [x['name'] for x in data['categories']]

accepted_ranks = np.array(['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species'])

tree_dict = {}
tree_rank_dict = {}
for label in tqdm(classes, desc="Getting taxonomy paths"):
    anc = worms.get_ancestors(label)
    children, ranks = recursive_child_snatcher(anc)
    children = np.array(children[1:])  # 자기 자신 제외
    ranks = np.array(ranks[1:])
    mask = np.isin(ranks, accepted_ranks)
    filtered_children = children[mask]
    filtered_ranks = ranks[mask]
    tree_dict[label] = filtered_children.tolist()
    tree_rank_dict[label] = filtered_ranks.tolist()

### make tree
bio_tree = BiologicalTree()
for class_name, taxonomy_path in tree_dict.items():
    bio_tree.add_taxonomy_path(taxonomy_path[::-1])
bio_tree.distance('Abyssocucumis abyssorum', 'Microstomus pacificus')

class_names = list(tree_dict.keys())
n_classes = len(class_names)

distance_matrix = pd.DataFrame(
    np.zeros((n_classes, n_classes)),
    index=class_names,
    columns=class_names
)

for i in tqdm(range(n_classes), desc="Calculating distances"):
    for j in range(i, n_classes):
        src_leaf = tree_dict[class_names[i]][-1]
        dst_leaf = tree_dict[class_names[j]][-1]
        dist = bio_tree.distance(src_leaf, dst_leaf)
        distance_matrix.iloc[i, j] = dist
        distance_matrix.iloc[j, i] = dist

distance_matrix.to_csv('./results/dist_categories.csv')

hierachical_category = np.zeros((79 , 8)).astype(object)
hierachical_category[:] = 'bin'
for i, k in enumerate(list(tree_dict.keys())):
    single_path = tree_dict[k]
    single_rank_path = tree_rank_dict[k]
    len_path = len(single_path)
    if not len_path == len(accepted_ranks[:len_path]):
        break
    hierachical_category[i,:len_path] = single_path
    hierachical_category[i,len_path:-1] = single_path[-1]
    hierachical_category[i,-1] = k

colnames = np.array(['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species', 'target'])
df = pd.DataFrame(hierachical_category, columns=colnames)
df2 = df.copy()
encoded_columns = []
id2name = {}
name2id = {}
for col in df.columns:
    le = LabelEncoder()
    df2[col] = le.fit_transform(df[col])
    encoded_columns.append(df2[col])
    id2name[col] = {_id: _name for _id, _name in zip(le.fit_transform(df[col]), df[col])}
    name2id[col] = {_name: _id for _id, _name in zip(le.fit_transform(df[col]), df[col])}

df_encoded = pd.concat(encoded_columns, axis=1)
hierachical_labelencoder = {'id2name' : id2name, 'name2id' : name2id}
with open('./results/hierachical_labelencoder.pkl', 'wb') as f:
    pickle.dump(hierachical_labelencoder, f)

len_cate = len(data['categories'])
cate_name2id = {}
cate_id2name = {}
for i in range(len_cate):
    cate_id = data['categories'][i]['id'] - 1  # 0~78
    data['categories'][i]['id'] = cate_id
    cate_name = data['categories'][i]['name']
    cate_name2id[cate_name] = cate_id
    cate_id2name[cate_id] = cate_name

categories = df.target
categories_id = [cate_name2id[i] for i in categories]
df_encoded.index = categories_id

df_selected_encoded = df_encoded[['Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']]
df_selected_encoded.to_csv('./results/hierarchical_label.csv', index=True)


