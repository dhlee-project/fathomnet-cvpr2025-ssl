
import pandas as pd
import sys
from ete3 import Tree

import numpy as np
import pandas as pd

from tqdm import tqdm
from ete3 import Tree
from fathomnet.api import worms
import pandas as pd


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

# Get list of accepted categories

import json
with open('./dataset/fathomnet-2025/dataset_train.json', 'r') as f:
    data = json.load(f)
classes = [x['name'] for x in data['categories']]
classes


def recursive_child_snatcher(anc):
    # Recursively gets a list of children and ranks from a fathomnet ancestor.
    children = [x.name for x in anc.children]
    childrens_ranks = [x.rank for x in anc.children]

    assert len(children) == 1  # bifurcating trees not implemented
    if len(anc.children[0].children) > 0:
        childrens_children, childrens_childrens_ranks = recursive_child_snatcher(anc.children[0])
        return children + childrens_children, childrens_ranks + childrens_childrens_ranks
    else:
        return children, childrens_ranks


# convert to an ete3 Tree (This is personal preference as I have worked with them before)
tree = Tree()
already_added = ['']
for label in tqdm(classes):
    if label in already_added:
        continue
    anc = worms.get_ancestors(label)
    children, ranks = recursive_child_snatcher(anc)
    children = [''] + children
    ranks = [''] + ranks
    for i in range(len(children) - 1):
        parent_name, child_name = children[i:i + 2]
        parent_rank, child_rank = ranks[i:i + 2]
        if child_name in already_added:
            continue
        parent_node = [node for node in tree.traverse() if node.name == parent_name][0]
        parent_node.rank = parent_rank
        child = Tree(name=child_name)
        child.rank = child_rank
        parent_node.add_child(child)
        already_added += [child_name]
print(tree)

# set distances to 0 for ranks not included in loss calculation
for node in tree.traverse():
    if node.name in classes:
        continue
    accepted_ranks = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
    if node.rank not in accepted_ranks:
        node.dist = 0
# write tree in newick format including distances
tree.write(outfile="tree.nh",format=3)

# make distance matrix
def tree_to_distance_matrix(tree, labels):
    n = len(labels)
    labels = sorted(labels)

    # Create a blank distance matrix
    dist_matrix = np.zeros((n, n))

    # Fill the matrix with pairwise distances
    for i, name1 in enumerate(labels):
        node1 = [node for node in tree.traverse() if node.name == name1][0]
        for j, name2 in enumerate(labels):
            if i <= j:

                d = node1.get_distance(str(name2))
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d  # symmetric

    df = pd.DataFrame(dist_matrix, index=labels, columns=labels)

    return df
df = tree_to_distance_matrix(tree, classes)
df.to_csv('./results/dist_categories.csv')
aa = pd.read_csv('./results/dist_categories.csv', index_col=0)
aa.loc['Abyssocucumis abyssorum', 'Asbestopluma monticola']
#
#
#
# ranks_dataset = []
# accepted_ranks = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
# for label in tqdm(classes):
#     anc = worms.get_ancestors(label)
#     children, ranks = recursive_child_snatcher(anc)
#     children = children[1:]
#     ranks = ranks[1:]
#     _dict = {}
#     for rank, child in zip(ranks, children):
#         # if rank in accepted_ranks:
#         _dict[rank] = child
#     ranks_dataset.append(_dict)
#
# from collections import OrderedDict
#
# # 모든 key 등장 순서대로 중복 없이 수집
# ordered_columns = list(OrderedDict.fromkeys(
#     key for d in ranks_dataset for key in d.keys()
# ))
#
# df = pd.DataFrame(ranks_dataset)
# biological_category = df
# # biological_category = pd.read_csv('./results/categories.csv')
# bio_tree = BiologicalTree()
#
# for i in range(len(biological_category)):
#     single_path = biological_category.iloc[i,:].values[::-1]
#     bio_tree.add_taxonomy_path(single_path)
# bio_tree.print_tree()
# bio_tree.find_path('Terebellida')
# bio_tree.find_path('Serpulidae')
# bio_tree.distance('Serpulidae', 'Serpulidae')
# bio_tree.distance('Sabellida', 'Terebellida')
# bio_tree.distance('Abyssocucumis abyssorum', 'Asbestopluma monticola')
# bio_tree.distance('Tunicata', 'Sebastes')
# bio_tree.distance('Abyssocucumis abyssorum', 'Asbestopluma monticola')

