

class biological_distance():
    def __init__(self, biological_matrix):
        self.biological_matrix = biological_matrix

    def get_csv_ancestor_list(self, df, name):
        """
        Get the ancestor list from the CSV file for a given name.
        The CSV file contains the list of ancestor names for each category.
        """

        values = list(df[df['Unnamed: 0'] == name].to_dict().values())

        ancestor_list = []
        for i in range(len(values)):
            ancestor_item = list(values[i].values())[0]
            ancestor_list.append(ancestor_item)

        ancestor_list = ancestor_list[1:]

        return ancestor_list

    def find_lca(self, ancestor1, ancestor2):
        """
        Find the Lowest Common Ancestor (LCA) of two ancestor lists.
        The LCA is the first common ancestor in the ancestor lists.
        """

        set2 = set(ancestor2)

        for ancestor in ancestor1:
            if ancestor in set2:
                return ancestor

        return None

    def get_distance_from_lca(self, ancestor_list, lca):
        """
        Get the distance from the LCA to a given ancestor list.
        The distance is the index of the LCA in the ancestor list.
        """

        if lca in ancestor_list:
            dist = ancestor_list.index(lca)
            return dist
        else:
            raise ValueError('Invalid LCA')

    def evaluation(self, name1, name2):
        """
        THIS FUNCTION IS FAST EVALUATION FUNCTION. USE THIS FUNCTION.
        Calculate the taxonomic distance between two names using the CSV file.
        The distance is the sum of the distances from the LCA to each name.
        """
        category_df = self.biological_matrix

        ancestor_list1 = self.get_csv_ancestor_list(category_df, name1)
        ancestor_list2 = self.get_csv_ancestor_list(category_df, name2)

        lca = self.find_lca(ancestor_list1, ancestor_list2)

        dist1 = self.get_distance_from_lca(ancestor_list1, lca)
        dist2 = self.get_distance_from_lca(ancestor_list2, lca)

        full_dist = dist1 + dist2
        return full_dist



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