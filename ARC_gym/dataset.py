from torch.utils.data import Dataset
import ARC_gym.utils.graphs as graphUtils
import ARC_gym.utils.tokenization as tok
import re
import numpy as np
import os,json
import random

from ARC_gym.Hodel_primitives import *

# Define a color map
COLOR_MAP = {
    0: 'black',
    1: 'steelblue',
    2: 'green',
    3: 'yellow',
    4: 'purple',
    5: 'orange',
    6: 'red',
    7: 'salmon',
    8: 'aquamarine',
    9: 'white'
}

class ARCGymVariableDataset(Dataset):
    '''
    This builds on top of ARCGymPatchesDataset, but instead of assuming fixed square grids, we allowed for variable
    size, non-square grids. This means that tokenization must be a bit more sophisticated.
    '''
    def __init__(self, primitives, metadata, k=10, init_grid_shape=[10, 10], base_dir="ARC/data/training",):
        self.metadata = metadata
        self.primitives = dict(primitives)
        self.k = k
        self.transformations_f = {k: eval(v) for k, v in self.primitives.items()}
        self.tf_list = sorted(self.primitives.keys())
        self.init_grid_shape = init_grid_shape
        self.base_dir = base_dir
        self.arc_files = os.listdir(base_dir)
        self.all_grids = []

        self.load_grids()

    def arc_to_numpy(self, fpath):
        with open(fpath) as f:
            content = json.load(f)

        grids = []
        for g in content["train"]:
            grids.append(np.array(g["input"], dtype="int8"))
            grids.append(np.array(g["output"], dtype="int8"))
        for g in content["test"]:
            grids.append(np.array(g["input"], dtype="int8"))
        return grids

    def load_grids(self):
        for fname in self.arc_files:
            fpath = os.path.join(self.base_dir, fname)
            self.all_grids.extend(self.arc_to_numpy(fpath))

    def augment(self, X):
        num_rotations = np.random.choice(np.arange(4))
        for _ in range(num_rotations):
            X = np.rot90(X)

        return X

    def sampleGridPatch(self, k):

        output = []

        for _ in range(k):
            width = 0
            height = 0
            while width <= self.init_grid_shape[0] or height <= self.init_grid_shape[1]:
                i = random.randint(0, len(self.all_grids) - 1)
                grid = self.all_grids[i]
                width = grid.shape[0]
                height = grid.shape[1]

            i = random.randint(0, grid.shape[0] - self.init_grid_shape[0] - 1)
            j = random.randint(0, grid.shape[1] - self.init_grid_shape[1] - 1)

            grid_sample = grid[i:i + self.init_grid_shape[0], j:j + self.init_grid_shape[1]]

            output.append(tuple(tuple(inner) for inner in grid_sample))

        return output

    def sample_transform(self):

        def most_identity(a_list, b_list):
            count_identical = 0
            for idx in range(len(a_list)):
                if a_list[idx] == b_list[idx]:
                    count_identical += 1

            if count_identical > len(a_list) / 2:
                return True
            else:
                return False

        def most_empty(a_list):
            count_empty = 0

            for a in a_list:
                if palette(a) == {0}:
                    count_empty += 1

            if count_empty > len(a_list) / 2:
                return True
            else:
                return False

        def all_valid_shape(a_list):
            for a in a_list:
                if len(a) > 30 or len(a) == 0 or len(a[0]) > 30 or len(a[0]) == 0:
                    return False

            return True

        def all_valid_colors(a_list):
            for a in a_list:
                if not palette(a).issubset(set(range(10))):
                    return False

            return True

        max_nt = self.metadata['num_nodes'][1]
        min_nt = self.metadata['num_nodes'][0]
        n = random.randint(min_nt, max_nt)

        gi = self.sampleGridPatch(self.k)

        go = []
        for g in gi:
            tmp_out = identity(g)
            go.append(tmp_out)

        prog = 'identity'
        desc = ''
        tfs = []
        for _ in range(n):
            valid = False
            while not valid:
                valid = True
                tf = random.choice(self.tf_list)

                go2 = []
                for g in go:
                    tmp_out = self.transformations_f[tf](g)
                    go2.append(tmp_out)

                if most_identity(go, go2):
                    valid = False
                    continue

                if most_empty(go2):
                    valid = False
                    continue

                if not all_valid_shape(go2):
                    valid = False
                    continue

                if not all_valid_colors(go2):
                    valid = False
                    continue

            prog = f'compose({self.primitives[tf]}, {prog})'
            desc = f'{tf}({desc})'
            go = go2
            tfs.append(tf)

        x = tok.tokenize_grid_batch(gi)
        y = tok.tokenize_grid_batch(go)

        return x, y, desc

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        S = {}
        S['xs'], S['ys'], S['task_desc'] = self.sample_transform()

        # TODO: BUG: this is wrong, the query set must use the same program as the support set!
        S['xq'], S['yq'], _ = self.sample_transform()

        return S

class ARCGymPatchesDataset(Dataset):
    '''
    This is like ARCGymDataset, but instead of randomly pixelized grids, it tries to produce more interesting &
    realistic grids by sampling square patches of the requested dimension from the ARC training set.
    '''

    def __init__(self, task_list, modules, metadata, k=5, grid_dim=5, base_dir="ARC/data/training", augment_data=True):
        self.task_list = task_list
        self.k = k
        self.metadata = metadata
        self.modules = modules
        self.base_dir = base_dir
        self.grid_dim = grid_dim
        self.arc_files = os.listdir(base_dir)
        self.augment_data = augment_data
        self.all_grids = []

        self.load_grids()

    def arc_to_numpy(self, fpath):
        with open(fpath) as f:
            content = json.load(f)

        grids = []
        for g in content["train"]:
            grids.append(np.array(g["input"], dtype="int8"))
            grids.append(np.array(g["output"], dtype="int8"))
        for g in content["test"]:
            grids.append(np.array(g["input"], dtype="int8"))
        return grids

    def load_grids(self):
        for fname in self.arc_files:
            fpath = os.path.join(self.base_dir, fname)
            self.all_grids.extend(self.arc_to_numpy(fpath))

    def augment(self, X):
        num_rotations = np.random.choice(np.arange(4))
        for _ in range(num_rotations):
            X = np.rot90(X)

        return X

    def sampleGridPatch(self):

        min_side = 0
        while min_side < self.grid_dim + 1:
            i = random.randint(0, len(self.all_grids) - 1)
            grid = self.all_grids[i]
            min_side = min(grid.shape[0], grid.shape[1])
        i = random.randint(0, grid.shape[0] - self.grid_dim - 1)
        j = random.randint(0, grid.shape[1] - self.grid_dim - 1)
        grid_sample = grid[i:i + self.grid_dim, j:j + self.grid_dim]

        return grid_sample


    def generateTaskSamples(self, G, k=3):

        data_x = []
        data_y = []
        for _ in range(k):
            # generate X
            X = self.sampleGridPatch()

            if self.augment_data:
                X = self.augment(X)

            # execute computational graph to obtain Y
            Y = graphUtils.executeCompGraph(G, X, self.modules)

            data_x.append(X)
            data_y.append(Y)

        return np.copy(data_x), np.copy(data_y)

    def __len__(self):
        return len(self.task_list)

    def __getitem__(self, idx):
        current_graph = self.task_list[idx]

        S = {}
        S['xs'], S['ys'] = self.generateTaskSamples(current_graph, self.k)
        S['xq'], S['yq'] = self.generateTaskSamples(current_graph, self.k)
        S['task_desc'] = graphUtils.get_desc(current_graph[1], self.modules)

        return S


class ARCGymDataset(Dataset):

    def __init__(self, task_list, modules, metadata, k=5, grid_size=5, augment_data=False):
        self.task_list = task_list
        self.modules = modules
        self.metadata = metadata
        self.grid_size = grid_size
        self.k = k
        self.augment_data = augment_data

    def __len__(self):
        return len(self.task_list)

    def generateGrid(self, G, metadata):
        def get_module_list(ts, modules):
            used_modules = []
            for edge in ts:
                used_modules.append(modules[edge[1]-1]['name'])

            return used_modules

        def get_swapped_colors(name):
            tokens = re.split(r'[(),]', name)
            return tokens[1], tokens[2]

        def inverse_color_lookup(col_name):
            for key, name in COLOR_MAP.items():
                if name == col_name:
                    return key

        def pick_colors(num_px, required_colors):
            color_list = []
            for rc in required_colors:
                color_list.append(np.random.choice(list(rc)))

            for _ in range(len(color_list), num_px):
                color_list.append(np.random.choice(np.arange(1, 10)))

            return color_list

        def normalize_probabilities(probs):
            prob_sum = sum(probs)
            if prob_sum == 0:
                return [0] * len(probs)  # Avoid division by zero
            else:
                return [p / prob_sum for p in probs]

        X = np.zeros((self.grid_size, self.grid_size))

        num_px_range = np.arange(metadata['num_pixels'][0], metadata['num_pixels'][1]+1)
        num_px = np.random.choice(num_px_range)

        # generate all possible pixel positions in the grid
        pixel_list = []
        spatial_dist = []

        space_dist_x = metadata['space_dist_x']
        space_dist_y = metadata['space_dist_y']
        eps = 1e-5
        if np.sum(space_dist_x) < 1.0 - eps or np.sum(space_dist_x) > 1.0 + eps:
            print("WARNING: spatial distribution probabilities for X axis did not sum up to exactly 1, normalizing probabilities.")
            space_dist_x = normalize_probabilities(space_dist_x)
            print("\tNew x axis distribution: ", space_dist_x)

        if np.sum(space_dist_y) < 1.0 - eps or np.sum(space_dist_y) > 1.0 + eps:
            print("WARNING: spatial distribution probabilities for Y axis did not sum up to exactly 1, normalizing probabilities.")
            space_dist_y = normalize_probabilities(space_dist_y)
            print("\tNew y axis distribution: ", space_dist_y)

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                prob = space_dist_x[x] * space_dist_y[(self.grid_size - y - 1)]
                spatial_dist.append(prob)
                pixel_list.append((y, x))

        pixel_list = np.array(pixel_list)
        pixel_indices = np.arange(len(pixel_list))

        used_modules = get_module_list(G[1], self.modules)
        required_colors = []
        for um in used_modules:
            if 'swap_pixels' in um:
                a, b = get_swapped_colors(um)
                col_a = inverse_color_lookup(a)
                col_b = inverse_color_lookup(b)
                required_colors.append((col_a, col_b))

        color_list = pick_colors(num_px, required_colors)
        tmp_indices = np.random.choice(pixel_indices, num_px, replace=False, p=spatial_dist)
        pixels = pixel_list[tmp_indices]
        for col_idx in range(num_px):
            px = pixels[col_idx]
            color = color_list[col_idx]
            X[px[0], px[1]] = color

        return X

    def augment(self, X):
        num_rotations = np.random.choice(np.arange(4))
        for _ in range(num_rotations):
            X = np.rot90(X)

        return X

    def generateTaskSamples(self, G, metadata, k=3):

        data_x = []
        data_y = []
        for _ in range(k):
            # generate X
            X = self.generateGrid(G, metadata)

            if self.augment_data:
                X = self.augment(X)

            # execute computational graph to obtain Y
            Y = graphUtils.executeCompGraph(G, X, self.modules)

            data_x.append(X)
            data_y.append(Y)

        return np.copy(data_x), np.copy(data_y)

    def __getitem__(self, idx):
        current_graph = self.task_list[idx]

        S = {}
        S['xs'], S['ys'] = self.generateTaskSamples(current_graph, self.metadata, self.k)
        S['xq'], S['yq'] = self.generateTaskSamples(current_graph, self.metadata, self.k)
        S['task_desc'] = graphUtils.get_desc(current_graph[1], self.modules)

        return S
