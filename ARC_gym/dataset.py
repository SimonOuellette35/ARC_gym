from torch.utils.data import Dataset
import ARC_gym.utils.graphs as graphUtils
import ARC_gym.utils.tokenization as tok
import re
import numpy as np
import os,json
import random
import ARC_gym.Hodel_primitives as Hodel
import math


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

# V2 uses a different primitives structure, a dictionary from primitive name to a lambda function.
class ARCGymVariableDatasetV2(Dataset):
    '''
    This builds on top of ARCGymPatchesDataset, but instead of assuming fixed square grids, we allowed for variable
    size, non-square grids. This means that tokenization must be a bit more sophisticated.
    '''
    def __init__(self, primitives, metadata, k=10, init_grid_shape=[10, 10], base_dir="ARC/data/training"):
        self.metadata = metadata
        self.primitives = primitives
        self.k = k
        self.transformations_f = {k: v for k, v in self.primitives.items()}
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

    def augment(self, grid):
        num_rotations = np.random.choice(np.arange(4))
        for _ in range(num_rotations):
            grid = Hodel.rot90(grid)

        def get_colors(grid):
            color_set = set()
            for row in grid:
                for c in row:
                    color_set.add(c)

            return list(color_set)

        # color swapping
        rnd = np.random.uniform()
        if rnd < 0.1:
            colors = get_colors(grid)
            from_color = np.random.choice(colors)
            to_color = np.random.choice(np.arange(1, 10))
            grid = Hodel.color_swap(grid, from_color, to_color)

        # symmetrize
        rnd = np.random.uniform()
        width = len(grid)
        height = len(grid[0])
        if rnd < 0.1 and width >= 3 and height >= 3:
            sym_idx = np.random.choice(np.arange(5))
            if sym_idx == 0:
                grid = Hodel.symmetrize_left_around_vertical(grid)
            elif sym_idx == 1:
                grid = Hodel.symmetrize_right_around_vertical(grid)
            elif sym_idx == 2:
                grid = Hodel.symmetrize_top_around_horizontal(grid)
            elif sym_idx == 3:
                grid = Hodel.symmetrize_bottom_around_horizontal(grid)
            else:
                grid = Hodel.symmetrize_quad(grid)

        # upscale by two (if < 15x15)
        rnd = np.random.uniform()
        width = len(grid)
        height = len(grid[0])
        if rnd < 0.1 and width < 15 and height < 15:
            up_idx = np.random.choice(np.arange(3))
            if up_idx == 0:
                grid = Hodel.upscale_vertical_by_two(grid)
            elif up_idx == 1:
                grid = Hodel.upscale_horizontal_by_two(grid)
            else:
                grid = Hodel.upscale_by_two(grid)

        # translations (with or without wrapping)
        rnd = np.random.uniform()
        if rnd < 0.15:
            xn = np.random.choice(np.arange(1, 5))
            yn = np.random.choice(np.arange(1, 5))
            xdir = np.random.choice(np.arange(2))
            ydir = np.random.choice(np.arange(2))
            wrap = np.random.choice(np.arange(2))

            if wrap == 1:
                for _ in range(xn):
                    if xdir == 0:
                        grid = Hodel.wrap_left(grid)
                    else:
                        grid = Hodel.wrap_right(grid)

                for _ in range(yn):
                    if ydir == 0:
                        grid = Hodel.wrap_up(grid)
                    else:
                        grid = Hodel.wrap_down(grid)
            else:
                for _ in range(xn):
                    if xdir == 0:
                        grid = Hodel.shift_left(grid)
                    else:
                        grid = Hodel.shift_right(grid)

                for _ in range(yn):
                    if ydir == 0:
                        grid = Hodel.shift_up(grid)
                    else:
                        grid = Hodel.shift_down(grid)

        return grid

    def generateGrid(self, width, height):

        X = np.zeros((width, height))
        num_px = np.random.choice(np.arange(1, 10))

        pixel_list = []
        for x in range(width):
            for y in range(height):
                pixel_list.append((x, y))

        pixel_list = np.array(pixel_list)
        pixel_indices = np.random.choice(len(pixel_list), num_px, replace=False)
        x_list = pixel_list[pixel_indices, 0]
        y_list = pixel_list[pixel_indices, 1]
        color_list = np.random.choice(np.arange(10), num_px)

        for i in range(num_px):
            X[x_list[i], y_list[i]] = color_list[i]

        return X

    def sampleGridPatch(self):

        rnd = np.random.uniform()

        if rnd < 0.5:
            # fully randomized grid
            width = 5
            height = 5
            grid_sample = self.generateGrid(width, height)
            grid_sample = tuple(tuple(inner) for inner in grid_sample.astype(int))
        else:
            # training set sub-grid
            i = random.randint(0, len(self.all_grids) - 1)
            grid = self.all_grids[i]

            from_width = 1
            to_width = grid.shape[0]
            from_height = 1
            to_height = grid.shape[1]

            if to_width <= 1:
                to_width = 2
            if to_height <= 1:
                to_height = 2

            width = np.random.choice(np.arange(from_width, min(to_width, 5)))
            height = np.random.choice(np.arange(from_height, min(to_height, 5)))

            i = random.randint(0, grid.shape[0] - width)
            j = random.randint(0, grid.shape[1] - height)

            grid_sample = grid[i:i + width, j:j + height]

            # add data augmentation
            grid_sample = tuple(tuple(inner) for inner in grid_sample.astype(int))
            grid_sample = self.augment(grid_sample)

        return grid_sample

    def generate_random_task(self):

        initial_grids, D, k = self.init_task_generation()

        rotation_group_used = False
        mirroring_group_used = False
        reps_used = []
        colors_used = []
        intermediate_grids = [(initial_grids, 'var0')]
        last_layer_grids = [(initial_grids, 'var0')]
        num_nodes_used = 0
        output_dict = {}    # Each key is the path of s, and each value is a list of samples of format:
                            # (input grid set, action taken, resulting heuristic value)
        for d in range(1, D+1):
            used_ll_grids = []

            # pick number of nodes at this layer
            num_nodes = int(np.random.choice(np.arange(1, math.pow(2, D-d) + 1)))
            num_nodes_used += num_nodes

            output_grids = []
            for node_idx in range(num_nodes):
                result = self.pick_and_apply_primitive(intermediate_grids, last_layer_grids, used_ll_grids,
                                                       reps_used, colors_used, rotation_group_used, mirroring_group_used)

                in_grid, out_grid, prim_name, in_paths, out_path, reps_used, rotation_group_used, mirroring_group_used = result
                output_grids.append((out_grid, out_path))

                # iterate over arguments to this node
                self.update_output_dict(in_grid, in_paths, prim_name, D - d, output_dict)

            last_layer_grids = []
            for og in output_grids:
                intermediate_grids.append(og)
                last_layer_grids.append(og)

        if len(last_layer_grids) != 1:
            print("BUG? Shouldn't the last layer output exactly 1 grid?")
            exit(-1)

        out_y = []

        grid_set = last_layer_grids[0][0]
        for grid in grid_set:
            y = tok.tokenize_grid(grid)
            out_y.append(y)

        desc = last_layer_grids[0][1]

        X, A, H = self.generate_from_path(desc, output_dict)

        return X, out_y, A, H, desc

    def sample_transform(self):
        X, Y, A, H, desc = self.generate_random_task()

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        S = {}
        S['xs'], S['ys'], S['task_desc'] = self.sample_transform()

        # TODO: BUG: this is wrong, the query set must use the same program as the support set!
        S['xq'], S['yq'], _ = self.sample_transform()

        return S

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
