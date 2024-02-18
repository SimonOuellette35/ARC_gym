from torch.utils.data import Dataset
import ARC_gym.utils.graphs as graphUtils
import re
import numpy as np

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
        S['unrolled_adj_mat'] = graphUtils.get_unrolled_adj_mat(current_graph[1], len(self.modules))

        return S
