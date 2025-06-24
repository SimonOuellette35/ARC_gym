import numpy as np
import json
import os
import random
from ARC_gym.utils.object_detector import ObjectDetector


class GridSampler:

    def __init__(self, min_dim=3, max_dim=30, training_path='ARC-AGI-2/data/training', grid_type_ratio=0.05, black_bg_prob=0.5):
        '''
        Parameters:
            @param min_dim: minimum dimension of generated grid (default: 3x3 grids)
            @param max_dim: maximum dimension of generated grid (default: 30x30 grids, as per ARC-AGI rules)
            @param training_path: path to the folder containing the ARC-AGI training tasks to sample from.
            @param grid_type_ratio: probability of sampling a purely randomized (uniform distribution) grid as opposed to a
                                    grid that is randomly cropped from an ARC-AGI training task.
            @param black_bg_prob: probability of using a black background when randomly sampling background color.
                                  Only used for grids generated via a random uniform distribution.
        '''
        self.grid_type_ratio = grid_type_ratio
        self.training_path = training_path
        self.min_dim = min_dim
        self.max_dim = max_dim
        self.black_bg_prob = black_bg_prob

        self.arc_files = os.listdir(training_path)
        self.training_grids = []
        self.load_training_grids()

    def uniform_random_sample(self, bg_color=None, min_dim=None, max_dim=None, force_square=False, monochrome_grid_ok=True):
        '''
        Parameters:
            @param force_square: True to force the generated to be square. Can be useful for some types of tasks where
                                 a non-square grid doesn't make sense.
            @param min_dim: Override default min_dim by using this value. Optional.
            @param max_dim: Override default max_dim by using this value. Optional.
            @param bg_color: If bg_color is set, will use that color for the grid background. Otherwise, will randomize
                             with higher probability of zero than the other colors (see self.black_bg_prob parameter.)
        '''
        while True:  # Keep trying until we get a non-uniform grid
            if min_dim is None:
                min_dim = self.min_dim

            if max_dim is None:
                max_dim = self.max_dim

            num_rows = np.random.randint(min_dim, max_dim + 1)

            if force_square:
                num_cols = num_rows
            else:
                num_cols = np.random.randint(min_dim, max_dim + 1)

            if bg_color is None:
                rnd = np.random.uniform()
                if rnd < self.black_bg_prob:
                    bg_color = np.random.choice(np.arange(1, 10))
                else:
                    bg_color = 0

            grid = np.ones((num_rows, num_cols), dtype=np.int8) * bg_color

            grid_type = np.random.uniform()

            available_colors = [c for c in range(10) if c != bg_color]

            # Randomly choose how many foreground pixels we're going to generate
            density = np.random.uniform(0.05, 0.3)  # Random density between 5% and 30%        
            num_fg_px = int(num_rows * num_cols * density)
            num_fg_px = max(1, min(num_fg_px, num_rows * num_cols - 1))  # Ensure at least 1 and not all positions

            # Randomly select positions for non-zero values
            nonzero_indices = np.random.choice(num_rows * num_cols, size=num_fg_px, replace=False)
            nonzero_positions = np.unravel_index(nonzero_indices, (num_rows, num_cols))

            if grid_type < 0.2 and monochrome_grid_ok:
                # monochrome grid
                # Choose a random color different from the background color    
                color = np.random.choice(available_colors)
                
                # Set the selected positions to the chosen color
                for i in range(num_fg_px):
                    grid[nonzero_positions[0][i], nonzero_positions[1][i]] = color
                    
            elif grid_type < 0.75:
                # Choose 2 to 4 colors different from the background color
                num_colors = np.random.randint(2, 5)  # Random number between 2 and 4 inclusive
                colors = np.random.choice(available_colors, size=num_colors, replace=False)
                
                # Assign random colors from the selected colors to each non-zero position
                for i in range(num_fg_px):
                    # Randomly select a color from the chosen colors
                    random_color = np.random.choice(colors)
                    # Set the selected position to the chosen color
                    grid[nonzero_positions[0][i], nonzero_positions[1][i]] = random_color

            else:
                # Assign random colors from the selected colors to each non-zero position
                for i in range(num_fg_px):
                    # Randomly select a color from the chosen colors
                    random_color = np.random.choice(available_colors)
                    # Set the selected position to the chosen color
                    grid[nonzero_positions[0][i], nonzero_positions[1][i]] = random_color

            # Check if the grid is uniform (all pixels are the same color)
            if not np.all(grid == grid[0,0]):
                return grid

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

    def load_training_grids(self):
        for fname in self.arc_files:
            fpath = os.path.join(self.training_path, fname)
            self.training_grids.extend(self.arc_to_numpy(fpath))

    def upscale_horizontal(self, grid):
        # Create a new grid with double the width
        new_width = grid.shape[1] * 2
        upscaled_grid = np.zeros((grid.shape[0], new_width), dtype=grid.dtype)
        
        # For each row, duplicate each pixel
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                pixel = grid[i, j]
                upscaled_grid[i, j*2] = pixel
                upscaled_grid[i, j*2 + 1] = pixel
                
        return upscaled_grid

    def upscale_vertical(self, grid):
        # Create a new grid with double the height
        new_height = grid.shape[0] * 2
        upscaled_grid = np.zeros((new_height, grid.shape[1]), dtype=grid.dtype)
        
        # For each column, duplicate each pixel
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                pixel = grid[i, j]
                upscaled_grid[i*2, j] = pixel
                upscaled_grid[i*2 + 1, j] = pixel
                
        return upscaled_grid

    def upscale(self, grid):
        # Create a new grid with double the height and width
        new_height = grid.shape[0] * 2
        new_width = grid.shape[1] * 2
        upscaled_grid = np.zeros((new_height, new_width), dtype=grid.dtype)
        
        # For each pixel, duplicate it both horizontally and vertically
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                pixel = grid[i, j]
                # Duplicate pixel in 2x2 block
                upscaled_grid[i*2, j*2] = pixel
                upscaled_grid[i*2, j*2 + 1] = pixel
                upscaled_grid[i*2 + 1, j*2] = pixel
                upscaled_grid[i*2 + 1, j*2 + 1] = pixel
                
        return upscaled_grid

    def wrap_up(self, grid):
        new_grid = np.zeros_like(grid)
        # Move all rows up by one position
        new_grid[:-1] = grid[1:]
        # Move the top row to the bottom
        new_grid[-1] = grid[0]
        return new_grid

    def wrap_down(self, grid):
        new_grid = np.zeros_like(grid)
        # Move all rows down by one position
        new_grid[1:] = grid[:-1]
        # Move the bottom row to the top
        new_grid[0] = grid[-1]
        return new_grid

    def wrap_left(self, grid):
        new_grid = np.zeros_like(grid)
        # Move all columns left by one position
        new_grid[:, :-1] = grid[:, 1:]
        # Move the rightmost column to the left
        new_grid[:, -1] = grid[:, 0]
        return new_grid

    def wrap_right(self, grid):
        new_grid = np.zeros_like(grid)
        # Move all columns right by one position
        new_grid[:, 1:] = grid[:, :-1]
        # Move the leftmost column to the right
        new_grid[:, 0] = grid[:, -1]
        return new_grid

    def shift_up(self, grid):
        new_grid = np.zeros_like(grid)
        # Move all rows up by one position, leaving bottom row as zeros
        new_grid[:-1] = grid[1:]
        return new_grid

    def shift_down(self, grid):
        new_grid = np.zeros_like(grid)
        # Move all rows down by one position, leaving top row as zeros
        new_grid[1:] = grid[:-1]
        return new_grid

    def shift_left(self, grid):
        new_grid = np.zeros_like(grid)
        # Move all columns left by one position, leaving rightmost column as zeros
        new_grid[:, :-1] = grid[:, 1:]
        return new_grid

    def shift_right(self, grid):
        new_grid = np.zeros_like(grid)
        # Move all columns right by one position, leaving leftmost column as zeros
        new_grid[:, 1:] = grid[:, :-1]
        return new_grid

    def augment(self, grid):
        num_rotations = np.random.choice(np.arange(4))
        for _ in range(num_rotations):
            grid = np.rot90(grid, k=-1)  # k=-1 rotates 90 degrees clockwise

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
            to_color = np.random.choice([c for c in range(1, 10) if c != from_color])

            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    if grid[i][j] == from_color:
                        grid[i][j] = to_color
                    elif grid[i][j] == to_color:
                        grid[i][j] = from_color

        # # upscale by two (if < 15x15)
        # rnd = np.random.uniform()
        # width = len(grid)
        # height = len(grid[0])
        # if rnd < 0.1 and width < 15 and height < 15:
        #     up_idx = np.random.choice(np.arange(3))
        #     if up_idx == 0:
        #         grid = self.upscale_horizontal(grid)
        #     elif up_idx == 1:
        #         grid = self.upscale_vertical(grid)
        #     else:
        #         grid = self.upscale(grid)

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
                        grid = self.wrap_left(grid)
                    else:
                        grid = self.wrap_right(grid)

                for _ in range(yn):
                    if ydir == 0:
                        grid = self.wrap_up(grid)
                    else:
                        grid = self.wrap_down(grid)
            else:
                for _ in range(xn):
                    if xdir == 0:
                        grid = self.shift_left(grid)
                    else:
                        grid = self.shift_right(grid)

                for _ in range(yn):
                    if ydir == 0:
                        grid = self.shift_up(grid)
                    else:
                        grid = self.shift_down(grid)

        return grid

    def training_set_crop(self, bg_color=None, min_dim=None, max_dim=None, force_square=False, monochrome_grid_ok=True):
        '''
        Parameters:
            @param force_square: True to force the generated to be square. Can be useful for some types of tasks where
                                 a non-square grid doesn't make sense.
            @param min_dim: Override default min_dim by using this value. Optional.
            @param max_dim: Override default max_dim by using this value. Optional.
        '''        
        if min_dim is None:
            min_dim = self.min_dim

        if max_dim is None:
            max_dim = self.max_dim

        # training set sub-grid
        valid_sample = False
        while not valid_sample:
            grid_idx = np.random.randint(0, len(self.training_grids) - 1)
            grid = self.training_grids[grid_idx]

            if grid.shape[0] < min_dim or grid.shape[1] < min_dim:
                continue

            num_rows = np.random.randint(min_dim, max_dim + 1)
            num_rows = min(num_rows, grid.shape[0])

            if force_square:
                num_cols = num_rows

                if num_cols > grid.shape[1]:
                    num_cols = num_rows = grid.shape[1]
            else:
                num_cols = np.random.randint(min_dim, max_dim + 1)
                num_cols = min(num_cols, grid.shape[1])

            i = np.random.randint(0, (grid.shape[0] - num_rows) + 1)
            j = np.random.randint(0, (grid.shape[1] - num_cols) + 1)

            grid_sample = grid[i:i + num_rows, j:j + num_cols]

            # If it's just a uniform grid sample, it's not interesting. Skip.
            if np.all(grid_sample == grid_sample[0,0]):
                continue

            if not monochrome_grid_ok:
                if len(np.unique(grid_sample)) <= 2:
                    continue

            if bg_color is not None:
                if bg_color not in np.unique(grid_sample):
                    continue

            valid_sample = True

        # add data augmentation
        return self.augment(grid_sample)

    def sample_distinct_colors_adjacent_training(self):
        training_examples = [
            ('4347f46a', 0),
            ('e74e1818', 0),
            ('50cb2852', 0),
            ('009d5c81', 0),
            ('025d127b', 0),
            ('03560426', 0),
            ('05f2a901', 2),
            ('0bb8deee', 0),
            ('11dc524f', 0),
            ('18447a8d', 0),
            ('184a9768', 0),
            ('1caeab9d', 0),
            ('2753e76c', 0),
            ('28bf18c6', 0),
            ('2a5f8217', 0),
            ('342ae2ed', 0),
            ('37d3e8b2', 2),
            ('39a8645d', 0),
            ('423a55dc', 2),
            ('4364c1c4', 2),
            ('4be741c5', 0),
            ('52364a65', 2),
            ('63613498', 0),
            ('72ca375d', 0),
            ('94be5b80', 1),
            ('98cf29f8', 2),
            ('9bebae7a', 0),
            ('a09f6c25', 2),
            ('a3325580', 0),
            ('d56f2372', 0),
            ('dce56571', 0),
            ('df978a02', 0),
            ('e21a174a', 0),
            ('e41c6fd3', 0)
        ]

        selected_example = random.choice(training_examples)

        # load the example from file.
        task_id = selected_example[0]
        json_path = '%s/%s.json' % (self.training_path, task_id)
        
        with open(json_path, 'r') as f:
            task_data = json.load(f)

        # sample from the grids, based on whether we use input or output or both.
        possible_grids = []
        grids_to_use = selected_example[1]

        tmp_list = task_data["train"]
        tmp_list.extend(task_data["test"])

        if grids_to_use == 0:
            for item in tmp_list:
                possible_grids.append(item["input"])
        elif grids_to_use == 1:
            for item in tmp_list:
                possible_grids.append(item["output"])
        else:
            for item in tmp_list:
                possible_grids.append(item["input"])
                possible_grids.append(item["output"])

        grid = random.choice(possible_grids)
        grid = np.array(grid, dtype=np.int8)

        # run the hand-crafted heuristic to extract the object mask
        if task_id == 'e74e1818':
            object_mask = ObjectDetector.get_objects(grid, 'distinct_colors')
        else:
            object_mask = ObjectDetector.get_objects(grid, 'distinct_colors_adjacent')

        return grid, object_mask

    def sample_distinct_colors_adjacent(self, min_dim=None, max_dim=None):
        if min_dim is None:
            min_dim = self.min_dim

        if max_dim is None:
            max_dim = self.max_dim

        a = np.random.uniform()

        if a < 0.25:
            return self.sample_distinct_colors_adjacent_training()

        # Generate grid dimensions
        num_rows = np.random.randint(min_dim, max_dim + 1)
        num_cols = np.random.randint(min_dim, max_dim + 1)

        # Generate background color (50% chance for 0, 50% for 1-9)
        if np.random.random() < 0.5:
            bg_color = 0
        else:
            bg_color = np.random.randint(1, 10)

        # Initialize grid with background color
        grid = np.full((num_rows, num_cols), bg_color)
        
        # Initialize object mask (0 for background, positive integers for objects)
        object_mask = np.zeros((num_rows, num_cols), dtype=int)

        # Generate 1 to 9 objects
        num_objects = np.random.randint(1, 10)
        object_colors = []
        
        # Generate unique colors for objects (different from background)
        available_colors = list(range(10))
        available_colors.remove(bg_color)
        object_colors = np.random.choice(available_colors, num_objects, replace=False)

        for obj_idx in range(num_objects):
            obj_color = object_colors[obj_idx]
            obj_id = obj_idx + 1  # Object IDs start from 1
            
            # 20% chance for rectangle, 80% for random shape
            if np.random.random() < 0.2:
                # Rectangle object
                max_obj_height = max(1, num_rows // num_objects)
                max_obj_width = max(1, num_cols // num_objects)
                
                obj_height = np.random.randint(1, max_obj_height + 1)
                obj_width = np.random.randint(1, max_obj_width + 1)
                
                # Random position for rectangle
                start_row = np.random.randint(0, num_rows - obj_height + 1)
                start_col = np.random.randint(0, num_cols - obj_width + 1)
                
                # Place rectangle
                grid[start_row:start_row + obj_height, start_col:start_col + obj_width] = obj_color
                object_mask[start_row:start_row + obj_height, start_col:start_col + obj_width] = obj_id
                
            else:
                # Random shape object (diagonally adjacent pixels)
                max_obj_size = max(1, (num_rows * num_cols) // (num_objects * 4))  # Ensure objects fit
                obj_size = np.random.randint(1, max_obj_size + 1)
                
                # Find a starting position
                start_row = np.random.randint(0, num_rows)
                start_col = np.random.randint(0, num_cols)
                
                # Generate random shape using flood fill approach
                visited = set()
                queue = [(start_row, start_col)]
                pixels_placed = 0
                
                while queue and pixels_placed < obj_size:
                    row, col = queue.pop(0)
                    
                    if (row, col) in visited or row < 0 or row >= num_rows or col < 0 or col >= num_cols:
                        continue
                        
                    if grid[row, col] != bg_color:  # Already occupied
                        continue
                        
                    visited.add((row, col))
                    grid[row, col] = obj_color
                    object_mask[row, col] = obj_id
                    pixels_placed += 1
                    
                    # Add adjacent positions (including diagonal)
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            new_row, new_col = row + dr, col + dc
                            if (new_row, new_col) not in visited:
                                queue.append((new_row, new_col))

        return grid, object_mask

    def sample_by_category(self, categories, min_dim=None, max_dim=None):

        selected_cat = np.random.choice(categories)

        if selected_cat == 'distinct_colors_adjacent':
            return self.sample_distinct_colors_adjacent(min_dim, max_dim)

    def sample(self, bg_color=None, min_dim=None, max_dim=None, force_square=False, monochrome_grid_ok=True):
        rnd = np.random.uniform()

        if rnd < self.grid_type_ratio:
            return self.uniform_random_sample(bg_color, min_dim, max_dim, force_square, monochrome_grid_ok)
        else:
            return self.training_set_crop(bg_color, min_dim, max_dim, force_square, monochrome_grid_ok)
