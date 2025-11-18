import numpy as np
import json
import os
import ARC_gym.grid_sampling.distinct_colors as DCA

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


    def sample_by_category(self, categories, min_dim=None, max_dim=None):

        selected_cat = np.random.choice(categories)

        # Distinct colors adjacent means that objects are grouped by their uniform color. Adjacent objects
        # are separated by the fact that they are of a different color. Diagonally adjacent pixels belong
        # to the object if they are of the same color as that object.
        if selected_cat == 'distinct_colors_adjacent':
            return DCA.sample_distinct_colors_adjacent(self.training_path, min_dim, max_dim)
        elif selected_cat == 'distinct_colors_adjacent_empty':
            return DCA.sample_distinct_colors_adjacent_empty(self.training_path, min_dim, max_dim)
        if selected_cat == 'distinct_colors_adjacent_fill':
            return DCA.sample_distinct_colors_adjacent(self.training_path, min_dim, max_dim, fill_mask=True)
        elif selected_cat == 'distinct_colors_adjacent_empty_fill':
            return DCA.sample_distinct_colors_adjacent_empty(self.training_path, min_dim, max_dim, fill_mask=True)
        elif selected_cat == 'uniform_rect_noisy_bg':
            return DCA.sample_uniform_rect_noisy_bg(self.training_path, min_dim, max_dim, empty=False)
        elif selected_cat == 'window_noisy_bg':
            return DCA.sample_uniform_rect_noisy_bg(self.training_path, min_dim, max_dim, empty=True)
        elif selected_cat == 'incomplete_rectangles':
            return DCA.sample_incomplete_rectangles(self.training_path, min_dim, max_dim)
        elif selected_cat == 'incomplete_rectangles_same_shape':
            return DCA.sample_incomplete_rectangles(self.training_path, min_dim, max_dim, all_same_shape=True)
        elif selected_cat == 'incomplete_pattern_dot_plus':
            return DCA.sample_incomplete_pattern(self.training_path, min_dim, max_dim, pattern='dot_plus')
        elif selected_cat == 'incomplete_pattern_dot_x':
            return DCA.sample_incomplete_pattern(self.training_path, min_dim, max_dim, pattern='dot_x')
        elif selected_cat == 'incomplete_pattern_plus_hollow':
            return DCA.sample_incomplete_pattern(self.training_path, min_dim, max_dim, pattern='plus_hollow')
        elif selected_cat == 'incomplete_pattern_x_hollow':
            return DCA.sample_incomplete_pattern(self.training_path, min_dim, max_dim, pattern='x_hollow')
        elif selected_cat == 'incomplete_pattern_plus_filled':
            return DCA.sample_incomplete_pattern(self.training_path, min_dim, max_dim, pattern='plus_filled')
        elif selected_cat == 'incomplete_pattern_x_filled':
            return DCA.sample_incomplete_pattern(self.training_path, min_dim, max_dim, pattern='x_filled')
        elif selected_cat == 'incomplete_pattern_square_hollow':
            return DCA.sample_incomplete_pattern(self.training_path, min_dim, max_dim, pattern='square_hollow')
        elif selected_cat == 'incomplete_pattern_square_filled':
            return DCA.sample_incomplete_pattern(self.training_path, min_dim, max_dim, pattern='square_filled')
        elif selected_cat == 'corner_objects':
            return DCA.sample_corner_objects(self.training_path, min_dim, max_dim)
        elif selected_cat == 'fixed_size_2col_shapes3x3':
            return DCA.sample_fixed_size_2col_shapes(self.training_path, min_dim, max_dim, obj_dim=3)
        elif selected_cat == 'fixed_size_2col_shapes4x4':
            return DCA.sample_fixed_size_2col_shapes(self.training_path, min_dim, max_dim, obj_dim=4)
        elif selected_cat == 'fixed_size_2col_shapes5x5':
            return DCA.sample_fixed_size_2col_shapes(self.training_path, min_dim, max_dim, obj_dim=5)
        elif selected_cat == 'fixed_size_2col_shapes3x3_bb':
            return DCA.sample_fixed_size_2col_shapes(self.training_path, min_dim, max_dim, obj_dim=3, obj_bg_param=0)
        elif selected_cat == 'fixed_size_2col_shapes4x4_bb':
            return DCA.sample_fixed_size_2col_shapes(self.training_path, min_dim, max_dim, obj_dim=4, obj_bg_param=0)
        elif selected_cat == 'fixed_size_2col_shapes5x5_bb':
            return DCA.sample_fixed_size_2col_shapes(self.training_path, min_dim, max_dim, obj_dim=5, obj_bg_param=0)
        elif selected_cat == 'four_corners':
            return DCA.sample_four_corners(self.training_path, min_dim, max_dim)
        elif selected_cat == 'inner_color_borders':
            return DCA.sample_inner_color_borders(self.training_path, 6, 8)
        elif selected_cat == 'single_object':
            return DCA.sample_single_object(self.training_path)
        elif selected_cat == 'single_object_noisy_bg':
            return DCA.sample_uniform_rect_noisy_bg(self.training_path, num_objects=1)
        elif selected_cat == 'simple_filled_rectangles':
            return DCA.sample_simple_filled_rectangles(self.training_path, min_dim, max_dim)
        elif selected_cat == 'non_symmetrical_shapes':
            return DCA.sample_non_symmetrical_shapes(self.training_path, min_dim, max_dim)
        
    def sample(self, bg_color=None, min_dim=None, max_dim=None, force_square=False, monochrome_grid_ok=True):
        rnd = np.random.uniform()

        if rnd < self.grid_type_ratio:
            return self.uniform_random_sample(bg_color, min_dim, max_dim, force_square, monochrome_grid_ok)
        else:
            return self.training_set_crop(bg_color, min_dim, max_dim, force_square, monochrome_grid_ok)
