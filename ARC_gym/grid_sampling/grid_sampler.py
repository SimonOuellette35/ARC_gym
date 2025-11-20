import numpy as np
import json
import os
import ARC_gym.grid_sampling.object_grid_generation as OGG

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

    def sample_inside_croppable_grids(self, min_dim, max_dim):
        while True:
            if np.random.uniform() < 0.2:
                # Generate random dimensions between min_dim and max_dim (inclusive)
                rows = np.random.randint(min_dim, max_dim + 1)
                cols = np.random.randint(min_dim, max_dim + 1)
                grid = np.random.randint(0, 10, size=(rows, cols), dtype=np.int8)
            else:
                grid = self.training_set_crop(min_dim=min_dim, max_dim=max_dim)
                cols = grid.shape[1]
                rows = grid.shape[0]

            # Determine the crop margin based on the minimum grid dimension
            min_margin = min(rows, cols) // 2  # max possible margin
            if min_margin >= 3:
                margin = 3
            elif min_margin >= 2:
                margin = 2
            else:
                margin = 1

            # Only crop if the resulting grid still has positive dimension
            if rows > 2 * margin and cols > 2 * margin:
                inside_grid = grid[margin:rows - margin, margin:cols - margin]
            else:
                inside_grid = grid

            if np.all(inside_grid == 0):
                continue

            return grid, inside_grid

    def sample_shearable_grids(self, min_dim, max_dim):
        valid = False
        while not valid:
            width = np.random.randint(min_dim, max_dim + 1)
            height = np.random.randint(min_dim, max_dim + 1)
            bg_color = 0
            input_grid = [[bg_color for _ in range(width)] for _ in range(height)]

            # Select type of generation
            gen_type = np.random.choice(['full', 'randomized', 'hollow'])
            
            # Select a section that's at least 5 pixels high
            min_height = 5
            max_height = height
            section_height = np.random.randint(min_height, max_height + 1)
            start_y = np.random.randint(0, height - section_height + 1)
            
            # Select width of section
            min_width = 3
            max_width = width
            section_width = np.random.randint(min_width, max_width + 1)
            start_x = np.random.randint(0, width - section_width + 1)
            
            # Select color (not background)
            color = np.random.choice([c for c in range(10) if c != bg_color])
            
            if gen_type == 'full':
                # Fill entire section with same color
                for y in range(start_y, start_y + section_height):
                    for x in range(start_x, start_x + section_width):
                        input_grid[y][x] = color
                        
            elif gen_type == 'randomized':
                # Fill section with random colors
                for y in range(start_y, start_y + section_height):
                    for x in range(start_x, start_x + section_width):
                        input_grid[y][x] = np.random.choice([c for c in range(10) if c != bg_color])
                        
            else:  # hollow
                # Create outline only
                for y in range(start_y, start_y + section_height):
                    for x in range(start_x, start_x + section_width):
                        if (y == start_y or y == start_y + section_height - 1 or 
                            x == start_x or x == start_x + section_width - 1):
                            input_grid[y][x] = color
                        else:
                            input_grid[y][x] = bg_color

            # Check if input grid has at least 10 non-zero pixels
            cells_int = np.array([list(row) for row in input_grid]).astype(int)
            if np.sum(cells_int > 0) < 10:
                continue

            valid = True

        return input_grid, None


    def sample_croppable_corners_grids(self, min_dim, max_dim):
        while True:
            if np.random.uniform() < 0.2:
                # Generate random dimensions between min_dim and max_dim (inclusive)
                rows = np.random.randint(min_dim, max_dim + 1)
                cols = np.random.randint(min_dim, max_dim + 1)
                grid = np.random.randint(0, 10, size=(rows, cols), dtype=np.int8)
            else:
                grid = self.training_set_crop(min_dim=min_dim, max_dim=max_dim)
                cols = grid.shape[1]
                rows = grid.shape[0]

            # Get 2x2 corner subgrids
            tl_2x2 = grid[0:2, 0:2]
            tr_2x2 = grid[0:2, -2:] if cols >= 2 else grid[0:2, 0:2]
            bl_2x2 = grid[-2:, 0:2] if rows >= 2 else grid[0:2, 0:2]
            br_2x2 = grid[-2:, -2:] if rows >= 2 and cols >= 2 else grid[0:2, 0:2]

            # Get 3x3 corner subgrids (handle edge cases for small grids)
            tl_3x3 = grid[0:min(3, rows), 0:min(3, cols)]
            tr_3x3 = grid[0:min(3, rows), max(0, cols-3):cols]
            bl_3x3 = grid[max(0, rows-3):rows, 0:min(3, cols)]
            br_3x3 = grid[max(0, rows-3):rows, max(0, cols-3):cols]

            # flatten for easier comparison
            corners_2x2 = [
                tuple(tl_2x2.flatten()),
                tuple(tr_2x2.flatten()),
                tuple(bl_2x2.flatten()),
                tuple(br_2x2.flatten())
            ]
            corners_3x3 = [
                tuple(tl_3x3.flatten()),
                tuple(tr_3x3.flatten()),
                tuple(bl_3x3.flatten()),
                tuple(br_3x3.flatten())
            ]

            # Check all 2x2 AND 3x3 corner subgrids are distinct
            if len(set(corners_2x2)) == 4 and len(set(corners_3x3)) == 4:
                return grid, None

    def sample_min_count_grids(self, min_dim, max_dim):
        valid = False
        while not valid:
            input_grid = self.uniform_random_sample(min_dim=min_dim, max_dim=max_dim, monochrome_grid_ok = False)

            # Count occurrences of each color in the input grid
            color_counts = {}
            for row in input_grid:
                for color in row:
                    color_counts[color] = color_counts.get(color, 0) + 1

            # Exclude color 0 from being the minimum-count color
            # Find the minimum count among colors that are NOT 0
            nonzero_colors = [c for c in color_counts if c != 0]
            if not nonzero_colors:
                continue  # try again if all colors are 0

            min_count = min(color_counts[c] for c in nonzero_colors)

            # Check if minimum is unique and the color with min count is not 0
            colors_with_min = [c for c in nonzero_colors if color_counts[c] == min_count]
            valid = len(colors_with_min) == 1

        return input_grid, None


    def sample_max_count_grids(self, min_dim, max_dim):

        valid = False
        while not valid:
            input_grid = self.uniform_random_sample(min_dim=min_dim, max_dim=max_dim, monochrome_grid_ok = False)
            # Count occurrences of each color in the input grid
            color_counts = {}
            for row in input_grid:
                for color in row:
                    color_counts[color] = color_counts.get(color, 0) + 1
            
            # Find the maximum count
            max_count = max(color_counts.values())
            
            # Check if maximum is unique by counting how many colors have this count
            colors_with_max = sum(1 for count in color_counts.values() if count == max_count)
            valid = colors_with_max == 1
        
        return input_grid, None

    def sample_count_and_draw_grids(self, bg_color):
        # Generate a random square grid with dimension between 3x3 and 6x6 and fill with background color
        dim = np.random.randint(4, 7)
        input_grid = np.full((dim, dim), bg_color)

        # Pick a random foreground color between 0 and 9, excluding bg_color
        possible_colors = [c for c in range(10) if c != bg_color]
        fg_color = np.random.choice(possible_colors)

        # Fill a random number of cells (at least 1, at most 5) with fg_color
        num_fg = np.random.randint(1, 6)
        
        # Randomly choose num_fg unique positions in the grid to set to fg_color
        positions = np.random.choice(9, num_fg, replace=False)
        for pos in positions:
            row, col = divmod(pos, 3)
            input_grid[row, col] = fg_color

        return input_grid, None

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


    def sample_by_category(self, categories, min_dim=None, max_dim=None, bg_color=0):

        selected_cat = np.random.choice(categories)

        # Distinct colors adjacent means that objects are grouped by their uniform color. Adjacent objects
        # are separated by the fact that they are of a different color. Diagonally adjacent pixels belong
        # to the object if they are of the same color as that object.
        if selected_cat == 'distinct_colors_adjacent':
            return OGG.sample_distinct_colors_adjacent(self.training_path, min_dim, max_dim)
        elif selected_cat == 'distinct_colors_adjacent_empty':
            return OGG.sample_distinct_colors_adjacent_empty(self.training_path, min_dim, max_dim)
        if selected_cat == 'distinct_colors_adjacent_fill':
            return OGG.sample_distinct_colors_adjacent(self.training_path, min_dim, max_dim, fill_mask=True)
        elif selected_cat == 'distinct_colors_adjacent_empty_fill':
            return OGG.sample_distinct_colors_adjacent_empty(self.training_path, min_dim, max_dim, fill_mask=True)
        elif selected_cat == 'uniform_rect_noisy_bg':
            return OGG.sample_uniform_rect_noisy_bg(self.training_path, min_dim, max_dim, empty=False)
        elif selected_cat == 'window_noisy_bg':
            return OGG.sample_uniform_rect_noisy_bg(self.training_path, min_dim, max_dim, empty=True)
        elif selected_cat == 'incomplete_rectangles':
            return OGG.sample_incomplete_rectangles(self.training_path, min_dim, max_dim)
        elif selected_cat == 'incomplete_rectangles_same_shape':
            return OGG.sample_incomplete_rectangles(self.training_path, min_dim, max_dim, all_same_shape=True)
        elif selected_cat == 'incomplete_pattern_dot_plus':
            return OGG.sample_incomplete_pattern(self.training_path, min_dim, max_dim, pattern='dot_plus')
        elif selected_cat == 'incomplete_pattern_dot_x':
            return OGG.sample_incomplete_pattern(self.training_path, min_dim, max_dim, pattern='dot_x')
        elif selected_cat == 'incomplete_pattern_plus_hollow':
            return OGG.sample_incomplete_pattern(self.training_path, min_dim, max_dim, pattern='plus_hollow')
        elif selected_cat == 'incomplete_pattern_x_hollow':
            return OGG.sample_incomplete_pattern(self.training_path, min_dim, max_dim, pattern='x_hollow')
        elif selected_cat == 'incomplete_pattern_plus_filled':
            return OGG.sample_incomplete_pattern(self.training_path, min_dim, max_dim, pattern='plus_filled')
        elif selected_cat == 'incomplete_pattern_x_filled':
            return OGG.sample_incomplete_pattern(self.training_path, min_dim, max_dim, pattern='x_filled')
        elif selected_cat == 'incomplete_pattern_square_hollow':
            return OGG.sample_incomplete_pattern(self.training_path, min_dim, max_dim, pattern='square_hollow')
        elif selected_cat == 'incomplete_pattern_square_filled':
            return OGG.sample_incomplete_pattern(self.training_path, min_dim, max_dim, pattern='square_filled')
        elif selected_cat == 'corner_objects':
            return OGG.sample_corner_objects(self.training_path, min_dim, max_dim)
        elif selected_cat == 'fixed_size_2col_shapes3x3':
            return OGG.sample_fixed_size_2col_shapes(self.training_path, min_dim, max_dim, obj_dim=3)
        elif selected_cat == 'fixed_size_2col_shapes4x4':
            return OGG.sample_fixed_size_2col_shapes(self.training_path, min_dim, max_dim, obj_dim=4)
        elif selected_cat == 'fixed_size_2col_shapes5x5':
            return OGG.sample_fixed_size_2col_shapes(self.training_path, min_dim, max_dim, obj_dim=5)
        elif selected_cat == 'fixed_size_2col_shapes3x3_bb':
            return OGG.sample_fixed_size_2col_shapes(self.training_path, min_dim, max_dim, obj_dim=3, obj_bg_param=0)
        elif selected_cat == 'fixed_size_2col_shapes4x4_bb':
            return OGG.sample_fixed_size_2col_shapes(self.training_path, min_dim, max_dim, obj_dim=4, obj_bg_param=0)
        elif selected_cat == 'fixed_size_2col_shapes5x5_bb':
            return OGG.sample_fixed_size_2col_shapes(self.training_path, min_dim, max_dim, obj_dim=5, obj_bg_param=0)
        elif selected_cat == 'four_corners':
            return OGG.sample_four_corners(self.training_path, min_dim, max_dim)
        elif selected_cat == 'inner_color_borders':
            return OGG.sample_inner_color_borders(self.training_path, 6, 8)
        elif selected_cat == 'single_object':
            return OGG.sample_single_object(self.training_path)
        elif selected_cat == 'single_object_noisy_bg':
            return OGG.sample_uniform_rect_noisy_bg(self.training_path, num_objects=1)
        elif selected_cat == 'simple_filled_rectangles':
            return OGG.sample_simple_filled_rectangles(self.training_path, min_dim, max_dim)
        elif selected_cat == 'non_symmetrical_shapes':
            return OGG.sample_non_symmetrical_shapes(self.training_path, min_dim, max_dim)
        elif selected_cat == 'min_count':
            return self.sample_min_count_grids(min_dim=3, max_dim=10)
        elif selected_cat == 'max_count':
            return self.sample_max_count_grids(min_dim=3, max_dim=10)
        elif selected_cat == 'count_and_draw':
            return self.sample_count_and_draw_grids(bg_color)
        elif selected_cat == 'croppable_corners':
            return self.sample_croppable_corners_grids(min_dim, max_dim)
        elif selected_cat == 'inside_croppable':
            return self.sample_inside_croppable_grids(min_dim, max_dim)
        elif selected_cat == 'shearable_grids':
            return self.sample_shearable_grids(min_dim=6, max_dim=20)

    def sample(self, bg_color=None, min_dim=None, max_dim=None, force_square=False, monochrome_grid_ok=True):
        rnd = np.random.uniform()

        if rnd < self.grid_type_ratio:
            return self.uniform_random_sample(bg_color, min_dim, max_dim, force_square, monochrome_grid_ok)
        else:
            return self.training_set_crop(bg_color, min_dim, max_dim, force_square, monochrome_grid_ok)
