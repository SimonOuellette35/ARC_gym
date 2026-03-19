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
        
        # Resolve training_path relative to ARC_gym package root if it's a relative path
        if not os.path.isabs(training_path):
            # Get the directory of this file (grid_sampler.py)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Go up to ARC_gym package root (from grid_sampling/ to ARC_gym/)
            package_root = os.path.dirname(os.path.dirname(current_dir))
            # Construct absolute path
            training_path = os.path.join(package_root, training_path)
        
        self.training_path = training_path
        self.min_dim = min_dim
        self.max_dim = max_dim
        self.black_bg_prob = black_bg_prob

        self.arc_files = os.listdir(training_path)
        self.training_grids = []
        self.load_training_grids()

    def uniform_random_sample(self, bg_color=None, min_dim=None, max_dim=None, force_square=False, monochrome_grid_ok=True, colors_present=None):
        '''
        Parameters:
            @param force_square: True to force the generated to be square. Can be useful for some types of tasks where
                                 a non-square grid doesn't make sense.
            @param min_dim: Override default min_dim by using this value. Optional.
            @param max_dim: Override default max_dim by using this value. Optional.
            @param bg_color: If bg_color is set, will use that color for the grid background. Otherwise, will randomize
                             with higher probability of zero than the other colors (see self.black_bg_prob parameter.)
        '''
        while True:  # Keep trying until we get a non-uniform grid that (if requested) contains colors_present
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

            # If colors_present is set, ensure we use its colors (if possible) in the grid
            if colors_present is not None:
                required_fg_colors = [c for c in colors_present if c != bg_color]
                # Remove duplicates and make sure they exist in available colors
                required_fg_colors = [c for c in set(required_fg_colors) if c in available_colors]
                if len(required_fg_colors) == 0:
                    # fallback: just randomize
                    colors_to_use = available_colors
                else:
                    colors_to_use = required_fg_colors.copy()
                    # Maybe supplement with additional random colors if needed
                    rest = [c for c in available_colors if c not in colors_to_use]
                    # Use up to 4 unique colors if grid_type < 0.75 case below
            else:
                colors_to_use = available_colors

            # Randomly choose how many foreground pixels we're going to generate
            density = np.random.uniform(0.05, 0.3)  # Random density between 5% and 30%        
            num_fg_px = int(num_rows * num_cols * density)
            num_fg_px = max(1, min(num_fg_px, num_rows * num_cols - 1))  # Ensure at least 1 and not all positions

            # Randomly select positions for non-zero values
            nonzero_indices = np.random.choice(num_rows * num_cols, size=num_fg_px, replace=False)
            nonzero_positions = np.unravel_index(nonzero_indices, (num_rows, num_cols))

            if grid_type < 0.2 and monochrome_grid_ok:
                # monochrome grid
                if colors_present is not None and len(required_fg_colors) > 0:
                    # Pick a required color
                    color = np.random.choice(required_fg_colors)
                else:
                    color = np.random.choice(available_colors)
                
                for i in range(num_fg_px):
                    grid[nonzero_positions[0][i], nonzero_positions[1][i]] = color

            elif grid_type < 0.75:
                # Choose between 2 to 4 foreground colors, but ensure all required_fg_colors are included
                min_needed = max(2, len(colors_to_use))
                num_colors = np.random.randint(min_needed, max(5, min_needed+1))
                # Select the colors to use
                if colors_present is not None and len(required_fg_colors) > 0:
                    # Always include all required_fg_colors
                    pool = [c for c in available_colors if c not in required_fg_colors]
                    n_extra = max(num_colors - len(required_fg_colors), 0)
                    extra = []
                    if len(pool) > 0 and n_extra > 0:
                        extra = np.random.choice(pool, size=min(n_extra, len(pool)), replace=False)
                        if not isinstance(extra, (list, np.ndarray)):
                            extra = [extra]
                    # guaranteed all required_fg_colors are used
                    colors = np.array(list(required_fg_colors) + list(extra))
                else:
                    colors = np.random.choice(available_colors, size=num_colors, replace=False)

                # Assign colors randomly to foreground pixels, but guarantee all required_fg_colors appear
                if colors_present is not None and len(required_fg_colors) > 0 and num_fg_px >= len(required_fg_colors):
                    indices = list(range(num_fg_px))
                    np.random.shuffle(indices)
                    # Assign one required_fg_color to the first len(required_fg_colors) pixels
                    for idx, color in zip(indices[:len(required_fg_colors)], required_fg_colors):
                        grid[nonzero_positions[0][idx], nonzero_positions[1][idx]] = color
                    # The remainder, assign randomly
                    for i in indices[len(required_fg_colors):]:
                        grid[nonzero_positions[0][i], nonzero_positions[1][i]] = np.random.choice(colors)
                else:
                    # Just assign randomly
                    for i in range(num_fg_px):
                        random_color = np.random.choice(colors)
                        grid[nonzero_positions[0][i], nonzero_positions[1][i]] = random_color
            else:
                # Assign random colors from available_colors to each non-zero position,
                # but ensure all required_fg_colors are assigned to at least one pixel if possible
                if colors_present is not None and len(required_fg_colors) > 0 and num_fg_px >= len(required_fg_colors):
                    indices = list(range(num_fg_px))
                    np.random.shuffle(indices)
                    # Guarantee all required_fg_colors
                    for idx, color in zip(indices[:len(required_fg_colors)], required_fg_colors):
                        grid[nonzero_positions[0][idx], nonzero_positions[1][idx]] = color
                    for i in indices[len(required_fg_colors):]:
                        random_color = np.random.choice(available_colors)
                        grid[nonzero_positions[0][i], nonzero_positions[1][i]] = random_color
                else:
                    # Just assign randomly
                    for i in range(num_fg_px):
                        random_color = np.random.choice(available_colors)
                        grid[nonzero_positions[0][i], nonzero_positions[1][i]] = random_color

            # Check if the grid is uniform (all pixels are the same color)
            if not np.all(grid == grid[0,0]):
                if colors_present is not None:
                    present_colors = set(np.unique(grid))
                    # Ensure colors_present are in the grid
                    if not all(color in present_colors for color in colors_present):
                        continue
                return grid

    def sample_inside_croppable_grids(self, min_dim, max_dim, colors_present=None):
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

            # Check colors_present constraint
            if colors_present is not None:
                grid_colors = set(np.unique(grid))
                if not all(color in grid_colors for color in colors_present):
                    continue

            # Initialize object mask (0 for background, positive integers for objects)
            object_mask = np.zeros((rows, cols), dtype=int)
            return grid, object_mask, None, ''

    def sample_shearable_grids(self, min_dim, max_dim, colors_present=None):
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
            available_colors = [c for c in range(10) if c != bg_color]
            if colors_present is not None:
                # Filter to use colors from colors_present if specified
                available_from_present = [c for c in colors_present if c != bg_color]
                if available_from_present:
                    available_colors = available_from_present
            
            color = np.random.choice(available_colors)
            
            if gen_type == 'full':
                # Fill entire section with same color
                for y in range(start_y, start_y + section_height):
                    for x in range(start_x, start_x + section_width):
                        input_grid[y][x] = color
                        
            elif gen_type == 'randomized':
                # Fill section with random colors
                for y in range(start_y, start_y + section_height):
                    for x in range(start_x, start_x + section_width):
                        input_grid[y][x] = np.random.choice(available_colors)
                        
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

            # Check colors_present constraint
            if colors_present is not None:
                grid_colors = set(np.unique(cells_int))
                if not all(color in grid_colors for color in colors_present):
                    # Try to add missing colors by replacing some existing pixels
                    missing_colors = [c for c in colors_present if c not in grid_colors]
                    for missing_color in missing_colors:
                        # Find a non-background pixel to replace
                        non_bg_positions = [(y, x) for y in range(height) for x in range(width) 
                                          if input_grid[y][x] != bg_color]
                        if non_bg_positions:
                            y, x = non_bg_positions[np.random.randint(len(non_bg_positions))]
                            input_grid[y][x] = missing_color

            valid = True

        # Initialize object mask (0 for background, positive integers for objects)
        object_mask = np.zeros((height, width), dtype=int)
        return np.array(input_grid), object_mask, None, ''


    def sample_croppable_corners_grids(self, min_dim, max_dim, colors_present=None):
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
            corners_valid = len(set(corners_2x2)) == 4 and len(set(corners_3x3)) == 4
            
            # Check colors_present constraint
            colors_valid = True
            if colors_present is not None:
                grid_colors = set(np.unique(grid))
                colors_valid = all(color in grid_colors for color in colors_present)
            
            if corners_valid and colors_valid:
                # Initialize object mask (0 for background, positive integers for objects)
                object_mask = np.zeros((rows, cols), dtype=int)
                return grid, object_mask, None, ''

    def sample_min_count_grids(self, min_dim, max_dim, colors_present=None):
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
            
            # Check colors_present constraint
            if colors_present is not None and valid:
                grid_colors = set(color_counts.keys())
                valid = all(color in grid_colors for color in colors_present)

        # Initialize object mask (0 for background, positive integers for objects)
        object_mask = np.zeros(input_grid.shape, dtype=int)
        return input_grid, object_mask, None, ''


    def sample_max_count_grids(self, min_dim, max_dim, colors_present=None):

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
            
            # Check colors_present constraint
            if colors_present is not None and valid:
                grid_colors = set(color_counts.keys())
                valid = all(color in grid_colors for color in colors_present)
        
        # Initialize object mask (0 for background, positive integers for objects)
        object_mask = np.zeros(input_grid.shape, dtype=int)
        return input_grid, object_mask, None, ''

    def sample_count_and_draw_grids(self, bg_color, colors_present=None):
        # Generate a random square grid with dimension between 3x3 and 6x6 and fill with background color
        dim = np.random.randint(4, 7)
        input_grid = np.full((dim, dim), bg_color)

        # Pick a random foreground color between 0 and 9, excluding bg_color
        possible_colors = [c for c in range(10) if c != bg_color]
        
        # If colors_present is specified, ensure we use colors from that list
        if colors_present is not None:
            # Filter possible colors to only include those in colors_present (excluding bg_color)
            available_fg_colors = [c for c in colors_present if c != bg_color]
            if available_fg_colors:
                fg_color = np.random.choice(available_fg_colors)
            else:
                fg_color = np.random.choice(possible_colors)
        else:
            fg_color = np.random.choice(possible_colors)

        # Fill a random number of cells (at least 1, at most 5) with fg_color
        num_fg = np.random.randint(1, 6)
        
        # Randomly choose num_fg unique positions in the grid to set to fg_color
        positions = np.random.choice(9, num_fg, replace=False)
        for pos in positions:
            row, col = divmod(pos, 3)
            input_grid[row, col] = fg_color

        # If colors_present is specified, ensure all required colors are present
        if colors_present is not None:
            grid_colors = set(np.unique(input_grid))
            missing_colors = [c for c in colors_present if c not in grid_colors]
            
            # Add missing colors by replacing some existing pixels
            for missing_color in missing_colors:
                # Find positions that are not bg_color and replace one with missing_color
                non_bg_positions = np.where(input_grid != bg_color)
                if len(non_bg_positions[0]) > 0:
                    idx = np.random.randint(len(non_bg_positions[0]))
                    input_grid[non_bg_positions[0][idx], non_bg_positions[1][idx]] = missing_color
                else:
                    # If no non-bg positions, replace a bg position
                    bg_positions = np.where(input_grid == bg_color)
                    if len(bg_positions[0]) > 0:
                        idx = np.random.randint(len(bg_positions[0]))
                        input_grid[bg_positions[0][idx], bg_positions[1][idx]] = missing_color

        # Initialize object mask (0 for background, positive integers for objects)
        object_mask = np.zeros((dim, dim), dtype=int)
        return input_grid, object_mask, None, ''

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


    def sample_by_category(self, categories, min_dim=None, max_dim=None, bg_color=0, colors_present=None):

        selected_cat = np.random.choice(categories)

        # Distinct colors adjacent means that objects are grouped by their uniform color. Adjacent objects
        # are separated by the fact that they are of a different color. Diagonally adjacent pixels belong
        # to the object if they are of the same color as that object.
        if selected_cat == 'distinct_colors_adjacent':
            return OGG.sample_distinct_colors_adjacent(self.training_path, min_dim, max_dim, colors_present=colors_present)
        elif selected_cat == 'distinct_colors_adjacent_empty':
            return OGG.sample_distinct_colors_adjacent_empty(self.training_path, min_dim, max_dim, colors_present=colors_present)
        elif selected_cat == 'distinct_colors_adjacent_fill':
            return OGG.sample_distinct_colors_adjacent(self.training_path, min_dim, max_dim, fill_mask=True, colors_present=colors_present)
        elif selected_cat == 'distinct_colors_adjacent_empty_fill':
            return OGG.sample_distinct_colors_adjacent_empty(self.training_path, min_dim, max_dim, fill_mask=True, colors_present=colors_present)
        elif selected_cat == 'uniform_rect_noisy_bg':
            return OGG.sample_uniform_rect_noisy_bg(self.training_path, min_dim, max_dim, empty=False, colors_present=colors_present)
        elif selected_cat == 'window_noisy_bg':
            return OGG.sample_uniform_rect_noisy_bg(self.training_path, min_dim, max_dim, empty=True, colors_present=colors_present)
        elif selected_cat == 'incomplete_rectangles':
            return OGG.sample_incomplete_rectangles(self.training_path, min_dim, max_dim, colors_present=colors_present)
        elif selected_cat == 'incomplete_rectangles_same_shape':
            return OGG.sample_incomplete_rectangles(self.training_path, min_dim, max_dim, all_same_shape=True, colors_present=colors_present)
        elif selected_cat == 'incomplete_pattern_dot_plus':
            return OGG.sample_incomplete_pattern(self.training_path, min_dim, max_dim, pattern='dot_plus', colors_present=colors_present)
        elif selected_cat == 'incomplete_pattern_dot_x':
            return OGG.sample_incomplete_pattern(self.training_path, min_dim, max_dim, pattern='dot_x', colors_present=colors_present)
        elif selected_cat == 'incomplete_pattern_plus_hollow':
            return OGG.sample_incomplete_pattern(self.training_path, min_dim, max_dim, pattern='plus_hollow', colors_present=colors_present)
        elif selected_cat == 'incomplete_pattern_x_hollow':
            return OGG.sample_incomplete_pattern(self.training_path, min_dim, max_dim, pattern='x_hollow', colors_present=colors_present)
        elif selected_cat == 'incomplete_pattern_plus_filled':
            return OGG.sample_incomplete_pattern(self.training_path, min_dim, max_dim, pattern='plus_filled', colors_present=colors_present)
        elif selected_cat == 'incomplete_pattern_x_filled':
            return OGG.sample_incomplete_pattern(self.training_path, min_dim, max_dim, pattern='x_filled', colors_present=colors_present)
        elif selected_cat == 'incomplete_pattern_square_hollow':
            return OGG.sample_incomplete_pattern(self.training_path, min_dim, max_dim, pattern='square_hollow', colors_present=colors_present)
        elif selected_cat == 'incomplete_pattern_square_filled':
            return OGG.sample_incomplete_pattern(self.training_path, min_dim, max_dim, pattern='square_filled', colors_present=colors_present)
        elif selected_cat == 'corner_objects':
            return OGG.sample_corner_objects(self.training_path, min_dim, max_dim, colors_present=colors_present)
        elif selected_cat == 'max_corner_objects':
            return OGG.sample_max_corner_objects(self.training_path, min_dim, max_dim, colors_present=colors_present)
        elif selected_cat == 'min_corner_objects':
            return OGG.sample_min_corner_objects(self.training_path, min_dim, max_dim, colors_present=colors_present)
        elif selected_cat == 'fixed_size_2col_shapes3x3':
            return OGG.sample_fixed_size_2col_shapes(self.training_path, min_dim, max_dim, obj_dim=3, colors_present=colors_present)
        elif selected_cat == 'fixed_size_2col_shapes4x4':
            return OGG.sample_fixed_size_2col_shapes(self.training_path, min_dim, max_dim, obj_dim=4, colors_present=colors_present)
        elif selected_cat == 'fixed_size_2col_shapes5x5':
            return OGG.sample_fixed_size_2col_shapes(self.training_path, min_dim, max_dim, obj_dim=5, colors_present=colors_present)
        elif selected_cat == 'fixed_size_2col_shapes3x3_bb':
            return OGG.sample_fixed_size_2col_shapes(self.training_path, min_dim, max_dim, obj_dim=3, obj_bg_param=0, colors_present=colors_present)
        elif selected_cat == 'fixed_size_2col_shapes4x4_bb':
            return OGG.sample_fixed_size_2col_shapes(self.training_path, min_dim, max_dim, obj_dim=4, obj_bg_param=0, colors_present=colors_present)
        elif selected_cat == 'fixed_size_2col_shapes5x5_bb':
            return OGG.sample_fixed_size_2col_shapes(self.training_path, min_dim, max_dim, obj_dim=5, obj_bg_param=0, colors_present=colors_present)
        elif selected_cat == 'four_corners':
            return OGG.sample_four_corners(self.training_path, min_dim, max_dim, colors_present=colors_present)
        elif selected_cat == 'inner_color_borders':
            return OGG.sample_inner_color_borders(self.training_path, 6, 8, colors_present=colors_present)
        elif selected_cat == 'single_object':
            return OGG.sample_single_object(self.training_path, colors_present=colors_present)
        elif selected_cat == 'single_object_noisy_bg':
            return OGG.sample_uniform_rect_noisy_bg(self.training_path, num_objects=1, colors_present=colors_present)
        elif selected_cat == 'simple_filled_rectangles':
            return OGG.sample_simple_filled_rectangles(self.training_path, min_dim, max_dim, colors_present=colors_present)
        elif selected_cat == 'non_symmetrical_shapes':
            return OGG.sample_non_symmetrical_shapes(self.training_path, min_dim, max_dim, colors_present=colors_present)
        elif selected_cat == 'min_count':
            return self.sample_min_count_grids(min_dim=3, max_dim=10, colors_present=colors_present)
        elif selected_cat == 'max_count':
            return self.sample_max_count_grids(min_dim=3, max_dim=10, colors_present=colors_present)
        elif selected_cat == 'count_and_draw':
            return self.sample_count_and_draw_grids(bg_color, colors_present=colors_present)
        elif selected_cat == 'croppable_corners':
            return self.sample_croppable_corners_grids(min_dim, max_dim, colors_present=colors_present)
        elif selected_cat == 'inside_croppable':
            return self.sample_inside_croppable_grids(min_dim, max_dim, colors_present=colors_present)
        elif selected_cat == 'shearable_grids':
            return self.sample_shearable_grids(min_dim=6, max_dim=20, colors_present=colors_present)
        elif selected_cat == 'twin_objects_h':
            return OGG.sample_twin_objects_h(self.training_path, min_dim=6, max_dim=30, colors_present=colors_present)
        elif selected_cat == 'twin_objects_v':
            return OGG.sample_twin_objects_v(self.training_path, min_dim=6, max_dim=30, colors_present=colors_present)
        elif selected_cat == 'max_inner_objs':
            return OGG.sample_max_inner_objs(self.training_path, min_dim=10, max_dim=30, colors_present=colors_present)
        elif selected_cat == 'min_inner_objs':
            return OGG.sample_min_inner_objs(self.training_path, min_dim=10, max_dim=30, colors_present=colors_present)
        elif selected_cat == 'odd_one_out_color':
            return OGG.sample_odd_one_out_color(self.training_path, min_dim=10, max_dim=30, colors_present=colors_present)        
        elif selected_cat == 'odd_one_out_width':
            return OGG.sample_odd_one_out_width(self.training_path, min_dim=10, max_dim=30, colors_present=colors_present)        
        elif selected_cat == 'odd_one_out_height':
            return OGG.sample_odd_one_out_height(self.training_path, min_dim=10, max_dim=30, colors_present=colors_present)        
        elif selected_cat == 'odd_one_out_size':
            return OGG.sample_odd_one_out_size(self.training_path, min_dim=10, max_dim=30, colors_present=colors_present)        
        elif selected_cat == 'odd_one_out_sym_h':
            return OGG.sample_odd_one_out_symmetry_h(self.training_path, min_dim=10, max_dim=30, colors_present=colors_present)        
        elif selected_cat == 'odd_one_out_sym_v':
            return OGG.sample_odd_one_out_symmetry_v(self.training_path, min_dim=10, max_dim=30, colors_present=colors_present)        
        elif selected_cat == 'odd_one_out_non_sym_h':
            return OGG.sample_odd_one_out_non_symmetry_h(self.training_path, min_dim=10, max_dim=30, colors_present=colors_present)        
        elif selected_cat == 'odd_one_out_non_sym_v':
            return OGG.sample_odd_one_out_non_symmetry_v(self.training_path, min_dim=10, max_dim=30, colors_present=colors_present)
        elif selected_cat == 'odd_one_out_subobj_count':
            return OGG.sample_odd_one_out_subobj_count(self.training_path, min_dim=10, max_dim=30, colors_present=colors_present)
        elif selected_cat == 'merge_2h':
            return self.sample_merge_task(self.training_path, min_dim=3, max_dim=15, bg_color=bg_color, colors_present=colors_present, 
                                          types=['horizontal'], num_subgrids_override=2, use_boundary=False)
        elif selected_cat == 'merge_2v':
            return self.sample_merge_task(self.training_path, min_dim=3, max_dim=15, bg_color=bg_color, colors_present=colors_present, 
                                          types=['vertical'], num_subgrids_override=2, use_boundary=False)
        elif selected_cat == 'merge_2x2':
            return self.sample_merge_task(self.training_path, min_dim=6, max_dim=15, bg_color=bg_color, colors_present=colors_present, 
                                          types=['square'], use_boundary=False)
        elif selected_cat == 'merge_3h':
            return self.sample_merge_task(self.training_path, min_dim=3, max_dim=20, bg_color=bg_color, colors_present=colors_present, 
                                          types=['horizontal'], num_subgrids_override=3, use_boundary=False)
        elif selected_cat == 'merge_3v':
            return self.sample_merge_task(self.training_path, min_dim=3, max_dim=20, bg_color=bg_color, colors_present=colors_present, 
                                          types=['vertical'], num_subgrids_override=3, use_boundary=False)
        elif selected_cat == 'merge_4h':
            return self.sample_merge_task(self.training_path, min_dim=3, max_dim=25, bg_color=bg_color, colors_present=colors_present, 
                                          types=['horizontal'], num_subgrids_override=4, use_boundary=False)
        elif selected_cat == 'merge_4v':
            return self.sample_merge_task(self.training_path, min_dim=3, max_dim=25, bg_color=bg_color, colors_present=colors_present, 
                                          types=['vertical'], num_subgrids_override=4, use_boundary=False)
        elif selected_cat == 'merge_2h_b':
            return self.sample_merge_task(self.training_path, min_dim=3, max_dim=15, bg_color=bg_color, colors_present=colors_present, 
                                          types=['horizontal'], num_subgrids_override=2, use_boundary=True)
        elif selected_cat == 'merge_2v_b':
            return self.sample_merge_task(self.training_path, min_dim=3, max_dim=15, bg_color=bg_color, colors_present=colors_present, 
                                          types=['vertical'], num_subgrids_override=2, use_boundary=True)
        elif selected_cat == 'merge_2x2_b':
            return self.sample_merge_task(self.training_path, min_dim=6, max_dim=15, bg_color=bg_color, colors_present=colors_present, 
                                          types=['square'], use_boundary=True)
        elif selected_cat == 'merge_3h_b':
            return self.sample_merge_task(self.training_path, min_dim=3, max_dim=20, bg_color=bg_color, colors_present=colors_present, 
                                          types=['horizontal'], num_subgrids_override=3, use_boundary=True)
        elif selected_cat == 'merge_3v_b':
            return self.sample_merge_task(self.training_path, min_dim=3, max_dim=20, bg_color=bg_color, colors_present=colors_present, 
                                          types=['vertical'], num_subgrids_override=3, use_boundary=True)
        elif selected_cat == 'merge_4h_b':
            return self.sample_merge_task(self.training_path, min_dim=3, max_dim=25, bg_color=bg_color, colors_present=colors_present, 
                                          types=['horizontal'], num_subgrids_override=4, use_boundary=True)
        elif selected_cat == 'merge_4v_b':
            return self.sample_merge_task(self.training_path, min_dim=3, max_dim=25, bg_color=bg_color, colors_present=colors_present, 
                                          types=['vertical'], num_subgrids_override=4, use_boundary=True)
        elif selected_cat == 'basic':
            return self.sample(bg_color, min_dim, max_dim, colors_present=colors_present)
        else:
            print(f"Invalid category {selected_cat}")
    
    def sample_merge_task(self, training_path, min_dim, max_dim, bg_color, colors_present, types=None, num_subgrids_override=None,
                          use_boundary=None):
        if min_dim is None:
            min_dim = 3

        if max_dim is None:
            max_dim = 30

        # Orientation: horizontal stripes, vertical stripes, or 2x2 blocks
        if types is None:
            orientation = np.random.choice(["horizontal", "vertical", "square"])
        else:
            orientation = np.random.choice(types)

        # For horizontal/vertical layouts, choose 2–4 sub-grids; for square we always have 4
        if orientation in ["horizontal", "vertical"]:
            num_subgrids = np.random.randint(2, 5)
        else:  # square 2x2 layout
            num_subgrids = 4

        if num_subgrids_override is not None:
            num_subgrids = num_subgrids_override

        # Decide whether there will be a boundary between sub-grids
        has_boundary = np.random.rand() < 0.5
        if use_boundary is not None:
            has_boundary = use_boundary

        boundary_thickness = 1 if has_boundary else 0

        # Determine grid dimensions and sub-grid extents so that
        # every sub-grid has equal size.
        if orientation == "horizontal":
            # All stripes share the same height (= num_rows) and width (= sub_w),
            # with optional 1-column boundaries between them.
            # num_cols = num_subgrids * sub_w + boundary_thickness * (num_subgrids - 1)
            # Each sub-grid must be at least 3 columns wide and at most 6.
            min_interior = max(3 * num_subgrids, min_dim - boundary_thickness * (num_subgrids - 1))
            # Cap each sub-grid width to at most 6:
            # interior_width / num_subgrids <= 6  => interior_width <= 6 * num_subgrids
            max_interior = max_dim - boundary_thickness * (num_subgrids - 1)
            max_interior = min(max_interior, 6 * num_subgrids)
            if max_interior < min_interior:
                max_interior = min_interior
            interior_width = np.random.randint(min_interior, max_interior + 1)
            # Force interior_width divisible by num_subgrids
            interior_width = interior_width - (interior_width % num_subgrids)
            if interior_width == 0:
                interior_width = num_subgrids
            sub_w = interior_width // num_subgrids
            num_cols = interior_width + boundary_thickness * (num_subgrids - 1)

            # Enforce a horizontally elongated grid: width >= height (and typically >).
            base_min_rows = min_dim
            base_max_rows = min(max_dim, num_cols)
            if base_max_rows < base_min_rows:
                base_max_rows = base_min_rows
            if base_max_rows > base_min_rows:
                num_rows = np.random.randint(base_min_rows, base_max_rows + 1)
            else:
                num_rows = base_min_rows

            num_rows_total, num_cols_total = num_rows, num_cols

        elif orientation == "vertical":
            # All stripes share the same width (= num_cols) and height (= sub_h),
            # with optional 1-row boundaries between them.
            # num_rows = num_subgrids * sub_h + boundary_thickness * (num_subgrids - 1)
            # Each sub-grid must be at least 3 rows tall and at most 6.
            min_interior = max(3 * num_subgrids, min_dim - boundary_thickness * (num_subgrids - 1))
            # Cap each sub-grid height to at most 6:
            # interior_height / num_subgrids <= 6  => interior_height <= 6 * num_subgrids
            max_interior = max_dim - boundary_thickness * (num_subgrids - 1)
            max_interior = min(max_interior, 6 * num_subgrids)
            if max_interior < min_interior:
                max_interior = min_interior
            interior_height = np.random.randint(min_interior, max_interior + 1)
            # Force interior_height divisible by num_subgrids
            interior_height = interior_height - (interior_height % num_subgrids)
            if interior_height == 0:
                interior_height = num_subgrids
            sub_h = interior_height // num_subgrids
            num_rows = interior_height + boundary_thickness * (num_subgrids - 1)

            # Enforce a vertically elongated grid: height >= width (and typically >).
            base_min_cols = min_dim
            base_max_cols = min(max_dim, num_rows)
            if base_max_cols < base_min_cols:
                base_max_cols = base_min_cols
            if base_max_cols > base_min_cols:
                num_cols = np.random.randint(base_min_cols, base_max_cols + 1)
            else:
                num_cols = base_min_cols

            num_rows_total, num_cols_total = num_rows, num_cols

        else:  # square 2x2 arrangement
            # 2x2 equal sub-grids. The side length is chosen so that:
            # - each sub-grid is between 3x3 and 6x6
            # - the overall grid is square and has odd dimensions.
            boundary = boundary_thickness
            # Side length (before enforcing odd) is 2 * sub + boundary.
            # After possibly adding +1 to make it odd, we require side <= max_dim.
            max_sub_from_dim = (max_dim - boundary - 1) // 2
            max_sub = min(6, max_sub_from_dim)
            # Minimum sub-grid side length is 3, but fall back gracefully if impossible.
            min_sub = 3
            if max_sub < min_sub:
                min_sub = 1
            sub = np.random.randint(min_sub, max_sub + 1)
            sub_h = sub
            sub_w = sub
            side = 2 * sub + boundary
            # When there is no boundary, make the grid exactly 2*sub by 2*sub
            # so that all rows/columns are part of the four sub-grids.
            # For boundary > 0, keeping the existing side is fine.
            num_rows = side
            num_cols = side

            num_rows_total, num_cols_total = num_rows, num_cols

        # Initialize grid and object mask
        grid = np.full((num_rows_total, num_cols_total), bg_color, dtype=np.int8)
        object_mask = np.zeros((num_rows_total, num_cols_total), dtype=int)

        # Decide foreground colors for each sub-grid
        all_colors = list(range(10))
        available_fg_colors = [c for c in all_colors if c != bg_color]

        subgrid_colors = []
        boundary_color = None
        if colors_present is not None:
            required = set(colors_present)
            # Colors that still need to appear as foreground (not satisfied by background)
            needed_fg = [c for c in required if c != bg_color]

            # We cannot have more distinct foreground colors than sub-grids
            if len(needed_fg) > num_subgrids:
                needed_fg = list(np.random.choice(needed_fg, size=num_subgrids, replace=False))

            # Start by assigning each needed color to its own sub-grid (as much as possible)
            for c in needed_fg:
                subgrid_colors.append(c)

            # Fill the remaining sub-grids
            while len(subgrid_colors) < num_subgrids:
                if not has_boundary:
                    # Prefer distinct colors between sub-grids when no boundary
                    remaining = [c for c in available_fg_colors if c not in subgrid_colors]
                    if remaining:
                        subgrid_colors.append(np.random.choice(remaining))
                    else:
                        subgrid_colors.append(np.random.choice(available_fg_colors))
                else:
                    subgrid_colors.append(np.random.choice(available_fg_colors))
        else:
            if has_boundary:
                # No constraints and boundaries: allow repetition
                subgrid_colors = list(np.random.choice(available_fg_colors, size=num_subgrids, replace=True))
            else:
                # No boundaries: try to give each sub-grid a distinct color when possible
                if len(available_fg_colors) >= num_subgrids:
                    subgrid_colors = list(np.random.choice(available_fg_colors, size=num_subgrids, replace=False))
                else:
                    # Not enough distinct colors; use all distinct ones then repeat if needed
                    chosen = list(np.random.choice(available_fg_colors, size=len(available_fg_colors), replace=False))
                    while len(chosen) < num_subgrids:
                        chosen.append(np.random.choice(available_fg_colors))
                    subgrid_colors = chosen[:num_subgrids]

        # Shuffle to avoid positional bias
        subgrid_colors = list(subgrid_colors)
        np.random.shuffle(subgrid_colors)

        # Choose boundary color if needed: must be distinct from background
        # and from all sub-grid foreground colors.
        if has_boundary:
            forbidden = {bg_color} | set(subgrid_colors)
            allowed = [c for c in all_colors if c not in forbidden]
            if allowed:
                boundary_color = int(np.random.choice(allowed))
            else:
                # Fallback: pick any non-background color (very unlikely path).
                boundary_color = int(np.random.choice([c for c in all_colors if c != bg_color]))

        # Fill sub-grids and object mask with random foreground pixels
        if orientation == "horizontal":
            col = 0
            for idx in range(num_subgrids):
                color = subgrid_colors[idx]
                obj_id = idx + 1
                h = num_rows_total
                w = sub_w
                area = h * w
                # Mark the entire sub-grid (including its background pixels) as one object
                object_mask[:, col : col + sub_w] = obj_id
                # Random number of foreground pixels in this sub-grid (at least 1)
                density = np.random.uniform(0.1, 0.6)
                num_fg = max(1, int(area * density))
                num_fg = min(num_fg, area)
                flat_indices = np.random.choice(area, size=num_fg, replace=False)
                rows = flat_indices // w
                cols = flat_indices % w
                # Map to global coordinates
                rows_global = rows
                cols_global = cols + col
                grid[rows_global, cols_global] = color
                col += sub_w
                if has_boundary and idx < num_subgrids - 1:
                    # Paint vertical boundary column with boundary_color (mask remains 0)
                    grid[:, col : col + boundary_thickness] = boundary_color
                    col += boundary_thickness

        elif orientation == "vertical":
            row = 0
            for idx in range(num_subgrids):
                color = subgrid_colors[idx]
                obj_id = idx + 1
                h = sub_h
                w = num_cols_total
                area = h * w
                # Mark the entire sub-grid (including its background pixels) as one object
                object_mask[row : row + sub_h, :] = obj_id
                # Random number of foreground pixels in this sub-grid (at least 1)
                density = np.random.uniform(0.1, 0.6)
                num_fg = max(1, int(area * density))
                num_fg = min(num_fg, area)
                flat_indices = np.random.choice(area, size=num_fg, replace=False)
                rows = flat_indices // w
                cols = flat_indices % w
                # Map to global coordinates
                rows_global = rows + row
                cols_global = cols
                grid[rows_global, cols_global] = color
                row += sub_h
                if has_boundary and idx < num_subgrids - 1:
                    # Paint horizontal boundary row with boundary_color (mask remains 0)
                    grid[row : row + boundary_thickness, :] = boundary_color
                    row += boundary_thickness

        else:  # square 2x2
            # Define row/col ranges from sub_h/sub_w and optional central boundary
            r0 = 0
            r1 = sub_h
            if has_boundary:
                r2 = r1 + boundary_thickness
                r3 = r2 + sub_h
            else:
                r2 = r1
                r3 = r2 + sub_h

            c0 = 0
            c1 = sub_w
            if has_boundary:
                c2 = c1 + boundary_thickness
                c3 = c2 + sub_w
            else:
                c2 = c1
                c3 = c2 + sub_w

            # Top-left sub-grid
            object_mask[r0:r1, c0:c1] = 1
            h = r1 - r0
            w = c1 - c0
            area = h * w
            density = np.random.uniform(0.1, 0.6)
            num_fg = max(1, int(area * density))
            num_fg = min(num_fg, area)
            flat_indices = np.random.choice(area, size=num_fg, replace=False)
            rows = flat_indices // w
            cols = flat_indices % w
            grid[r0 + rows, c0 + cols] = subgrid_colors[0]

            # Top-right sub-grid
            object_mask[r0:r1, c2:c3] = 2
            h = r1 - r0
            w = c3 - c2
            area = h * w
            density = np.random.uniform(0.1, 0.6)
            num_fg = max(1, int(area * density))
            num_fg = min(num_fg, area)
            flat_indices = np.random.choice(area, size=num_fg, replace=False)
            rows = flat_indices // w
            cols = flat_indices % w
            grid[r0 + rows, c2 + cols] = subgrid_colors[1]

            # Bottom-left sub-grid
            object_mask[r2:r3, c0:c1] = 3
            h = r3 - r2
            w = c1 - c0
            area = h * w
            density = np.random.uniform(0.1, 0.6)
            num_fg = max(1, int(area * density))
            num_fg = min(num_fg, area)
            flat_indices = np.random.choice(area, size=num_fg, replace=False)
            rows = flat_indices // w
            cols = flat_indices % w
            grid[r2 + rows, c0 + cols] = subgrid_colors[2]

            # Bottom-right sub-grid
            object_mask[r2:r3, c2:c3] = 4
            h = r3 - r2
            w = c3 - c2
            area = h * w
            density = np.random.uniform(0.1, 0.6)
            num_fg = max(1, int(area * density))
            num_fg = min(num_fg, area)
            flat_indices = np.random.choice(area, size=num_fg, replace=False)
            rows = flat_indices // w
            cols = flat_indices % w
            grid[r2 + rows, c2 + cols] = subgrid_colors[3]

            # Finally, paint the cross boundaries if present.
            if has_boundary:
                # Horizontal boundary between top and bottom halves
                grid[r1:r2, :] = boundary_color
                # Vertical boundary between left and right halves
                grid[:, c1:c2] = boundary_color

        if has_boundary:
            hint = 'split-grid-b'
        else:
            hint = 'split-grid'
        return grid, object_mask, None, hint

    
    def sample(self, bg_color=None, min_dim=None, max_dim=None, force_square=False, monochrome_grid_ok=True, colors_present=None):
        rnd = np.random.uniform()

        if rnd < self.grid_type_ratio:
            grid = self.uniform_random_sample(bg_color, min_dim, max_dim, force_square, monochrome_grid_ok, colors_present)
            # Initialize object mask (0 for background, positive integers for objects)
            object_mask = np.zeros(grid.shape, dtype=int)
            return grid, object_mask, None, ''

        else:
            if colors_present is not None:
                for _ in range(100):
                    grid = self.training_set_crop(bg_color, min_dim, max_dim, force_square, monochrome_grid_ok)
                    grid_colors = set(np.unique(grid))
                    if all(color in grid_colors for color in colors_present):
                        # Initialize object mask (0 for background, positive integers for objects)
                        object_mask = np.zeros(grid.shape, dtype=int)
                        return grid, object_mask, None, ''

                # If after 100 tries not all required colors were found, just return the last grid anyway
                # Initialize object mask (0 for background, positive integers for objects)
                object_mask = np.zeros(grid.shape, dtype=int)
                return grid, object_mask, None, ''

            else:
                grid = self.training_set_crop(bg_color, min_dim, max_dim, force_square, monochrome_grid_ok)
                # Initialize object mask (0 for background, positive integers for objects)
                object_mask = np.zeros(grid.shape, dtype=int)
                return grid, object_mask, None, ''

