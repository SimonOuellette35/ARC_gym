from ARC_gym.utils.object_detector import ObjectDetector
import random
import numpy as np
import json
import scipy.ndimage


def ensure_colors_present(available_colors, num_objects, colors_present, bg_color):
    """
    Helper function to ensure required colors are present in the generated objects.
    
    Args:
        available_colors: List of available colors (excluding bg_color)
        num_objects: Current number of objects
        colors_present: List of required colors or None
        bg_color: Background color
        
    Returns:
        tuple: (object_colors, updated_num_objects)
    """
    if colors_present is not None:
        # Ensure bg_color is present (it's already the background)
        required_fg_colors = [c for c in colors_present if c != bg_color]
        
        # Make sure we have enough objects to accommodate required colors
        if len(required_fg_colors) > num_objects:
            num_objects = len(required_fg_colors)
        
        # Start with required colors, then add random ones if needed
        object_colors = required_fg_colors.copy()
        remaining_colors = [c for c in available_colors if c not in required_fg_colors]
        
        # Add more colors if we need more objects
        if len(object_colors) < num_objects and remaining_colors:
            additional_needed = num_objects - len(object_colors)
            additional_colors = np.random.choice(remaining_colors, 
                                               min(additional_needed, len(remaining_colors)), 
                                               replace=False)
            object_colors.extend(additional_colors)
        
        object_colors = np.array(object_colors)
    else:
        object_colors = np.random.choice(available_colors, num_objects, replace=False)
    
    return object_colors, num_objects



def return_training_objects(training_examples, training_path, obj_category, crop_augment=True):
    
    while True:
        selected_example = random.choice(training_examples)

        # load the example from file.
        task_id = selected_example[0]
        json_path = '%s/%s.json' % (training_path, task_id)
        
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

        grid_idx = random.randrange(len(possible_grids))
        # for gi, gt in enumerate(possible_grids):
        #     print(f"Grid #{gi}: {gt}")

        grid = possible_grids[grid_idx]
        grid = np.array(grid, dtype=np.int8)

        # run the hand-crafted heuristic to extract the object mask
        if task_id == 'e74e1818':
            object_mask = ObjectDetector.get_objects(grid, 'distinct_colors')
        else:
            object_mask = ObjectDetector.get_objects(grid, obj_category, task_id, grid_idx)

        if object_mask is None or object_mask.shape == ():
            print(f"==> Got an object_mask None on training_examples = {training_examples}, obj_category = {obj_category}")
            continue
        
        a = np.random.uniform()
        if a < 0.3:
            # bg color augmentation
            grid, object_mask = get_bg_color_swap(grid, object_mask)

        a = np.random.uniform()
        if a < 0.5:
            # fg colors remapping
            # Randomly permute colors 0-9 for a one-to-one mapping
            color_mapping = np.random.permutation(10)

            # Remap grid colors
            grid = color_mapping[grid]

        a = np.random.uniform()
        if a < 0.75:
            return get_subgrid(grid, object_mask)
        else:
            return grid, object_mask, None

def get_bg_color_swap(grid, object_mask):
    """
    Given a grid and its object_mask, swap the background color (the most common color in grid where object_mask == 0)
    with a randomly chosen color (from 0-9, not present in any object).
    Returns the new grid and the unchanged object_mask.
    """
    # Find the most common color in grid where object_mask == 0
    bg_pixels = grid[object_mask == 0]
    if len(bg_pixels) == 0:
        # No background, return as is
        return grid, object_mask
    # Get the most frequent color among background pixels
    vals, counts = np.unique(bg_pixels, return_counts=True)
    bg_color = vals[np.argmax(counts)]

    # Find all colors used by objects (object_mask > 0)
    object_colors = np.unique(grid[object_mask > 0])

    # Find all possible colors (0-9) not used by any object and not the current bg_color
    possible_colors = [c for c in range(10) if c not in object_colors and c != bg_color]
    if not possible_colors:
        # No available color to swap, return as is
        return grid, object_mask
    
    new_bg_color = np.random.choice(possible_colors)

    # Swap the background color in the grid
    new_grid = np.array(grid, copy=True)
    
    # 50% chance to only swap colors around the object, 50% to swap the empty inside as well
    a = np.random.uniform()
    if a < 0.5:
        new_grid[(object_mask == 0) & (grid == bg_color)] = new_bg_color
    else:
        new_grid[(grid == bg_color)] = new_bg_color
        
    return new_grid, object_mask

def get_subgrid(grid, object_mask):
    """
    Return a randomly selected subgrid from the grid that:
    - contains at least one object (check the corresponding positions in object_mask)
    - does NOT truncate any object (in object_mask, id 0 is the background, so only background can be truncated)
    
    Args:
        grid: The input grid
        object_mask: Object mask where 0 is background, positive integers are object IDs
        
    Returns:
        tuple: (subgrid, sub_object_mask)
    """
    np_grid = np.array(grid)
    np_mask = np.array(object_mask)
    if np_grid.shape != np_mask.shape:
        print(f"Shape mismatch: grid.shape={np_grid.shape}, object_mask.shape={np_mask.shape}")
        import sys
        sys.exit(1)


    grid_height, grid_width = grid.shape
    if grid_height < 10 or grid_width < 10:
        return grid, object_mask, None

    # Find all unique object IDs (excluding background 0)
    unique_objects = np.unique(object_mask)
    unique_objects = unique_objects[unique_objects != 0]  # Remove background
    
    if len(unique_objects) == 0:
        # No objects in the grid, return the full grid
        return grid, object_mask, None
    
    # Create a dictionary to store bounding boxes for each object
    object_bounds = {}
    for obj_id in unique_objects:
        obj_positions = np.where(object_mask == obj_id)
        min_row, max_row = np.min(obj_positions[0]), np.max(obj_positions[0])
        min_col, max_col = np.min(obj_positions[1]), np.max(obj_positions[1])
        object_bounds[obj_id] = (min_row, max_row, min_col, max_col)
    
    # Try to find a valid subgrid (with maximum attempts to avoid infinite loops)
    max_attempts = 1000
    attempts = 0
    
    while attempts < max_attempts:
        # Randomly choose subgrid dimensions (at least 1x1, at most full grid)
        sub_height = np.random.randint(5, grid_height + 1)
        sub_width = np.random.randint(5, grid_width + 1)
        
        # Randomly choose subgrid position
        max_start_row = grid_height - sub_height
        max_start_col = grid_width - sub_width
        
        if max_start_row < 0 or max_start_col < 0:
            attempts += 1
            continue
            
        start_row = np.random.randint(0, max_start_row + 1)
        start_col = np.random.randint(0, max_start_col + 1)
        
        end_row = start_row + sub_height
        end_col = start_col + sub_width
        
        # Extract the subgrid and its object mask
        sub_grid = grid[start_row:end_row, start_col:end_col]
        sub_object_mask = object_mask[start_row:end_row, start_col:end_col]
        
        # Check if subgrid contains at least one object
        if not np.any(sub_object_mask > 0):
            attempts += 1
            continue
        
        # Check if any objects are truncated
        objects_in_subgrid = np.unique(sub_object_mask)
        objects_in_subgrid = objects_in_subgrid[objects_in_subgrid != 0]
        
        truncated = False
        for obj_id in objects_in_subgrid:
            # Get the bounding box of this object in the full grid
            min_row, max_row, min_col, max_col = object_bounds[obj_id]
            
            # Check if the entire object fits within the subgrid bounds
            if (min_row < start_row or max_row >= end_row or 
                min_col < start_col or max_col >= end_col):
                truncated = True
                break
        
        if not truncated:
            # Found a valid subgrid
            return sub_grid, sub_object_mask, None
        
        attempts += 1
    
    # If no valid subgrid found after max attempts, return the full grid
    return grid, object_mask, None
    

def sample_distinct_colors_adjacent_training(training_path, fill_mask):
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
        ('e21a174a', 0),
        ('e41c6fd3', 0)
    ]

    if fill_mask:
        return return_training_objects(training_examples, training_path, 'distinct_colors_adjacent_fill')
    else:
        return return_training_objects(training_examples, training_path, 'distinct_colors_adjacent')

def sample_distinct_colors_adjacent_empty_training(training_path, fill_mask):
    training_examples = [
        # simple empty shapes
        ('025d127b', 2),
        ('a3f84088', 0),
        ('a680ac02', 1),
        ('445eab21', 0),
        ('44d8ac46', 0),
        ('84f2aca1', 0),
        ('868de0fa', 0),
        ('4347f46a', 1),
        ('8ba14f53', 0),
        ('b9630600', 0),
        ('c0f76784', 0),
        ('d5d6de2d', 0),
        ('dc2e9a9d', 2),
        ('f3e62deb', 2),
        ('fc754716', 1),
        ('00d62c1b', 0),

        # "semi-empty" shapes
        ('00dbd492', 0),
        ('0a2355a6', 2),
        ('18419cfa', 2),
        ('1c56ad9f', 0),
        ('42918530', 2),
        ('7d1f7ee8', 2),
        ('b7fb29bc', 0),
        ('d37a1ef5', 0),
        ('e7dd8335', 0)
    ]

    if fill_mask:
        return return_training_objects(training_examples, training_path, 'distinct_colors_adjacent_empty_fill')
    else:
        return return_training_objects(training_examples, training_path, 'distinct_colors_adjacent_empty')

def sample_non_symmetrical_shapes_training(training_path):
    training_examples = [
        ('025d127b', 2),
        ('05f2a901', 2),
        ('11dc524f', 2),
        ('18447a8d', 0),
        ('184a9768', 1),
        ('28bf18c6', 2),
        ('37d3e8b2', 2),
        ('423a55dc', 1),
        ('4364c1c4', 2),
        ('52364a65', 2),
        ('63613498', 2),        
        ('72ca375d', 0),
        ('9bebae7a', 2),
        ('a09f6c25', 2),
        ('a3325580', 0),
        ('d56f2372', 0),
        ('dce56571', 0),
        ('dc2e9a9d', 2),
        ('18419cfa', 0),
        ('1c56ad9f', 1),
        ('d37a1ef5', 2)
    ]

    return return_training_objects(training_examples, training_path, 'distinct_colors_adjacent_empty')

def sample_incomplete_rectangles_training(training_path):
    training_examples = [
        ('60b61512', 0),
        ('6d75e8bb', 0),
        ('8fbca751', 0),
        ('3aa6fb7a', 0)
    ]

    return return_training_objects(training_examples, training_path, 'incomplete_rectangles')

def sample_incomplete_pattern_training(training_path, pattern):
    if pattern == 'dot_plus':
        training_examples = [
            ('d364b489', 0)
        ]
        return return_training_objects(training_examples, training_path, 'pattern_dot_plus')
    
    elif pattern == 'square_hollow':
        training_examples = [
            ('9caba7c3', 0),
        ]

        return return_training_objects(training_examples, training_path, 'pattern_square_hollow')
    elif pattern == 'plus_filled':
        training_examples = [
            ('14754a24', 0)
        ]

        return return_training_objects(training_examples, training_path, 'pattern_plus_filled')

def sample_uniform_rect_noisy_bg_training(training_path):
    training_examples = [
        ('25094a63', 0),
        ('8731374e', 0)
    ]

    return return_training_objects(training_examples, training_path, 'uniform_color_noisy_bg')

def sample_four_corners_training(training_path):
    training_examples = [
        ('af902bf9', 0)
    ]

    return return_training_objects(training_examples, training_path, 'distinct_colors_adjacent_empty')

def sample_single_object_training(training_path):
    training_examples = [
        ('11852cab', 1),
        ('150deff5', 2),
        ('19bb5feb', 0),
        ('1c56ad9f', 2),
        ('1cf80156', 0),
        ('2013d3e2', 0),
        ('25d487eb', 0),
        ('2697da3f', 0),
        ('28bf18c6', 0),
        ('396d80d7', 0),
        ('4938f0c2', 2),
        ('4c5c2cf0', 2),
        ('73182012', 0),
        ('7468f01a', 0),
        ('bf32578f', 2),
        ('e40b9e2f', 2),
        ('e7dd8335', 2)
    ]

    return return_training_objects(training_examples, training_path, 'single_object')

def sample_fixed_size_2col_shapes_training(training_path, obj_dim):
    if obj_dim == 3:
        training_examples = [
            ('1c0d0a4b', 2),
            ('45737921', 2),
            ('39a8645d', 0),
            ('60b61512', 0),
            ('662c240a', 0),
            ('760b3cac', 0),
            ('e133d23d', 0),
            ('e78887d1', 2)
        ]
    elif obj_dim == 4:
        training_examples = [
            ('75b8110e', 0),
            ('94f9d214', 0),
            ('99b1bc43', 0),
            ('bbb1b8b6', 0),
            ('cf98881b', 0),
            ('e345f17b', 0)
        ]
    elif obj_dim == 5:
        training_examples = [
            ('337b420f', 0),
            ('42918530', 2),
            ('4e45f183', 2),
            ('6a11f6da', 0),
            ('ea9794b1', 0)
        ]

    # TODO: the following are grid splitting tasks, should be moved to another kind of generator
    #('3428a4f5', 0),
    #('34b99a2b', 0),
    #('506d28a5', 0),
    #('5d2a5c43', 0),
    #('6430c8c4', 0),
    #('66f2d22f', 0),

    return return_training_objects(training_examples, training_path, 'fixed_size_2col_shapes')


def sample_distinct_colors_adjacent(training_path, min_dim=None, max_dim=None, fill_mask=False, colors_present=None):
    if min_dim is None:
        min_dim = 3

    if max_dim is None:
        max_dim = 30

    a = np.random.uniform()

    if a < 0.25:
        return sample_distinct_colors_adjacent_training(training_path, fill_mask)

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
    
    # Generate unique colors for objects (different from background)
    available_colors = list(range(10))
    available_colors.remove(bg_color)
    
    # Ensure required colors are present if specified
    object_colors, num_objects = ensure_colors_present(available_colors, num_objects, colors_present, bg_color)

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
            
            # Generate random shape using flood fill approach (border only)
            visited = set()
            queue = [(start_row, start_col)]
            pixels_placed = 0
            shape_pixels = set()  # Store all pixels that would be part of the shape
            
            # First pass: determine the full shape
            while queue and pixels_placed < obj_size:
                row, col = queue.pop(0)
                
                if (row, col) in visited or row < 0 or row >= num_rows or col < 0 or col >= num_cols:
                    continue
                    
                if grid[row, col] != bg_color:  # Already occupied
                    continue
                    
                visited.add((row, col))
                shape_pixels.add((row, col))
                pixels_placed += 1
                
                # Add adjacent positions (including diagonal)
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        new_row, new_col = row + dr, col + dc
                        if (new_row, new_col) not in visited:
                            queue.append((new_row, new_col))
            
            # Second pass: only fill border pixels (pixels that have at least one neighbor not in the shape)
            for row, col in shape_pixels:
                is_border = False
                
                # Check if this pixel is on the border (has at least one neighbor not in shape)
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        neighbor_row, neighbor_col = row + dr, col + dc
                        
                        # If neighbor is outside grid bounds or not part of shape, this is a border pixel
                        if (neighbor_row < 0 or neighbor_row >= num_rows or 
                            neighbor_col < 0 or neighbor_col >= num_cols or
                            (neighbor_row, neighbor_col) not in shape_pixels):
                            is_border = True
                            break
                    if is_border:
                        break
                
                # Only fill border pixels
                if is_border:
                    grid[row, col] = obj_color
                    if fill_mask:
                        # For fill_mask=True, the object_mask should include the whole shape, not just the border
                        for r, c in shape_pixels:
                            object_mask[r, c] = obj_id
                    else:
                        object_mask[row, col] = obj_id

    return grid, object_mask, None

def count_corners(object_mask, obj_id):
    """
    Count the number of corners for a specific object.
    For each pixel in the object:
    - Count 4-connected neighbors that are part of the object
    - Count diagonal neighbors that are part of the object
    - If exactly 1 diagonal neighbor: outer corner
    - If exactly 3 diagonal neighbors and 4 4-connected neighbors: inner corner
    """
    h, w = object_mask.shape
    corners = 0
    
    obj_pixels = np.argwhere(object_mask == obj_id)
    
    for r, c in obj_pixels:
        # Check 4-connected neighbors (up, down, left, right)
        up = (r - 1 >= 0 and object_mask[r - 1, c] == obj_id)
        down = (r + 1 < h and object_mask[r + 1, c] == obj_id)
        left = (c - 1 >= 0 and object_mask[r, c - 1] == obj_id)
        right = (c + 1 < w and object_mask[r, c + 1] == obj_id)
        
        num_4_neighbors = sum([up, down, left, right])
        
        # Check diagonal neighbors (4 diagonal positions)
        up_left = (r - 1 >= 0 and c - 1 >= 0 and object_mask[r - 1, c - 1] == obj_id)
        up_right = (r - 1 >= 0 and c + 1 < w and object_mask[r - 1, c + 1] == obj_id)
        down_left = (r + 1 < h and c - 1 >= 0 and object_mask[r + 1, c - 1] == obj_id)
        down_right = (r + 1 < h and c + 1 < w and object_mask[r + 1, c + 1] == obj_id)
        
        num_diagonal_neighbors = sum([up_left, up_right, down_left, down_right])
        
        # Outer corner: exactly 1 diagonal neighbor
        if num_diagonal_neighbors == 1:
            corners += 1
        # Inner corner: exactly 3 diagonal neighbors and 4 4-connected neighbors
        elif num_diagonal_neighbors == 3 and num_4_neighbors == 4:
            corners += 1
    
    return corners

def sample_max_corner_objects(training_path, min_dim=None, max_dim=None, colors_present=None):
    while True:
        grid, object_mask, _ = sample_corner_objects(training_path, min_dim, max_dim, colors_present)
        
        # Get all unique object IDs (excluding background 0)
        unique_objects = np.unique(object_mask)
        unique_objects = unique_objects[unique_objects != 0]
        
        if len(unique_objects) <= 1:
            continue

        # Count corners for each object
        corner_counts = {}
        for obj_id in unique_objects:
            corner_counts[obj_id] = count_corners(object_mask, obj_id)

        corner_values = list(corner_counts.values())
        max_corners = max(corner_values)
        
        # Count how many objects have the maximum number of corners
        objects_with_max = sum(1 for count in corner_values if count == max_corners)
        
        # Check if exactly one object has max corners and it's strictly more than others
        if objects_with_max == 1:
            # Verify it's strictly more than all others
            other_corners = [count for count in corner_values if count < max_corners]
            if len(other_corners) == len(corner_values) - 1:
                return grid, object_mask, None


def sample_min_corner_objects(training_path, min_dim=None, max_dim=None, colors_present=None):
    while True:
        grid, object_mask, _ = sample_corner_objects(training_path, min_dim, max_dim, colors_present)
        
        # Get all unique object IDs (excluding background 0)
        unique_objects = np.unique(object_mask)
        unique_objects = unique_objects[unique_objects != 0]
        
        if len(unique_objects) <= 1:
            continue
        
        # Count corners for each object
        corner_counts = {}
        for obj_id in unique_objects:
            corner_counts[obj_id] = count_corners(object_mask, obj_id)

        corner_values = list(corner_counts.values())
        min_corners = min(corner_values)
        
        # Count how many objects have the maximum number of corners
        objects_with_min = sum(1 for count in corner_values if count == min_corners)
        
        # Check if exactly one object has max corners and it's strictly more than others
        if objects_with_min == 1:
            # Verify it's strictly more than all others
            other_corners = [count for count in corner_values if count > min_corners]
            if len(other_corners) == len(corner_values) - 1:
                return grid, object_mask, None


def sample_corner_objects(training_path, min_dim=None, max_dim=None, colors_present=None):
    if min_dim is None:
        min_dim = 16

    if max_dim is None:
        max_dim = 30

    a = np.random.uniform()

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

    # Generate 1 to 4 objects
    num_objects = np.random.randint(1, 5)
    object_colors = []
    
    # Generate unique colors for objects (different from background)
    available_colors = list(range(10))
    available_colors.remove(bg_color)
    object_colors = np.random.choice(available_colors, num_objects, replace=False)

    for obj_idx in range(num_objects):
        # For each object, generate a filled rectangle of random size (min 8x8, up to grid size), non-overlapping
        obj_color = object_colors[obj_idx]
        obj_id = obj_idx + 1  # Object IDs start from 1

        # Determine rectangle size
        min_rect_size = 6
        max_rect_height = num_rows
        max_rect_width = num_cols

        if min_rect_size >= max_rect_height // 2:
            rect_height = min_rect_size
        else:
            rect_height = np.random.randint(min_rect_size, max_rect_height // 2)
        if min_rect_size >= max_rect_width // 2:
            rect_width = min_rect_size
        else:
            rect_width = np.random.randint(min_rect_size, max_rect_width // 2)
        
        # Ensure rectangle fits within grid dimensions
        rect_height = min(rect_height, num_rows)
        rect_width = min(rect_width, num_cols)

        # Try to find a non-overlapping position for the rectangle
        found_spot = False
        max_attempts = 100
        for _ in range(max_attempts):
            top = np.random.randint(0, num_rows - rect_height + 1)
            left = np.random.randint(0, num_cols - rect_width + 1)
            # Check for overlap in object_mask (must be 0 everywhere in the rectangle)
            # Also check for overlap in grid (must be bg_color everywhere in the rectangle)
            rect_mask = object_mask[top:top+rect_height, left:left+rect_width]
            rect_grid = grid[top:top+rect_height, left:left+rect_width]
            if np.all(rect_mask == 0) and np.all(rect_grid == bg_color):
                # Place the rectangle
                grid[top:top+rect_height, left:left+rect_width] = obj_color
                object_mask[top:top+rect_height, left:left+rect_width] = obj_id
                found_spot = True
                break
            
        # If no non-overlapping spot is found, skip placing this object
        if not found_spot:
            continue

        edge_idx = np.random.choice(np.arange(5))

        if edge_idx == 0:
            if np.random.uniform() > 0.3:
                continue

            # Top edge
            # Sample a rectangle of width as before, and height between 2 and grid height - 2
            min_seg_len = 2
            max_seg_len = max(2, rect_width - 4)
            if max_seg_len < min_seg_len:
                seg_len = min_seg_len
            else:
                seg_len = np.random.randint(min_seg_len, max_seg_len + 1)
            # Rectangle height: between 2 and rect_height - 2
            min_rect_height = 2
            max_rect_height = max(2, rect_height - 2)
            if max_rect_height < min_rect_height:
                rect_seg_height = min_rect_height
            else:
                rect_seg_height = np.random.randint(min_rect_height, max_rect_height + 1)
            # Random start position along the top edge
            if left + 2 >= left + rect_width - seg_len - 1:
                continue
            start_col = np.random.randint(left + 2, left + rect_width - seg_len - 1)
            start_row = top
            # Draw the rectangle with bg_color
            grid[start_row:start_row + rect_seg_height, start_col:start_col + seg_len] = bg_color
            object_mask[start_row:start_row + rect_seg_height, start_col:start_col + seg_len] = 0
        elif edge_idx == 1:
            # Bottom edge
            # Sample a rectangle of width as before, and height between 2 and grid height - 2
            min_seg_len = 2
            max_seg_len = max(2, rect_width - 4)
            if max_seg_len < min_seg_len:
                seg_len = min_seg_len
            else:
                seg_len = np.random.randint(min_seg_len, max_seg_len + 1)
            # Rectangle height: between 2 and rect_height - 2
            min_rect_height = 2
            max_rect_height = max(2, rect_height - 4)
            if max_rect_height < min_rect_height:
                rect_seg_height = min_rect_height
            else:
                rect_seg_height = np.random.randint(min_rect_height, max_rect_height + 1)
            # Random start position along the bottom edge
            if left + 2 >= left + rect_width - seg_len - 1:
                continue
            start_col = np.random.randint(left + 2, left + rect_width - seg_len - 1)
            start_row = top + rect_height - rect_seg_height
            # Draw the rectangle with bg_color
            grid[start_row:start_row + rect_seg_height, start_col:start_col + seg_len] = bg_color
            object_mask[start_row:start_row + rect_seg_height, start_col:start_col + seg_len] = 0
        elif edge_idx == 2:
            # Left edge
            # Sample a rectangle of height as before, and width between 2 and rect_width - 2
            min_seg_len = 2
            max_seg_len = max(2, rect_height - 4)
            if max_seg_len < min_seg_len:
                seg_len = min_seg_len
            else:
                seg_len = np.random.randint(min_seg_len, max_seg_len + 1)
            # Rectangle width: between 2 and rect_width - 2
            min_rect_width = 2
            max_rect_width = max(2, rect_width - 2)
            if max_rect_width < min_rect_width:
                rect_seg_width = min_rect_width
            else:
                rect_seg_width = np.random.randint(min_rect_width, max_rect_width + 1)
            # Random start position along the left edge
            if top + 2 >= top + rect_height - seg_len - 1:
                continue
            start_row = np.random.randint(top + 2, top + rect_height - seg_len - 1)
            start_col = left
            # Draw the rectangle with bg_color
            grid[start_row:start_row + seg_len, start_col:start_col + rect_seg_width] = bg_color
            object_mask[start_row:start_row + seg_len, start_col:start_col + rect_seg_width] = 0
        elif edge_idx == 3:
            # Right edge
            # Sample a rectangle of height as before, and width between 2 and rect_width - 2
            min_seg_len = 2
            max_seg_len = max(2, rect_height - 4)
            if max_seg_len < min_seg_len:
                seg_len = min_seg_len
            else:
                seg_len = np.random.randint(min_seg_len, max_seg_len + 1)
            # Rectangle width: between 2 and rect_width - 2
            min_rect_width = 2
            max_rect_width = max(2, rect_width - 2)
            if max_rect_width < min_rect_width:
                rect_seg_width = min_rect_width
            else:
                rect_seg_width = np.random.randint(min_rect_width, max_rect_width + 1)
            # Random start position along the right edge
            if top + 2 >= top + rect_height - seg_len - 1:
                continue
            start_row = np.random.randint(top + 2, top + rect_height - seg_len - 1)
            start_col = left + rect_width - rect_seg_width
            # Draw the rectangle with bg_color
            grid[start_row:start_row + seg_len, start_col:start_col + rect_seg_width] = bg_color
            object_mask[start_row:start_row + seg_len, start_col:start_col + rect_seg_width] = 0

        # randomly hollow out the object
        if np.random.uniform() < 0.5:
            # Hollow out the object by filling inside non-edge, non-background pixels with bg_color
            # Find all non-background pixels (i.e., object pixels)
            obj_pixels = np.argwhere(object_mask > 0)
            if obj_pixels.shape[0] > 0:
                rows, cols = obj_pixels[:, 0], obj_pixels[:, 1]
                min_row, max_row = rows.min(), rows.max()
                min_col, max_col = cols.min(), cols.max()
                # Edge pixels: those on the border of the object bounding box, or with a background neighbor
                edge_mask = np.zeros_like(object_mask, dtype=bool)
                for r, c in obj_pixels:
                    # If on the bounding box edge, it's an edge pixel
                    if r == min_row or r == max_row or c == min_col or c == max_col:
                        edge_mask[r, c] = True
                    else:
                        # If any 4-neighbor is background, it's an edge pixel
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr, nc = r+dr, c+dc
                            if nr < 0 or nr >= object_mask.shape[0] or nc < 0 or nc >= object_mask.shape[1]:
                                edge_mask[r, c] = True
                                break
                            if object_mask[nr, nc] == 0:
                                edge_mask[r, c] = True
                                break

                # Fill inside (non-edge) object pixels with bg_color and set mask to 0
                inside_mask = (object_mask > 0) & (~edge_mask)
                grid[inside_mask] = bg_color

    return grid, object_mask, None


def sample_distinct_colors_adjacent_empty(training_path, min_dim=None, max_dim=None, fill_mask=False, colors_present=None):
    if min_dim is None:
        min_dim = 3

    if max_dim is None:
        max_dim = 30

    a = np.random.uniform()

    if a < 0.25:
        return sample_distinct_colors_adjacent_empty_training(training_path, fill_mask)

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

    # Generate 1 to 7 objects
    num_objects = np.random.randint(1, 8)
    
    # Generate unique colors for objects (different from background)
    available_colors = list(range(10))
    available_colors.remove(bg_color)
    
    # Ensure required colors are present if specified
    object_colors, num_objects = ensure_colors_present(available_colors, num_objects, colors_present, bg_color)

    for obj_idx in range(num_objects):
        obj_color = object_colors[obj_idx]
        obj_id = obj_idx + 1  # Object IDs start from 1

        # Rectangle object
        max_obj_height = max(3, num_rows // 3)
        max_obj_width = max(3, num_cols // 3)

        obj_height = np.random.randint(3, max_obj_height + 1)
        obj_width = np.random.randint(3, max_obj_width + 1)

        # Try to find a free spot for this object, up to N attempts
        found_spot = False
        max_attempts = 50
        for _ in range(max_attempts):
            start_row = np.random.randint(0, num_rows - obj_height + 1)
            start_col = np.random.randint(0, num_cols - obj_width + 1)

            # Check if this region overlaps with any existing object
            region = object_mask[start_row:start_row + obj_height, start_col:start_col + obj_width]
            if np.any(region != 0):
                continue  # Overlaps, try another position

            # Place rectangle: only the contour (border) of the rectangle to obj_color, leaving the interior as bg_color

            # Top and bottom rows
            grid[start_row, start_col:start_col + obj_width] = obj_color
            grid[start_row + obj_height - 1, start_col:start_col + obj_width] = obj_color

            # Left and right columns (excluding corners to avoid double-setting)
            if obj_height > 2:
                grid[start_row + 1:start_row + obj_height - 1, start_col] = obj_color
                grid[start_row + 1:start_row + obj_height - 1, start_col + obj_width - 1] = obj_color

            # Fill the entire rectangle area in the object mask (not just the border)
            if fill_mask:
                object_mask[start_row:start_row + obj_height, start_col:start_col + obj_width] = obj_id
            else:
                # Fill only the border (non-background) pixels in the object mask
                # Top and bottom rows
                object_mask[start_row, start_col:start_col + obj_width] = obj_id
                object_mask[start_row + obj_height - 1, start_col:start_col + obj_width] = obj_id

                # Left and right columns (excluding corners)
                if obj_height > 2:
                    object_mask[start_row + 1:start_row + obj_height - 1, start_col] = obj_id
                    object_mask[start_row + 1:start_row + obj_height - 1, start_col + obj_width - 1] = obj_id

            found_spot = True
            break  # Successfully placed this object

        if not found_spot:
            # No more space for this object, stop placing further objects
            break
            
    return grid, object_mask, None

def sample_single_object(training_path, min_dim=None, max_dim=None, colors_present=None):
    if min_dim is None:
        min_dim = 3

    if max_dim is None:
        max_dim = 30

    a = np.random.uniform()

    if a < 0.25:
        return sample_single_object_training(training_path)

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

    num_objects = 1
    
    # Generate unique colors for objects (different from background)
    available_colors = list(range(10))
    available_colors.remove(bg_color)
    
    # Ensure required colors are present if specified
    object_colors, num_objects = ensure_colors_present(available_colors, num_objects, colors_present, bg_color)

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

            # 50% chance for uniform color, 50% for random colors
            if np.random.random() < 0.5:
                # Uniform color
                grid[start_row:start_row + obj_height, start_col:start_col + obj_width] = obj_color
            else:
                # Random colors (not background)
                available_colors = [c for c in range(10) if c != bg_color]
                random_rect = np.random.choice(available_colors, size=(obj_height, obj_width))
                grid[start_row:start_row + obj_height, start_col:start_col + obj_width] = random_rect

            object_mask[start_row:start_row + obj_height, start_col:start_col + obj_width] = obj_id

        else:
            # Random shape object (diagonally adjacent pixels)
            max_obj_size = max(1, (num_rows * num_cols) // (num_objects * 4))  # Ensure objects fit
            obj_size = np.random.randint(1, max_obj_size + 1)

            # Find a starting position
            start_row = np.random.randint(0, num_rows)
            start_col = np.random.randint(0, num_cols)

            # Generate random shape using flood fill approach (border only)
            visited = set()
            queue = [(start_row, start_col)]
            pixels_placed = 0
            shape_pixels = set()  # Store all pixels that would be part of the shape

            # First pass: determine the full shape
            while queue and pixels_placed < obj_size:
                row, col = queue.pop(0)

                if (row, col) in visited or row < 0 or row >= num_rows or col < 0 or col >= num_cols:
                    continue

                if grid[row, col] != bg_color:  # Already occupied
                    continue

                visited.add((row, col))
                shape_pixels.add((row, col))
                pixels_placed += 1

                # Add adjacent positions (including diagonal)
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        new_row, new_col = row + dr, col + dc
                        if (new_row, new_col) not in visited:
                            queue.append((new_row, new_col))

            # 50% chance for uniform color, 50% for random colors
            if np.random.random() < 0.5:
                # Uniform color for all border pixels
                for row, col in shape_pixels:
                    is_border = False
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            neighbor_row, neighbor_col = row + dr, col + dc
                            if (neighbor_row < 0 or neighbor_row >= num_rows or 
                                neighbor_col < 0 or neighbor_col >= num_cols or
                                (neighbor_row, neighbor_col) not in shape_pixels):
                                is_border = True
                                break
                        if is_border:
                            break
                    if is_border:
                        grid[row, col] = obj_color
                        object_mask[row, col] = obj_id
            else:
                # Random color for each border pixel (not background)
                available_colors = [c for c in range(10) if c != bg_color]
                for row, col in shape_pixels:
                    is_border = False
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            neighbor_row, neighbor_col = row + dr, col + dc
                            if (neighbor_row < 0 or neighbor_row >= num_rows or 
                                neighbor_col < 0 or neighbor_col >= num_cols or
                                (neighbor_row, neighbor_col) not in shape_pixels):
                                is_border = True
                                break
                        if is_border:
                            break
                    if is_border:
                        grid[row, col] = np.random.choice(available_colors)
                        object_mask[row, col] = obj_id

    return grid, object_mask, None

def sample_simple_filled_rectangles(training_path, min_dim=None, max_dim=None, colors_present=None):
    if min_dim is None:
        min_dim = 6

    if max_dim is None:
        max_dim = 30

    # Generate grid dimensions
    num_rows = np.random.randint(min_dim, max_dim + 1)
    num_cols = np.random.randint(min_dim, max_dim + 1)

    # Generate background color (50% chance for 0, 50% for 1-9)
    if np.random.random() < 0.5:
        bg_color = 0
    else:
        bg_color = np.random.randint(1, 10)

    # Initialize grid with background color (no random pixels in background)
    grid = np.full((num_rows, num_cols), bg_color)
    
    # Initialize object mask (0 for background, positive integers for objects)
    object_mask = np.zeros((num_rows, num_cols), dtype=int)

    # Generate 1 to 6 objects
    num_objects = np.random.randint(1, 7)

    # Generate unique colors for objects (different from background)
    available_colors = list(range(10))
    available_colors.remove(bg_color)
    
    # Ensure required colors are present if specified
    object_colors, num_objects = ensure_colors_present(available_colors, num_objects, colors_present, bg_color)

    max_attempts_per_object = 50  # Prevent infinite loops if grid is crowded

    for obj_idx in range(num_objects):
        obj_color = object_colors[obj_idx]
        obj_id = obj_idx + 1  # Object IDs start from 1

        # Generate a filled rectangle of a uniform, randomly selected color.
        max_obj_height = max(3, num_rows // 3)
        max_obj_width = max(3, num_cols // 3)

        for _ in range(max_attempts_per_object):
            obj_height = np.random.randint(3, max_obj_height + 1)
            obj_width = np.random.randint(3, max_obj_width + 1)

            # Find all possible top-left positions where the rectangle fits
            possible_rows = num_rows - obj_height + 1
            possible_cols = num_cols - obj_width + 1
            if possible_rows <= 0 or possible_cols <= 0:
                continue  # Object too big for grid

            # Try a random position
            start_row = np.random.randint(0, possible_rows)
            start_col = np.random.randint(0, possible_cols)

            # Check for overlap in object_mask
            region = object_mask[start_row:start_row + obj_height, start_col:start_col + obj_width]
            if np.any(region != 0):
                continue  # Overlaps with existing object, try again

            # Full rectangle (filled)
            grid[start_row:start_row + obj_height, start_col:start_col + obj_width] = obj_color
            object_mask[start_row:start_row + obj_height, start_col:start_col + obj_width] = obj_id

            break

    return grid, object_mask, None


def sample_uniform_rect_noisy_bg(training_path, min_dim=None, max_dim=None, empty=False, num_objects=None, colors_present=None):
    if min_dim is None:
        min_dim = 6

    if max_dim is None:
        max_dim = 30

    a = np.random.uniform()
    if a < 0.05 and num_objects is None and not empty:
        return sample_uniform_rect_noisy_bg_training(training_path)
    
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
    
    # Randomly determine the sparsity: probability that a pixel is colored (not background)
    sparsity = np.random.uniform(0.25, 1.)  # 25% to 100% of pixels colored

    # For each pixel, with probability=sparsity, assign a random color (0-9)
    # Otherwise, leave as bg_color (already set)
    mask = np.random.rand(num_rows, num_cols) < sparsity
    random_colors = np.random.randint(0, 10, size=(num_rows, num_cols))
    grid[mask] = random_colors[mask]
    
    # Initialize object mask (0 for background, positive integers for objects)
    object_mask = np.zeros((num_rows, num_cols), dtype=int)

    # Generate 1 to 6 objects
    if num_objects is None:
        num_objects = np.random.randint(1, 7)
    
    # Generate unique colors for objects (different from background)
    available_colors = list(range(10))
    available_colors.remove(bg_color)
    
    # Ensure required colors are present if specified
    object_colors, num_objects = ensure_colors_present(available_colors, num_objects, colors_present, bg_color)

    max_attempts_per_object = 50  # Prevent infinite loops if grid is crowded

    for obj_idx in range(num_objects):
        obj_color = object_colors[obj_idx]
        obj_id = obj_idx + 1  # Object IDs start from 1

        # Generate a rectangle of a uniform, randomly selected color.
        # 50% chance of empty (border only) or full rectangle.
        max_obj_height = max(3, num_rows // 3)
        max_obj_width = max(3, num_cols // 3)

        for _ in range(max_attempts_per_object):
            obj_height = np.random.randint(3, max_obj_height + 1)
            obj_width = np.random.randint(3, max_obj_width + 1)

            # Find all possible top-left positions where the rectangle fits
            possible_rows = num_rows - obj_height + 1
            possible_cols = num_cols - obj_width + 1
            if possible_rows <= 0 or possible_cols <= 0:
                continue  # Object too big for grid

            # Try a random position
            start_row = np.random.randint(0, possible_rows)
            start_col = np.random.randint(0, possible_cols)

            # Check for overlap in object_mask
            region = object_mask[start_row:start_row + obj_height, start_col:start_col + obj_width]
            if np.any(region != 0):
                continue  # Overlaps with existing object, try again

            if not empty:
                # Full rectangle
                grid[start_row:start_row + obj_height, start_col:start_col + obj_width] = obj_color
                # No overlap, place the object
                object_mask[start_row:start_row + obj_height, start_col:start_col + obj_width] = obj_id
            else:
                # Empty rectangle (border only)
                # Top and bottom rows
                grid[start_row, start_col:start_col + obj_width] = obj_color
                grid[start_row + obj_height - 1, start_col:start_col + obj_width] = obj_color
                # Left and right columns (excluding corners already set)
                if obj_height > 2:
                    grid[start_row + 1:start_row + obj_height - 1, start_col] = obj_color
                    grid[start_row + 1:start_row + obj_height - 1, start_col + obj_width - 1] = obj_color

                # For empty rectangles, set the object_mask to the border (1s on border, 0s inside)
                # Top and bottom rows
                object_mask[start_row, start_col:start_col + obj_width] = obj_id
                object_mask[start_row + obj_height - 1, start_col:start_col + obj_width] = obj_id
                # Left and right columns (excluding corners already set)
                if obj_height > 2:
                    object_mask[start_row + 1:start_row + obj_height - 1, start_col] = obj_id
                    object_mask[start_row + 1:start_row + obj_height - 1, start_col + obj_width - 1] = obj_id

            break

    return grid, object_mask, None

def get_pattern(bg_color, pattern, num_patterns):

    def random_removal(pattern_grid, exclude=None, to_remove=None):
        px_count = 0
        for row in pattern_grid:
            for val in row:
                if val != bg_color:
                    px_count += 1

        # Determine how many to remove
        if to_remove is not None:
            num_to_remove = to_remove
        else:
            num_to_remove = np.random.randint(1, px_count // 2 + 1)

        # Randomly remove 'num_to_remove' non-bg_color pixels from pattern (replace with bg_color)
        # If exclude is not None, do not remove the pixel at 'exclude' (row, col)

        # Find all non-bg_color pixel coordinates
        coords = []
        for i, row in enumerate(pattern_grid):
            for j, val in enumerate(row):
                if val != bg_color:
                    if exclude is not None and (i, j) == tuple(exclude):
                        continue
                    coords.append((i, j))

        if len(coords) == 0 or num_to_remove == 0:
            return pattern_grid

        # If num_to_remove > available, just remove all
        num_to_remove = min(num_to_remove, len(coords))
        remove_coords = np.random.choice(len(coords), size=num_to_remove, replace=False)
        # Make a copy to avoid mutating input
        new_pattern = [list(r) for r in pattern_grid]
        for idx in remove_coords:
            i, j = coords[idx]
            new_pattern[i][j] = bg_color

        return new_pattern

    # dot that gets transformed into a '+'
    output_patterns = []
    output_masks = []
    for _ in range(num_patterns):
        if pattern == 'dot_plus':
            col = np.random.choice([c for c in range(10) if c != bg_color])
            pattern_grid = [
                [bg_color, bg_color, bg_color],
                [bg_color, col, bg_color],
                [bg_color, bg_color, bg_color]
            ]
            pattern_mask = [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0]
            ]
        elif pattern == 'dot_x':
            # dot that gets transformed into a 'x'
            col = np.random.choice([c for c in range(10) if c != bg_color])
            pattern_grid = [
                [bg_color, bg_color, bg_color],
                [bg_color, col, bg_color],
                [bg_color, bg_color, bg_color]
            ]
            pattern_mask = [
                [1, 0, 1],
                [0, 1, 0],
                [1, 0, 1]
            ]
        elif pattern == 'plus_hollow':
            # empty plus with only 1 random pixel removed
            col = np.random.choice([c for c in range(10) if c != bg_color])
            pattern_grid = [
                [bg_color, col, bg_color],
                [col, bg_color, col],
                [bg_color, col, bg_color]
            ]
            pattern_grid = random_removal(pattern_grid, to_remove=1)
            pattern_mask = [
                [0, 1, 0],
                [1, 0, 1],
                [0, 1, 0]
            ]
        elif pattern == 'x_hollow':
            # empty x with random removal
            col = np.random.choice([c for c in range(10) if c != bg_color])
            pattern_grid = [
                [col, bg_color, col],
                [bg_color, bg_color, bg_color],
                [col, bg_color, col]
            ]
            pattern_grid = random_removal(pattern_grid, to_remove=1)
            pattern_mask = [
                [1, 0, 1],
                [0, 0, 0],
                [1, 0, 1]
            ]
        elif pattern == 'x_filled':
            # filled x with random removal
            col = np.random.choice([c for c in range(10) if c != bg_color])
            pattern_grid = [
                [col, bg_color, col],
                [bg_color, col, bg_color],
                [col, bg_color, col]
            ]
            pattern_grid = random_removal(pattern_grid, to_remove=1)
            pattern_mask = [
                [1, 0, 1],
                [0, 1, 0],
                [1, 0, 1]
            ]
        elif pattern == 'plus_filled':
            # filled + with random removal
            col = np.random.choice([c for c in range(10) if c != bg_color])
            pattern_grid = [
                [bg_color, col, bg_color],
                [col, col, col],
                [bg_color, col, bg_color]
            ]
            pattern_grid = random_removal(pattern_grid, to_remove=1)
            pattern_mask = [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0]
            ]
        elif pattern == 'square_hollow':
            # empty square
            col = np.random.choice([c for c in range(10) if c != bg_color])
            pattern_grid = [
                [col, col, col],
                [col, bg_color, col],
                [col, col, col]
            ]
            pattern_grid = random_removal(pattern_grid)
            pattern_mask = [
                [1, 1, 1],
                [1, 0, 1],
                [1, 1, 1]
            ]
        else:
            # filled square (with random color inside)
            col1 = np.random.choice([c for c in range(10) if c != bg_color])
            col2 = np.random.choice([c for c in range(10) if c != bg_color])
            pattern_grid = [
                [col1, col1, col1],
                [col1, col2, col1],
                [col1, col1, col1]
            ]
            pattern_grid = random_removal(pattern_grid, exclude=[1, 1])   # don't remove the central dot
            pattern_mask = [
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1]
            ]

        output_patterns.append(np.array(pattern_grid))
        output_masks.append(np.array(pattern_mask))

    return output_patterns, output_masks

def sample_incomplete_pattern(training_path, min_dim=None, max_dim=None, pattern='dot_plus', colors_present=None):
    training_patterns = ['dot_plus', 'plus_filled', 'square_hollow']

    if min_dim is None:
        min_dim = 5

    if max_dim is None:
        max_dim = 30

    a = np.random.uniform()

    if a < 0.1 and pattern in training_patterns:
        return sample_incomplete_pattern_training(training_path, pattern)

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
    
    # Generate 1 to 7 objects
    num_objects = np.random.randint(1, 8)
    
    # Generate unique colors for objects (different from background)
    available_colors = list(range(10))
    available_colors.remove(bg_color)
    
    # Ensure required colors are present if specified
    object_colors, num_objects = ensure_colors_present(available_colors, num_objects, colors_present, bg_color)

    # IMPORTANT: Pass the actual grid background color to get_pattern
    # Randomly choose number of patterns to paste (1 to 6)
    num_patterns = np.random.randint(1, 7)

    pattern_grids, pattern_masks = get_pattern(bg_color, pattern, num_patterns)

    pattern_height, pattern_width = pattern_grids[0].shape
    mask_height, mask_width = pattern_masks[0].shape

    # # 50% chance to add random noise to the background
    # if np.random.random() < 0.5:
    #     # Find all colors used in all pattern_grids (excluding background)
    #     pattern_colors = set()
    #     for pg in pattern_grids:
    #         pattern_colors.update(np.unique(pg))
    #     if bg_color in pattern_colors:
    #         pattern_colors.remove(bg_color)
    #     # Also exclude all object colors (to be extra safe)
    #     pattern_colors.update(object_colors)
    #     # Choose a noise color that is not in pattern_colors or bg_color
    #     possible_noise_colors = [c for c in range(10) if c not in pattern_colors and c != bg_color]
    #     if possible_noise_colors:
    #         noise_color = np.random.choice(possible_noise_colors)
    #         # Number of noisy pixels: between 1 and 50% of the grid area
    #         num_noisy_pixels = np.random.randint(1, int(0.5 * num_rows * num_cols) + 1)
    #         # Generate random positions for the noise
    #         noisy_indices = np.unravel_index(
    #             np.random.choice(num_rows * num_cols, num_noisy_pixels, replace=False),
    #             (num_rows, num_cols)
    #         )
    #         grid[noisy_indices] = noise_color

    # To prevent overlapping, we will keep a mask of occupied cells (using pattern_mask, not pattern_grid)
    occupied_mask = np.zeros((num_rows, num_cols), dtype=bool)

    for obj_idx in range(num_patterns):
        # Use the pre-generated pattern_grid and pattern_mask for this object
        pattern_grid = pattern_grids[obj_idx]
        pattern_mask = pattern_masks[obj_idx]
        pattern_height, pattern_width = pattern_grid.shape
        mask_height, mask_width = pattern_mask.shape

        # Randomly select a color for this pattern (not background)
        obj_color = object_colors[obj_idx % len(object_colors)]

        # Find possible top-left positions where the pattern fits
        max_row = num_rows - pattern_height
        max_col = num_cols - pattern_width

        if max_row < 0 or max_col < 0:
            # Pattern doesn't fit, skip this placement
            continue

        # Find all possible top-left positions where the pattern_mask does not overlap with occupied_mask
        possible_positions = []
        for row in range(0, max_row + 1):
            for col in range(0, max_col + 1):
                # Check if any cell in the region covered by pattern_mask is already occupied
                region = occupied_mask[row:row+pattern_height, col:col+pattern_width]
                overlap = np.any(region & (pattern_mask.astype(bool)))
                if not overlap:
                    possible_positions.append((row, col))

        if not possible_positions:
            # No valid position to place this pattern without overlap
            continue

        # Randomly select one of the valid positions
        start_row, start_col = possible_positions[np.random.randint(len(possible_positions))]

        # Paste the pattern onto the grid and update object_mask and occupied_mask
        for i in range(pattern_height):
            for j in range(pattern_width):
                grid_row = start_row + i
                grid_col = start_col + j
                # Only overwrite background cells with obj_color where pattern_mask is 1 and pattern_grid is not bg_color
                if (0 <= grid_row < grid.shape[0] and 0 <= grid_col < grid.shape[1] and
                    pattern_mask[i, j] and pattern_grid[i, j] != bg_color):
                    grid[grid_row, grid_col] = obj_color

        for i in range(mask_height):
            for j in range(mask_width):
                grid_row = start_row + i
                grid_col = start_col + j
                if (0 <= grid_row < object_mask.shape[0] and
                    0 <= grid_col < object_mask.shape[1] and
                    pattern_mask[i, j]):
                    object_mask[grid_row, grid_col] = obj_idx + 1
                    occupied_mask[grid_row, grid_col] = True

    return grid, object_mask, None

def sample_fixed_size_2col_shapes(training_path, min_dim=None, max_dim=None, obj_dim=3, obj_bg_param=None, colors_present=None):
    if min_dim is None:
        min_dim = 5

    if max_dim is None:
        max_dim = 30

    a = np.random.uniform()

    if a < 0.25 and obj_bg_param is None:
        return sample_fixed_size_2col_shapes_training(training_path, obj_dim)

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

    # Generate 1 to 7 objects
    num_objects = np.random.randint(1, 8)
    
    # Generate unique colors for objects (different from background)
    available_colors = list(range(10))
    available_colors.remove(bg_color)

    # Rectangle object
    obj_height = obj_dim
    obj_width = obj_dim

    # Choose exactly 2 random colors between 0 and 9 (inclusive),
    # but if obj_bg_param is not None, one of the two colors MUST be obj_bg_param
    if obj_bg_param is not None:
        # Pick a second color different from bg_color_param
        color_choices = [c for c in range(10) if c != obj_bg_param]
        second_color = np.random.choice(color_choices)
        two_colors = np.array([obj_bg_param, second_color])
    elif colors_present is not None:
        # If colors_present is specified, try to use colors from that list
        if len(colors_present) >= 2:
            two_colors = np.array(colors_present[:2])
        else:
            # If not enough colors in colors_present, use what we have and add random ones
            remaining_colors = [c for c in range(10) if c not in colors_present]
            if remaining_colors:
                additional_color = np.random.choice(remaining_colors)
                two_colors = np.array(colors_present + [additional_color])[:2]
            else:
                two_colors = np.random.choice(range(10), 2, replace=False)
    else:
        two_colors = np.random.choice(range(10), 2, replace=False)

    for obj_idx in range(num_objects):
        obj_id = obj_idx + 1  # Object IDs start from 1

        # Try to find a free spot for this object, up to N attempts
        found_spot = False
        max_attempts = 50
        for _ in range(max_attempts):
            start_row = np.random.randint(0, num_rows - obj_height + 1)
            start_col = np.random.randint(0, num_cols - obj_width + 1)

            # Check if this region overlaps with any existing object
            region = object_mask[start_row:start_row + obj_height, start_col:start_col + obj_width]
            if np.any(region != 0):
                continue  # Overlaps, try another position

            # Fill the entire rectangle area with random colors from the two selected colors
            shape_pixels = np.random.choice(two_colors, size=(obj_height, obj_width))
            grid[start_row:start_row + obj_height, start_col:start_col + obj_width] = shape_pixels

            # Fill the entire rectangle area in the object mask
            object_mask[start_row:start_row + obj_height, start_col:start_col + obj_width] = obj_id

            found_spot = True
            break  # Successfully placed this object

        if not found_spot:
            # No more space for this object, stop placing further objects
            break
            
    return grid, object_mask, None

def sample_non_symmetrical_shapes(training_path, min_dim=None, max_dim=None, colors_present=None):
    if min_dim is None:
        min_dim = 5

    if max_dim is None:
        max_dim = 30

    a = np.random.uniform()

    if a < 0.4:
        return sample_non_symmetrical_shapes_training(training_path)

    # Helper function to check if a shape is asymmetric under 90/180/270 rotation
    def is_asymmetric(shape_mask):
        for k in [1, 2, 3]:
            if np.array_equal(shape_mask, np.rot90(shape_mask, k=k)):
                return False
        return True

    def is_8_connected(mask):
        # 8-connectivity structure
        structure = np.ones((3, 3), dtype=int)
        labeled, num = scipy.ndimage.label(mask, structure=structure)
        # Only one connected component and all 1s are in that component
        return num == 1 and np.sum(mask) == np.sum(labeled == 1)

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
    object_mask = np.zeros((num_rows, num_cols), dtype=int)

    # Generate 1 to 7 objects
    num_objects = np.random.randint(1, 8)

    # Generate unique colors for objects (different from background)
    available_colors = list(range(10))
    available_colors.remove(bg_color)
    
    # Ensure required colors are present if specified
    object_colors, num_objects = ensure_colors_present(available_colors, num_objects, colors_present, bg_color)

    min_shape_dim = 3
    max_shape_dim = 6

    for obj_idx in range(num_objects):
        obj_color = object_colors[obj_idx]
        obj_id = obj_idx + 1  # Object IDs start from 1

        found_spot = False
        max_attempts = 50
        for _ in range(max_attempts):
            # Randomly choose shape size, ensuring it fits within grid dimensions
            max_height = min(max_shape_dim, num_rows)
            max_width = min(max_shape_dim, num_cols)
            
            # Ensure minimum dimensions don't exceed grid size
            if min_shape_dim > max_height or min_shape_dim > max_width:
                break  # Skip this object if it can't fit
                
            shape_height = np.random.randint(min_shape_dim, max_height + 1)
            shape_width = np.random.randint(min_shape_dim, max_width + 1)

            # Generate a random binary mask for the shape
            for _ in range(30):  # Try up to 30 times to get an asymmetric, 8-connected shape
                mask = (np.random.rand(shape_height, shape_width) > np.random.uniform(0.3, 0.7)).astype(np.uint8)

                # Ensure at least 60% filled
                if np.sum(mask) < int(0.6 * shape_height * shape_width):
                    continue

                # Ensure the mask is a single 8-connected component and not all 1s or all 0s
                if np.sum(mask) == 0 or np.sum(mask) == mask.size:
                    continue
                if not is_8_connected(mask):
                    continue

                # Must be asymmetric under 90/180/270 rotation
                if is_asymmetric(mask):
                    break
            else:
                continue  # Could not generate asymmetric, 8-connected mask, try new position

            # Try to find a free spot for this object
            start_row = np.random.randint(0, num_rows - shape_height + 1)
            start_col = np.random.randint(0, num_cols - shape_width + 1)
            region = object_mask[start_row:start_row + shape_height, start_col:start_col + shape_width]
            if np.any((mask == 1) & (region != 0)):
                continue  # Overlaps, try another position

            # Place the shape in the grid and object mask
            grid_region = grid[start_row:start_row + shape_height, start_col:start_col + shape_width]
            object_region = object_mask[start_row:start_row + shape_height, start_col:start_col + shape_width]
            grid_region[mask == 1] = obj_color
            object_region[mask == 1] = obj_id

            found_spot = True
            break  # Successfully placed this object

        if not found_spot:
            # No more space for this object, stop placing further objects
            break

    return grid, object_mask, None


def sample_inner_color_borders(training_path, min_dim=None, max_dim=None, colors_present=None):
    if min_dim is None:
        min_dim = 6

    if max_dim is None:
        max_dim = 7

    # Generate grid dimensions
    # num_rows == num_cols, and can be either 6 or 8, randomly chosen
    size = np.random.choice([6, 8])
    num_rows = num_cols = size

    # Choose 3 distinct colors (not including the background)
    available_colors = list(range(10))
    
    if colors_present is not None:
        # If colors_present is specified, try to use colors from that list
        if len(colors_present) >= 3:
            border_colors = np.array(colors_present[:3])
        else:
            # If not enough colors in colors_present, use what we have and add random ones
            remaining_colors = [c for c in available_colors if c not in colors_present]
            needed = 3 - len(colors_present)
            if len(remaining_colors) >= needed:
                additional_colors = np.random.choice(remaining_colors, needed, replace=False)
                border_colors = np.array(list(colors_present) + list(additional_colors))
            else:
                border_colors = np.random.choice(available_colors, 3, replace=False)
    else:
        border_colors = np.random.choice(available_colors, 3, replace=False)
    
    color_a, color_b, color_c = border_colors

    # For 6x6 or 8x8 grid, set the three borders explicitly
    grid = np.full((num_rows, num_cols), 0)

    if size == 6:
        # Outermost border: color_a
        grid[0, :] = color_a
        grid[-1, :] = color_a
        grid[:, 0] = color_a
        grid[:, -1] = color_a

        # Next inner border: color_b
        grid[1, 1:-1] = color_b
        grid[-2, 1:-1] = color_b
        grid[1:-1, 1] = color_b
        grid[1:-1, -2] = color_b

        # Innermost border: color_c
        grid[2, 2:-2] = color_c
        grid[-3, 2:-2] = color_c
        grid[2:-2, 2] = color_c
        grid[2:-2, -3] = color_c

        # Center (2x2) remains bg_color

    elif size == 8:
        # Outermost border: color_a
        grid[0, :] = color_a
        grid[-1, :] = color_a
        grid[:, 0] = color_a
        grid[:, -1] = color_a

        # Next inner border: color_b
        grid[1, 1:-1] = color_b
        grid[-2, 1:-1] = color_b
        grid[1:-1, 1] = color_b
        grid[1:-1, -2] = color_b

        # Innermost border: color_c (distinct from color_a and color_b)
        grid[2, 2:-2] = color_c
        grid[-3, 2:-2] = color_c
        grid[2:-2, 2] = color_c
        grid[2:-2, -3] = color_c

        # Center (4x4) is color_a
        grid[3:5, 3:5] = color_a

    # Guarantee exactly 3 distinct colors in the grid (not counting bg_color)
    # (color_a, color_b, and color_c for 6x6; color_a and color_b for 8x8, but color_c is not used in 8x8)

    # Initialize object mask (0 for background, positive integers for objects)
    object_mask = np.zeros((num_rows, num_cols), dtype=int)

    return grid, object_mask, None

def sample_four_corners(training_path, min_dim=None, max_dim=None, colors_present=None):
    if min_dim is None:
        min_dim = 5

    if max_dim is None:
        max_dim = 30

    a = np.random.uniform()

    if a < 0.05:
       return sample_four_corners_training(training_path)

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

    # Generate 1 to 5 objects
    num_objects = np.random.randint(1, 6)
    
    # Generate unique colors for objects (different from background)
    available_colors = list(range(10))
    available_colors.remove(bg_color)
    
    # Ensure required colors are present if specified
    object_colors, num_objects = ensure_colors_present(available_colors, num_objects, colors_present, bg_color)

    # Rectangle object
    max_obj_height = max(3, num_rows // 2)
    max_obj_width = max(3, num_cols // 2)

    for obj_idx in range(num_objects):
        obj_id = obj_idx + 1  # Object IDs start from 1

        obj_height = np.random.randint(3, max_obj_height + 1)
        obj_width = np.random.randint(3, max_obj_width + 1)

        # Try to find a free spot for this object, up to N attempts
        found_spot = False
        max_attempts = 50
        for _ in range(max_attempts):
            start_row = np.random.randint(0, num_rows - obj_height + 1)
            start_col = np.random.randint(0, num_cols - obj_width + 1)

            # Check if this region overlaps with any existing object
            region = object_mask[start_row:start_row + obj_height, start_col:start_col + obj_width]
            if np.any(region != 0):
                continue  # Overlaps, try another position

            # Fill the entire rectangle area in the object mask (the full rectangle is the object)
            object_mask[start_row:start_row + obj_height, start_col:start_col + obj_width] = obj_id

            # Now, draw only the 4 corners in the grid
            # Decide if all corners are the same color or different colors
            if np.random.rand() < 0.5:
                # All corners same color
                obj_color = object_colors[obj_idx]
                grid[start_row, start_col] = obj_color  # top-left
                grid[start_row, start_col + obj_width - 1] = obj_color  # top-right
                grid[start_row + obj_height - 1, start_col] = obj_color  # bottom-left
                grid[start_row + obj_height - 1, start_col + obj_width - 1] = obj_color  # bottom-right
            else:
                # All corners different colors
                # Pick 4 unique colors (excluding bg_color and already used for this object if possible)
                available_colors = list(range(10))
                if bg_color in available_colors:
                    available_colors.remove(bg_color)
                # Remove the object's main color if possible, to maximize color diversity
                if object_colors[obj_idx] in available_colors:
                    available_colors.remove(object_colors[obj_idx])
                # If not enough colors, allow repeats
                if len(available_colors) < 4:
                    corner_colors = np.random.choice(list(range(10)), 4, replace=True)
                else:
                    corner_colors = np.random.choice(available_colors, 4, replace=False)
                grid[start_row, start_col] = corner_colors[0]  # top-left
                grid[start_row, start_col + obj_width - 1] = corner_colors[1]  # top-right
                grid[start_row + obj_height - 1, start_col] = corner_colors[2]  # bottom-left
                grid[start_row + obj_height - 1, start_col + obj_width - 1] = corner_colors[3]  # bottom-right

            found_spot = True
            break  # Successfully placed this object

        if not found_spot:
            # No more space for this object, stop placing further objects
            break
            
    return grid, object_mask, None
   
def sample_odd_one_out_width(training_path, min_dim=None, max_dim=None, colors_present=None):
    if min_dim is None:
        min_dim = 5

    if max_dim is None:
        max_dim = 30

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
    object_mask = np.zeros((num_rows, num_cols), dtype=int)

    # 3 to 7 objects; any color except bg; all same width except one (odd one out)
    num_objects = np.random.randint(3, 8)
    available_colors = [c for c in range(10) if c != bg_color]
    odd_out_idx = np.random.randint(0, num_objects)

    max_obj_height = max(3, num_rows // 2)
    max_obj_width = max(3, num_cols // 2)
    common_width = np.random.randint(3, max_obj_width + 1)
    other_widths = [w for w in range(3, max_obj_width + 1) if w != common_width]
    odd_width = int(np.random.choice(other_widths)) if other_widths else common_width

    for obj_idx in range(num_objects):
        obj_color = int(np.random.choice(available_colors))
        obj_width = odd_width if obj_idx == odd_out_idx else common_width
        obj_id = obj_idx + 1
        shape_type = np.random.choice(['filled_rect', 'empty_rect'])
        forbidden_mask = scipy.ndimage.binary_dilation(object_mask != 0, structure=np.ones((3, 3)))

        found_spot = False
        for _ in range(50):
            obj_height = np.random.randint(3, max_obj_height + 1)
            if num_rows < obj_height or num_cols < obj_width:
                continue
            start_row = np.random.randint(0, num_rows - obj_height + 1)
            start_col = np.random.randint(0, num_cols - obj_width + 1)
            region = forbidden_mask[start_row:start_row + obj_height, start_col:start_col + obj_width]
            if np.any(region):
                continue

            if shape_type == 'filled_rect':
                grid[start_row:start_row + obj_height, start_col:start_col + obj_width] = obj_color
                object_mask[start_row:start_row + obj_height, start_col:start_col + obj_width] = obj_id
            else:  # empty_rect
                grid[start_row, start_col:start_col + obj_width] = obj_color
                grid[start_row + obj_height - 1, start_col:start_col + obj_width] = obj_color
                if obj_height > 2:
                    grid[start_row + 1:start_row + obj_height - 1, start_col] = obj_color
                    grid[start_row + 1:start_row + obj_height - 1, start_col + obj_width - 1] = obj_color
                object_mask[start_row, start_col:start_col + obj_width] = obj_id
                object_mask[start_row + obj_height - 1, start_col:start_col + obj_width] = obj_id
                if obj_height > 2:
                    object_mask[start_row + 1:start_row + obj_height - 1, start_col] = obj_id
                    object_mask[start_row + 1:start_row + obj_height - 1, start_col + obj_width - 1] = obj_id
            found_spot = True
            break

        if not found_spot:
            break

    return grid, object_mask, None

def sample_odd_one_out_height(training_path, min_dim=None, max_dim=None, colors_present=None):
    if min_dim is None:
        min_dim = 5

    if max_dim is None:
        max_dim = 30

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
    object_mask = np.zeros((num_rows, num_cols), dtype=int)

    # 3 to 7 objects; any color except bg; all same height except one (odd one out)
    num_objects = np.random.randint(3, 8)
    available_colors = [c for c in range(10) if c != bg_color]
    odd_out_idx = np.random.randint(0, num_objects)

    max_obj_height = max(3, num_rows // 2)
    max_obj_width = max(3, num_cols // 2)
    common_height = np.random.randint(3, max_obj_height + 1)
    other_heights = [h for h in range(3, max_obj_height + 1) if h != common_height]
    odd_height = int(np.random.choice(other_heights)) if other_heights else common_height

    for obj_idx in range(num_objects):
        obj_color = int(np.random.choice(available_colors))
        obj_height = odd_height if obj_idx == odd_out_idx else common_height
        obj_id = obj_idx + 1
        shape_type = np.random.choice(['filled_rect', 'empty_rect'])
        forbidden_mask = scipy.ndimage.binary_dilation(object_mask != 0, structure=np.ones((3, 3)))

        found_spot = False
        for _ in range(50):
            obj_width = np.random.randint(3, max_obj_width + 1)
            if num_rows < obj_height or num_cols < obj_width:
                continue
            start_row = np.random.randint(0, num_rows - obj_height + 1)
            start_col = np.random.randint(0, num_cols - obj_width + 1)
            region = forbidden_mask[start_row:start_row + obj_height, start_col:start_col + obj_width]
            if np.any(region):
                continue

            if shape_type == 'filled_rect':
                grid[start_row:start_row + obj_height, start_col:start_col + obj_width] = obj_color
                object_mask[start_row:start_row + obj_height, start_col:start_col + obj_width] = obj_id
            else:  # empty_rect
                grid[start_row, start_col:start_col + obj_width] = obj_color
                grid[start_row + obj_height - 1, start_col:start_col + obj_width] = obj_color
                if obj_height > 2:
                    grid[start_row + 1:start_row + obj_height - 1, start_col] = obj_color
                    grid[start_row + 1:start_row + obj_height - 1, start_col + obj_width - 1] = obj_color
                object_mask[start_row, start_col:start_col + obj_width] = obj_id
                object_mask[start_row + obj_height - 1, start_col:start_col + obj_width] = obj_id
                if obj_height > 2:
                    object_mask[start_row + 1:start_row + obj_height - 1, start_col] = obj_id
                    object_mask[start_row + 1:start_row + obj_height - 1, start_col + obj_width - 1] = obj_id
            found_spot = True
            break

        if not found_spot:
            break

    return grid, object_mask, None

def _object_pixel_count(shape_type, h, w):
    """Number of pixels in the object (filled area or border for empty rect)."""
    if shape_type == 'filled_rect':
        return h * w
    return 2 * h + 2 * w - 4  # empty_rect: border only


def _enumerate_dims_for_size(target_size, max_h, max_w):
    """Yield (shape_type, h, w) such that pixel count equals target_size and fits in max_h, max_w."""
    for shape_type in ('filled_rect', 'empty_rect'):
        if shape_type == 'filled_rect':
            for h in range(3, min(target_size // 3, max_h) + 1):
                if target_size % h == 0:
                    w = target_size // h
                    if 3 <= w <= max_w:
                        yield shape_type, h, w
        else:
            half = (target_size + 4) // 2
            if half < 6:
                continue
            for h in range(3, min(half - 2, max_h) + 1):
                w = half - h
                if 3 <= w <= max_w:
                    yield shape_type, h, w


def _grow_blob_exact_size(num_rows, num_cols, forbidden_mask, target_size):
    """Grow a connected blob of exactly target_size pixels by random frontier growth. Returns set of (r,c) or None."""
    if target_size < 3:
        return None
    for _ in range(80):
        seed_r = np.random.randint(0, num_rows)
        seed_c = np.random.randint(0, num_cols)
        if forbidden_mask[seed_r, seed_c]:
            continue
        blob = {(seed_r, seed_c)}
        frontier = set()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            r, c = seed_r + dr, seed_c + dc
            if 0 <= r < num_rows and 0 <= c < num_cols and not forbidden_mask[r, c] and (r, c) not in blob:
                frontier.add((r, c))
        while len(blob) < target_size and frontier:
            r, c = list(frontier)[np.random.randint(len(frontier))]
            frontier.discard((r, c))
            blob.add((r, c))
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < num_rows and 0 <= nc < num_cols and not forbidden_mask[nr, nc] and (nr, nc) not in blob:
                    frontier.add((nr, nc))
        if len(blob) == target_size:
            return blob
    return None


def sample_odd_one_out_size(training_path, min_dim=None, max_dim=None, colors_present=None):
    if min_dim is None:
        min_dim = 5

    if max_dim is None:
        max_dim = 30

    while True:
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
        object_mask = np.zeros((num_rows, num_cols), dtype=int)

        max_obj_height = max(3, num_rows // 2)
        max_obj_width = max(3, num_cols // 2)

        # Build achievable sizes (pixel counts that fit in grid)
        achievable_sizes = set()
        for h in range(3, max_obj_height + 1):
            for w in range(3, max_obj_width + 1):
                achievable_sizes.add(h * w)
                achievable_sizes.add(2 * h + 2 * w - 4)
        achievable_sizes = [s for s in achievable_sizes if s >= 8]

        # 3 to 7 objects; any color except bg; all same pixel count except one (odd one out)
        num_objects = np.random.randint(3, 8)
        available_colors = [c for c in range(10) if c != bg_color]
        odd_out_idx = np.random.randint(0, num_objects)
        common_size = int(np.random.choice(achievable_sizes))
        other_sizes = [s for s in achievable_sizes if s != common_size]
        odd_size = int(np.random.choice(other_sizes)) if other_sizes else common_size

        for obj_idx in range(num_objects):
            obj_color = int(np.random.choice(available_colors))
            target_size = odd_size if obj_idx == odd_out_idx else common_size
            options = list(_enumerate_dims_for_size(target_size, max_obj_height, max_obj_width))
            if not options:
                break
            obj_id = obj_idx + 1
            forbidden_mask = scipy.ndimage.binary_dilation(object_mask != 0, structure=np.ones((3, 3)))

            found_spot = False
            for _ in range(50):
                use_blob = np.random.random() < 0.5
                if use_blob:
                    pixels = _grow_blob_exact_size(num_rows, num_cols, forbidden_mask, target_size)
                    if pixels is not None:
                        for r, c in pixels:
                            grid[r, c] = obj_color
                            object_mask[r, c] = obj_id
                        found_spot = True
                        break
                    continue
                # Rectangle path
                shape_type, obj_height, obj_width = options[np.random.randint(len(options))]
                if num_rows < obj_height or num_cols < obj_width:
                    continue
                start_row = np.random.randint(0, num_rows - obj_height + 1)
                start_col = np.random.randint(0, num_cols - obj_width + 1)
                region = forbidden_mask[start_row:start_row + obj_height, start_col:start_col + obj_width]
                if np.any(region):
                    continue
                if shape_type == 'filled_rect':
                    grid[start_row:start_row + obj_height, start_col:start_col + obj_width] = obj_color
                    object_mask[start_row:start_row + obj_height, start_col:start_col + obj_width] = obj_id
                else:
                    grid[start_row, start_col:start_col + obj_width] = obj_color
                    grid[start_row + obj_height - 1, start_col:start_col + obj_width] = obj_color
                    if obj_height > 2:
                        grid[start_row + 1:start_row + obj_height - 1, start_col] = obj_color
                        grid[start_row + 1:start_row + obj_height - 1, start_col + obj_width - 1] = obj_color
                    object_mask[start_row, start_col:start_col + obj_width] = obj_id
                    object_mask[start_row + obj_height - 1, start_col:start_col + obj_width] = obj_id
                    if obj_height > 2:
                        object_mask[start_row + 1:start_row + obj_height - 1, start_col] = obj_id
                        object_mask[start_row + 1:start_row + obj_height - 1, start_col + obj_width - 1] = obj_id
                found_spot = True
                break

            if not found_spot:
                break

        placed_ids = set(np.unique(object_mask)) - {0}
        # Need at least 3 objects and the odd-one-out must be among them (so exactly 1 has odd_size, rest common_size)
        if len(placed_ids) >= 3 and (odd_out_idx + 1) in placed_ids:
            return grid, object_mask, None


def _is_connected(pixels):
    """True iff the set of (r,c) forms a single 4-connected component."""
    if len(pixels) <= 1:
        return True
    start = next(iter(pixels))
    visited = set()
    queue = [start]
    while queue:
        r, c = queue.pop(0)
        if (r, c) not in pixels or (r, c) in visited:
            continue
        visited.add((r, c))
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            queue.append((r + dr, c + dc))
    return len(visited) == len(pixels)


def _grow_blob_symmetric_horizontal(num_rows, num_cols, forbidden_mask, sr, sc, H, W, target_size):
    """Grow a connected blob in [sr:sr+H, sc:sc+W] in the left half only, then mirror. Seed on axis so result is one whole shape."""
    axis_col = sc + W // 2  # center column (included in "left half"); blob must touch this so left+mirror are connected
    for _ in range(30):
        seed_r = np.random.randint(sr, min(sr + H, num_rows))
        seed_c = axis_col
        if forbidden_mask[seed_r, seed_c]:
            continue
        visited = set()
        queue = [(seed_r, seed_c)]
        left_pixels = set()
        while queue and len(left_pixels) < target_size:
            r, c = queue.pop(0)
            if (r, c) in visited or r < sr or r >= sr + H or c < sc or c > axis_col:
                continue
            if forbidden_mask[r, c]:
                continue
            visited.add((r, c))
            left_pixels.add((r, c))
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                queue.append((r + dr, c + dc))
        if len(left_pixels) < 3:
            continue
        full = set(left_pixels)
        for (r, c) in left_pixels:
            mirror_c = sc + W - 1 - (c - sc)
            if sc <= mirror_c < sc + W and mirror_c != c:
                full.add((r, mirror_c))
        if _is_connected(full):
            return full
    return None


def _is_horizontally_symmetric(pixels):
    """True iff the set of (r,c) is symmetric about some vertical axis (min_c+max_c - c)."""
    if not pixels:
        return True
    cols = [c for (_, c) in pixels]
    min_c, max_c = min(cols), max(cols)
    center_double = min_c + max_c  # mirror of c is center_double - c
    for (r, c) in pixels:
        mirror_c = center_double - c
        if (r, mirror_c) not in pixels:
            return False
    return True


def _is_vertically_symmetric(pixels):
    """True iff the set of (r,c) is symmetric about some horizontal axis (min_r+max_r - r)."""
    if not pixels:
        return True
    rows = [r for (r, _) in pixels]
    min_r, max_r = min(rows), max(rows)
    center_double = min_r + max_r
    for (r, c) in pixels:
        mirror_r = center_double - r
        if (mirror_r, c) not in pixels:
            return False
    return True


def _grow_blob_symmetric_vertical(num_rows, num_cols, forbidden_mask, sr, sc, H, W, target_size):
    """Grow a connected blob in [sr:sr+H, sc:sc+W] in the top half only, then mirror. Seed on axis so result is one whole shape."""
    axis_row = sr + H // 2
    for _ in range(30):
        seed_r = axis_row
        seed_c = np.random.randint(sc, min(sc + W, num_cols))
        if forbidden_mask[seed_r, seed_c]:
            continue
        visited = set()
        queue = [(seed_r, seed_c)]
        top_pixels = set()
        while queue and len(top_pixels) < target_size:
            r, c = queue.pop(0)
            if (r, c) in visited or r < sr or r > axis_row or c < sc or c >= sc + W:
                continue
            if forbidden_mask[r, c]:
                continue
            visited.add((r, c))
            top_pixels.add((r, c))
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                queue.append((r + dr, c + dc))
        if len(top_pixels) < 3:
            continue
        full = set(top_pixels)
        for (r, c) in top_pixels:
            mirror_r = 2 * sr + H - 1 - r
            if sr <= mirror_r < sr + H and mirror_r != r:
                full.add((mirror_r, c))
        if _is_connected(full):
            return full
    return None


def _grow_blob_asymmetric_vertical(num_rows, num_cols, forbidden_mask, target_size):
    """Grow a connected blob that is NOT vertically symmetric. Returns set of (r,c) or None."""
    for _ in range(80):
        seed_r = np.random.randint(0, num_rows)
        seed_c = np.random.randint(0, num_cols)
        if forbidden_mask[seed_r, seed_c]:
            continue
        visited = set()
        queue = [(seed_r, seed_c)]
        pixels = set()
        while queue and len(pixels) < target_size:
            r, c = queue.pop(0)
            if (r, c) in visited or r < 0 or r >= num_rows or c < 0 or c >= num_cols:
                continue
            if forbidden_mask[r, c]:
                continue
            visited.add((r, c))
            pixels.add((r, c))
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                queue.append((r + dr, c + dc))
        if len(pixels) < 3:
            continue
        if not _is_vertically_symmetric(pixels):
            return pixels

        # Blob is symmetric by chance: try adding a neighbor pixel that truly breaks symmetry.
        # Re-check symmetry after each candidate addition to avoid issues from a changed bounding box.
        for r, c in list(pixels):
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= num_rows or nc < 0 or nc >= num_cols:
                    continue
                if (nr, nc) in pixels or forbidden_mask[nr, nc]:
                    continue
                candidate = set(pixels)
                candidate.add((nr, nc))
                if not _is_vertically_symmetric(candidate):
                    return candidate
    return None


def _grow_blob_asymmetric(num_rows, num_cols, forbidden_mask, target_size):
    """Grow a connected blob from a random free cell that is NOT horizontally symmetric. Returns set of (r,c) or None."""
    for _ in range(80):
        seed_r = np.random.randint(0, num_rows)
        seed_c = np.random.randint(0, num_cols)
        if forbidden_mask[seed_r, seed_c]:
            continue
        visited = set()
        queue = [(seed_r, seed_c)]
        pixels = set()
        while queue and len(pixels) < target_size:
            r, c = queue.pop(0)
            if (r, c) in visited or r < 0 or r >= num_rows or c < 0 or c >= num_cols:
                continue
            if forbidden_mask[r, c]:
                continue
            visited.add((r, c))
            pixels.add((r, c))
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                queue.append((r + dr, c + dc))
        if len(pixels) < 3:
            continue
        if not _is_horizontally_symmetric(pixels):
            return pixels

        # Blob is symmetric by chance: try adding a neighbor pixel that truly breaks symmetry.
        # Re-check symmetry after each candidate addition to avoid artifacts from a changed bounding box.
        for r, c in list(pixels):
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= num_rows or nc < 0 or nc >= num_cols:
                    continue
                if (nr, nc) in pixels or forbidden_mask[nr, nc]:
                    continue
                candidate = set(pixels)
                candidate.add((nr, nc))
                if not _is_horizontally_symmetric(candidate):
                    return candidate
    return None


def sample_odd_one_out_non_symmetry_h(training_path, min_dim=None, max_dim=None, colors_present=None):
    if min_dim is None:
        min_dim = 5

    if max_dim is None:
        max_dim = 30

    while True:
        num_rows = np.random.randint(min_dim, max_dim + 1)
        num_cols = np.random.randint(min_dim, max_dim + 1)

        if np.random.random() < 0.5:
            bg_color = 0
        else:
            bg_color = np.random.randint(1, 10)

        grid = np.full((num_rows, num_cols), bg_color)
        object_mask = np.zeros((num_rows, num_cols), dtype=int)

        max_obj_height = max(3, num_rows // 2)
        max_obj_width = max(3, num_cols // 2)
        max_blob_size = max(8, (num_rows * num_cols) // (8 * 4))

        num_objects = np.random.randint(3, 8)
        available_colors = [c for c in range(10) if c != bg_color]
        odd_out_idx = np.random.randint(0, num_objects)

        for obj_idx in range(num_objects):
            obj_color = int(np.random.choice(available_colors))
            obj_id = obj_idx + 1
            forbidden_mask = scipy.ndimage.binary_dilation(object_mask != 0, structure=np.ones((3, 3)))
            is_odd_out = obj_idx == odd_out_idx
            target_size = np.random.randint(5, max_blob_size + 1)

            found_spot = False
            if is_odd_out:
                for _ in range(50):
                    H = np.random.randint(3, max_obj_height + 1)
                    W = np.random.randint(3, max_obj_width + 1)
                    if num_rows < H or num_cols < W:
                        continue
                    sr = np.random.randint(0, num_rows - H + 1)
                    sc = np.random.randint(0, num_cols - W + 1)

                    if np.any(forbidden_mask[sr:sr + H, sc:sc + W]):
                        continue

                    # Odd-out object: horizontally symmetric
                    pixels = _grow_blob_symmetric_horizontal(
                        num_rows, num_cols, forbidden_mask, sr, sc, H, W, target_size
                    )
                    if pixels:
                        for r, c in pixels:
                            grid[r, c] = obj_color
                            object_mask[r, c] = obj_id
                        found_spot = True
                        break
            else:
                # Other objects: horizontally non-symmetric
                for _ in range(50):
                    pixels = _grow_blob_asymmetric(num_rows, num_cols, forbidden_mask, target_size)

                    if pixels:
                        for r, c in pixels:
                            grid[r, c] = obj_color
                            object_mask[r, c] = obj_id
                        found_spot = True
                        break

            if not found_spot:
                break

        placed_ids = set(np.unique(object_mask)) - {0}
        if len(placed_ids) >= 3 and (odd_out_idx + 1) in placed_ids:
            return grid, object_mask, None


def sample_odd_one_out_non_symmetry_v(training_path, min_dim=None, max_dim=None, colors_present=None):
    if min_dim is None:
        min_dim = 5

    if max_dim is None:
        max_dim = 30

    while True:
        num_rows = np.random.randint(min_dim, max_dim + 1)
        num_cols = np.random.randint(min_dim, max_dim + 1)

        if np.random.random() < 0.5:
            bg_color = 0
        else:
            bg_color = np.random.randint(1, 10)

        grid = np.full((num_rows, num_cols), bg_color)
        object_mask = np.zeros((num_rows, num_cols), dtype=int)

        max_obj_height = max(3, num_rows // 2)
        max_obj_width = max(3, num_cols // 2)
        max_blob_size = max(8, (num_rows * num_cols) // (8 * 4))

        num_objects = np.random.randint(3, 8)
        available_colors = [c for c in range(10) if c != bg_color]
        odd_out_idx = np.random.randint(0, num_objects)

        for obj_idx in range(num_objects):
            obj_color = int(np.random.choice(available_colors))
            obj_id = obj_idx + 1
            forbidden_mask = scipy.ndimage.binary_dilation(object_mask != 0, structure=np.ones((3, 3)))
            is_odd_out = obj_idx == odd_out_idx
            target_size = np.random.randint(5, max_blob_size + 1)

            found_spot = False
            if is_odd_out:
                for _ in range(50):
                    H = np.random.randint(3, max_obj_height + 1)
                    W = np.random.randint(3, max_obj_width + 1)
                    if num_rows < H or num_cols < W:
                        continue
                    sr = np.random.randint(0, num_rows - H + 1)
                    sc = np.random.randint(0, num_cols - W + 1)

                    if np.any(forbidden_mask[sr:sr + H, sc:sc + W]):
                        continue

                    # Odd-out object: vertically symmetric
                    pixels = _grow_blob_symmetric_vertical(
                        num_rows, num_cols, forbidden_mask, sr, sc, H, W, target_size
                    )
                    if pixels:
                        for r, c in pixels:
                            grid[r, c] = obj_color
                            object_mask[r, c] = obj_id
                        found_spot = True
                        break
            else:
                # Other objects: vertically non-symmetric
                for _ in range(50):
                    pixels = _grow_blob_asymmetric_vertical(num_rows, num_cols, forbidden_mask, target_size)

                    if pixels:
                        for r, c in pixels:
                            grid[r, c] = obj_color
                            object_mask[r, c] = obj_id
                        found_spot = True
                        break

            if not found_spot:
                break

        placed_ids = set(np.unique(object_mask)) - {0}
        if len(placed_ids) >= 3 and (odd_out_idx + 1) in placed_ids:
            return grid, object_mask, None


def sample_odd_one_out_symmetry_h(training_path, min_dim=None, max_dim=None, colors_present=None):
    if min_dim is None:
        min_dim = 5

    if max_dim is None:
        max_dim = 30

    while True:
        num_rows = np.random.randint(min_dim, max_dim + 1)
        num_cols = np.random.randint(min_dim, max_dim + 1)

        if np.random.random() < 0.5:
            bg_color = 0
        else:
            bg_color = np.random.randint(1, 10)

        grid = np.full((num_rows, num_cols), bg_color)
        object_mask = np.zeros((num_rows, num_cols), dtype=int)

        max_obj_height = max(3, num_rows // 2)
        max_obj_width = max(3, num_cols // 2)
        max_blob_size = max(8, (num_rows * num_cols) // (8 * 4))

        num_objects = np.random.randint(3, 8)
        available_colors = [c for c in range(10) if c != bg_color]
        odd_out_idx = np.random.randint(0, num_objects)

        for obj_idx in range(num_objects):
            obj_color = int(np.random.choice(available_colors))
            obj_id = obj_idx + 1
            forbidden_mask = scipy.ndimage.binary_dilation(object_mask != 0, structure=np.ones((3, 3)))
            is_odd_out = obj_idx == odd_out_idx
            target_size = np.random.randint(5, max_blob_size + 1)

            found_spot = False
            if is_odd_out:
                pixels = _grow_blob_asymmetric(num_rows, num_cols, forbidden_mask, target_size)
                if pixels:
                    for r, c in pixels:
                        grid[r, c] = obj_color
                        object_mask[r, c] = obj_id
                    found_spot = True
            else:
                for _ in range(50):
                    H = np.random.randint(3, max_obj_height + 1)
                    W = np.random.randint(3, max_obj_width + 1)
                    if num_rows < H or num_cols < W:
                        continue
                    sr = np.random.randint(0, num_rows - H + 1)
                    sc = np.random.randint(0, num_cols - W + 1)
                    if np.any(forbidden_mask[sr:sr + H, sc:sc + W]):
                        continue
                    pixels = _grow_blob_symmetric_horizontal(
                        num_rows, num_cols, forbidden_mask, sr, sc, H, W, target_size
                    )
                    if pixels:
                        for r, c in pixels:
                            grid[r, c] = obj_color
                            object_mask[r, c] = obj_id
                        found_spot = True
                        break

            if not found_spot:
                break

        placed_ids = set(np.unique(object_mask)) - {0}
        if len(placed_ids) >= 3 and (odd_out_idx + 1) in placed_ids:
            return grid, object_mask, None


def sample_odd_one_out_symmetry_v(training_path, min_dim=None, max_dim=None, colors_present=None):
    if min_dim is None:
        min_dim = 5

    if max_dim is None:
        max_dim = 30

    while True:
        num_rows = np.random.randint(min_dim, max_dim + 1)
        num_cols = np.random.randint(min_dim, max_dim + 1)

        if np.random.random() < 0.5:
            bg_color = 0
        else:
            bg_color = np.random.randint(1, 10)

        grid = np.full((num_rows, num_cols), bg_color)
        object_mask = np.zeros((num_rows, num_cols), dtype=int)

        max_obj_height = max(3, num_rows // 2)
        max_obj_width = max(3, num_cols // 2)
        max_blob_size = max(8, (num_rows * num_cols) // (8 * 4))

        num_objects = np.random.randint(3, 8)
        available_colors = [c for c in range(10) if c != bg_color]
        odd_out_idx = np.random.randint(0, num_objects)

        for obj_idx in range(num_objects):
            obj_color = int(np.random.choice(available_colors))
            obj_id = obj_idx + 1
            forbidden_mask = scipy.ndimage.binary_dilation(object_mask != 0, structure=np.ones((3, 3)))
            is_odd_out = obj_idx == odd_out_idx
            target_size = np.random.randint(5, max_blob_size + 1)

            found_spot = False
            if is_odd_out:
                pixels = _grow_blob_asymmetric_vertical(num_rows, num_cols, forbidden_mask, target_size)
                if pixels:
                    for r, c in pixels:
                        grid[r, c] = obj_color
                        object_mask[r, c] = obj_id
                    found_spot = True
            else:
                for _ in range(50):
                    H = np.random.randint(3, max_obj_height + 1)
                    W = np.random.randint(3, max_obj_width + 1)
                    if num_rows < H or num_cols < W:
                        continue
                    sr = np.random.randint(0, num_rows - H + 1)
                    sc = np.random.randint(0, num_cols - W + 1)
                    if np.any(forbidden_mask[sr:sr + H, sc:sc + W]):
                        continue
                    pixels = _grow_blob_symmetric_vertical(
                        num_rows, num_cols, forbidden_mask, sr, sc, H, W, target_size
                    )
                    if pixels:
                        for r, c in pixels:
                            grid[r, c] = obj_color
                            object_mask[r, c] = obj_id
                        found_spot = True
                        break

            if not found_spot:
                break

        placed_ids = set(np.unique(object_mask)) - {0}
        if len(placed_ids) >= 3 and (odd_out_idx + 1) in placed_ids:
            return grid, object_mask, None


def sample_odd_one_out_color(training_path, min_dim=None, max_dim=None, colors_present=None):
    if min_dim is None:
        min_dim = 5

    if max_dim is None:
        max_dim = 30

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
    object_mask = np.zeros((num_rows, num_cols), dtype=int)

    # 3 to 7 objects; one shared color, one "odd" color (both != bg)
    num_objects = np.random.randint(3, 8)
    available_colors = [c for c in range(10) if c != bg_color]
    main_color = int(np.random.choice(available_colors))
    odd_colors = [c for c in available_colors if c != main_color]
    odd_color = int(np.random.choice(odd_colors)) if odd_colors else main_color
    odd_out_idx = np.random.randint(0, num_objects)

    max_obj_height = max(3, num_rows // 2)
    max_obj_width = max(3, num_cols // 2)
    max_arbitrary_size = max(3, (num_rows * num_cols) // (num_objects * 4))

    for obj_idx in range(num_objects):
        obj_color = odd_color if obj_idx == odd_out_idx else main_color
        obj_id = obj_idx + 1
        shape_type = np.random.choice(['filled_rect', 'empty_rect', 'arbitrary'])
        # Forbidden = existing objects plus 1-cell border (no adjacency)
        forbidden_mask = scipy.ndimage.binary_dilation(object_mask != 0, structure=np.ones((3, 3)))

        found_spot = False
        for _ in range(50):
            if shape_type == 'filled_rect':
                obj_height = np.random.randint(3, max_obj_height + 1)
                obj_width = np.random.randint(3, max_obj_width + 1)
                if num_rows < obj_height or num_cols < obj_width:
                    continue
                start_row = np.random.randint(0, num_rows - obj_height + 1)
                start_col = np.random.randint(0, num_cols - obj_width + 1)
                region = forbidden_mask[start_row:start_row + obj_height, start_col:start_col + obj_width]
                if np.any(region):
                    continue
                grid[start_row:start_row + obj_height, start_col:start_col + obj_width] = obj_color
                object_mask[start_row:start_row + obj_height, start_col:start_col + obj_width] = obj_id
                found_spot = True
                break

            elif shape_type == 'empty_rect':
                obj_height = np.random.randint(3, max_obj_height + 1)
                obj_width = np.random.randint(3, max_obj_width + 1)
                if num_rows < obj_height or num_cols < obj_width:
                    continue
                start_row = np.random.randint(0, num_rows - obj_height + 1)
                start_col = np.random.randint(0, num_cols - obj_width + 1)
                region = forbidden_mask[start_row:start_row + obj_height, start_col:start_col + obj_width]
                if np.any(region):
                    continue
                grid[start_row, start_col:start_col + obj_width] = obj_color
                grid[start_row + obj_height - 1, start_col:start_col + obj_width] = obj_color
                if obj_height > 2:
                    grid[start_row + 1:start_row + obj_height - 1, start_col] = obj_color
                    grid[start_row + 1:start_row + obj_height - 1, start_col + obj_width - 1] = obj_color
                object_mask[start_row, start_col:start_col + obj_width] = obj_id
                object_mask[start_row + obj_height - 1, start_col:start_col + obj_width] = obj_id
                if obj_height > 2:
                    object_mask[start_row + 1:start_row + obj_height - 1, start_col] = obj_id
                    object_mask[start_row + 1:start_row + obj_height - 1, start_col + obj_width - 1] = obj_id
                found_spot = True
                break

            else:  # arbitrary
                obj_size = np.random.randint(3, max_arbitrary_size + 1)
                start_row = np.random.randint(0, num_rows)
                start_col = np.random.randint(0, num_cols)
                if forbidden_mask[start_row, start_col]:
                    continue
                visited = set()
                queue = [(start_row, start_col)]
                shape_pixels = set()
                while queue and len(shape_pixels) < obj_size:
                    row, col = queue.pop(0)
                    if (row, col) in visited or row < 0 or row >= num_rows or col < 0 or col >= num_cols:
                        continue
                    if forbidden_mask[row, col]:
                        continue
                    visited.add((row, col))
                    shape_pixels.add((row, col))
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            queue.append((row + dr, col + dc))
                if len(shape_pixels) < 3:
                    continue
                for r, c in shape_pixels:
                    grid[r, c] = obj_color
                    object_mask[r, c] = obj_id
                found_spot = True
                break

        if not found_spot:
            break

    return grid, object_mask, None


def sample_incomplete_rectangles(training_path, min_dim=None, max_dim=None, all_same_shape=False, colors_present=None):
    if min_dim is None:
        min_dim = 5

    if max_dim is None:
        max_dim = 30

    a = np.random.uniform()

    if a < 0.25:
        return sample_incomplete_rectangles_training(training_path)

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

    # Generate 1 to 7 objects
    num_objects = np.random.randint(1, 8)
    object_colors = []
    
    # Generate unique colors for objects (different from background)
    available_colors = list(range(10))
    available_colors.remove(bg_color)
    object_colors = np.random.choice(available_colors, num_objects, replace=False)

    # Rectangle object
    max_obj_height = max(3, num_rows // 2)
    max_obj_width = max(3, num_cols // 2)

    if all_same_shape:
        obj_height = np.random.randint(3, max_obj_height + 1)
        obj_width = np.random.randint(3, max_obj_width + 1)

    for obj_idx in range(num_objects):
        obj_color = object_colors[obj_idx]
        obj_id = obj_idx + 1  # Object IDs start from 1

        if not all_same_shape:
            obj_height = np.random.randint(3, max_obj_height + 1)
            obj_width = np.random.randint(3, max_obj_width + 1)

        # Try to find a free spot for this object, up to N attempts
        found_spot = False
        max_attempts = 50
        for _ in range(max_attempts):
            start_row = np.random.randint(0, num_rows - obj_height + 1)
            start_col = np.random.randint(0, num_cols - obj_width + 1)

            # Check if this region overlaps with any existing object
            region = object_mask[start_row:start_row + obj_height, start_col:start_col + obj_width]
            if np.any(region != 0):
                continue  # Overlaps, try another position

            # Fill the entire rectangle area with the object color
            grid[start_row:start_row + obj_height, start_col:start_col + obj_width] = obj_color

            # Fill the entire rectangle area in the object mask
            object_mask[start_row:start_row + obj_height, start_col:start_col + obj_width] = obj_id

            found_spot = True
            break  # Successfully placed this object

        # Randomly remove 1 to 50% of the rectangle's pixels (set them back to bg_color)
        rect_area = obj_height * obj_width
        num_remove = np.random.randint(1, (rect_area // 2) + 1)
        # Get all pixel indices in the rectangle
        rect_indices = [(r, c) for r in range(start_row, start_row + obj_height)
                               for c in range(start_col, start_col + obj_width)]
        remove_indices = np.random.choice(len(rect_indices), num_remove, replace=False)
        for idx in remove_indices:
            r, c = rect_indices[idx]
            grid[r, c] = bg_color

        if not found_spot:
            # No more space for this object, stop placing further objects
            break
            
    return grid, object_mask, None


def sample_twin_objects_v(training_path, min_dim=None, max_dim=None, colors_present=None):
    if min_dim is None:
        min_dim = 5

    if max_dim is None:
        max_dim = 30

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
    object_mask = np.zeros((num_rows, num_cols), dtype=int)
    sub_objs_mask = []  # list of obj_masks per twin (2 per twin), each with unique sub-object ID

    # Generate 1 to 3 twin objects (each twin = two vertically stacked rectangles)
    num_objects = np.random.randint(1, 4)

    # Two colors per twin object
    available_colors = list(range(10))
    available_colors.remove(bg_color)
    object_colors, _ = ensure_colors_present(available_colors, 2 * num_objects, colors_present, bg_color)

    min_rect_dim = 2
    max_rect_height = 5
    max_rect_width = 6

    for obj_idx in range(num_objects):
        color_top = object_colors[2 * obj_idx]
        color_bottom = object_colors[2 * obj_idx + 1]
        obj_id = obj_idx + 1

        found_spot = False
        for _ in range(50):
            h1 = np.random.randint(min_rect_dim, min(max_rect_height, num_rows - min_rect_dim) + 1)
            if num_rows - h1 < min_rect_dim:
                continue
            w1 = np.random.randint(min_rect_dim, min(max_rect_width, num_cols) + 1)
            h2 = np.random.randint(min_rect_dim, min(max_rect_height, num_rows - h1) + 1)
            w2 = np.random.randint(min_rect_dim, min(max_rect_width, num_cols) + 1)

            r1 = np.random.randint(0, num_rows - (h1 + h2) + 1)
            c1 = np.random.randint(0, num_cols - w1 + 1)
            r2 = r1 + h1
            # Bottom rect must share at least one column with top rect so they touch
            c2_lo = max(0, c1 - w2 + 1)
            c2_hi = min(num_cols - w2, c1 + w1 - 1)
            if c2_lo > c2_hi:
                continue
            c2 = np.random.randint(c2_lo, c2_hi + 1)

            top_region = object_mask[r1:r1 + h1, c1:c1 + w1]
            bottom_region = object_mask[r2:r2 + h2, c2:c2 + w2]
            if np.any(top_region != 0) or np.any(bottom_region != 0):
                continue

            # No touching other twins: 4-neighbors of this twin must be empty
            our_cells = set()
            for rr in range(r1, r1 + h1):
                for cc in range(c1, c1 + w1):
                    our_cells.add((rr, cc))
            for rr in range(r2, r2 + h2):
                for cc in range(c2, c2 + w2):
                    our_cells.add((rr, cc))
            boundary = set()
            for (rr, cc) in our_cells:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < num_rows and 0 <= nc < num_cols and (nr, nc) not in our_cells:
                        boundary.add((nr, nc))
            if any(object_mask[r, c] != 0 for r, c in boundary):
                continue

            grid[r1:r1 + h1, c1:c1 + w1] = color_top
            object_mask[r1:r1 + h1, c1:c1 + w1] = obj_id
            grid[r2:r2 + h2, c2:c2 + w2] = color_bottom
            object_mask[r2:r2 + h2, c2:c2 + w2] = obj_id

            # One mask per twin: bbox-sized, 0=bg, 1=top rect, 2=bottom rect
            r_lo, c_lo = r1, min(c1, c2)
            twin_h, twin_w = h1 + h2, max(c1 + w1, c2 + w2) - c_lo
            obj_mask = np.zeros((twin_h, twin_w), dtype=int)
            obj_mask[0:h1, c1 - c_lo:c1 - c_lo + w1] = 1
            obj_mask[h1:h1 + h2, c2 - c_lo:c2 - c_lo + w2] = 2
            sub_objs_mask.append(obj_mask)
            found_spot = True
            break

        if not found_spot:
            break

    return grid, object_mask, sub_objs_mask


def sample_twin_objects_h(training_path, min_dim=None, max_dim=None, colors_present=None):
    if min_dim is None:
        min_dim = 5

    if max_dim is None:
        max_dim = 30

    num_rows = np.random.randint(min_dim, max_dim + 1)
    num_cols = np.random.randint(min_dim, max_dim + 1)

    if np.random.random() < 0.5:
        bg_color = 0
    else:
        bg_color = np.random.randint(1, 10)

    grid = np.full((num_rows, num_cols), bg_color)
    object_mask = np.zeros((num_rows, num_cols), dtype=int)
    sub_objs_mask = []

    num_objects = np.random.randint(1, 4)

    available_colors = list(range(10))
    available_colors.remove(bg_color)
    object_colors, _ = ensure_colors_present(available_colors, 2 * num_objects, colors_present, bg_color)

    min_rect_dim = 2
    max_rect_height = 5
    max_rect_width = 6

    for obj_idx in range(num_objects):
        color_left = object_colors[2 * obj_idx]
        color_right = object_colors[2 * obj_idx + 1]
        obj_id = obj_idx + 1

        found_spot = False
        for _ in range(50):
            w1 = np.random.randint(min_rect_dim, min(max_rect_width, num_cols - min_rect_dim) + 1)
            if num_cols - w1 < min_rect_dim:
                continue
            h1 = np.random.randint(min_rect_dim, min(max_rect_height, num_rows) + 1)
            w2 = np.random.randint(min_rect_dim, min(max_rect_width, num_cols - w1) + 1)
            h2 = np.random.randint(min_rect_dim, min(max_rect_height, num_rows) + 1)

            c1 = np.random.randint(0, num_cols - w1 - w2 + 1)
            r1 = np.random.randint(0, num_rows - h1 + 1)
            c2 = c1 + w1
            r2_lo = max(0, r1 - h2 + 1)
            r2_hi = min(num_rows - h2, r1 + h1 - 1)
            if r2_lo > r2_hi:
                continue
            r2 = np.random.randint(r2_lo, r2_hi + 1)

            left_region = object_mask[r1:r1 + h1, c1:c1 + w1]
            right_region = object_mask[r2:r2 + h2, c2:c2 + w2]
            if np.any(left_region != 0) or np.any(right_region != 0):
                continue

            our_cells = set()
            for rr in range(r1, r1 + h1):
                for cc in range(c1, c1 + w1):
                    our_cells.add((rr, cc))
            for rr in range(r2, r2 + h2):
                for cc in range(c2, c2 + w2):
                    our_cells.add((rr, cc))
            boundary = set()
            for (rr, cc) in our_cells:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < num_rows and 0 <= nc < num_cols and (nr, nc) not in our_cells:
                        boundary.add((nr, nc))
            if any(object_mask[r, c] != 0 for r, c in boundary):
                continue

            grid[r1:r1 + h1, c1:c1 + w1] = color_left
            object_mask[r1:r1 + h1, c1:c1 + w1] = obj_id
            grid[r2:r2 + h2, c2:c2 + w2] = color_right
            object_mask[r2:r2 + h2, c2:c2 + w2] = obj_id

            r_lo, c_lo = min(r1, r2), c1
            twin_h = max(r1 + h1, r2 + h2) - r_lo
            twin_w = w1 + w2
            obj_mask = np.zeros((twin_h, twin_w), dtype=int)
            obj_mask[r1 - r_lo:r1 - r_lo + h1, 0:w1] = 1
            obj_mask[r2 - r_lo:r2 - r_lo + h2, w1:w1 + w2] = 2
            sub_objs_mask.append(obj_mask)
            found_spot = True
            break

        if not found_spot:
            break

    return grid, object_mask, sub_objs_mask

def _sub_cells_and_forbidden(sub_type, h, w, pos):
    """Return (cells, forbidden) in local rect coords. forbidden = cells union their neighbors (for non-adjacency)."""
    cells = set()
    if sub_type == "pixel":
        dr, dc = pos
        cells = {(dr, dc)}
    elif sub_type == "plus":
        cr, cc = pos
        cells = {(cr, cc), (cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)}
    else:  # 2x2 square
        tr, tc = pos
        cells = {(tr + i, tc + j) for i in range(2) for j in range(2)}
    forbidden = set()
    for (r, c) in cells:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                forbidden.add((nr, nc))
    forbidden |= cells
    return cells, forbidden


def _place_sub_in_rect(grid, object_mask, sub_objs_mask, r0, c0, h, w, obj_id, rect_color, sub_color, sub_type, sub_idx, used_or_adjacent, max_attempts=80, mask_idx=-1):
    """Place one sub-object so it does not overlap or touch existing subs. used_or_adjacent is set of local (dr,dc). Returns True if placed."""
    mask = sub_objs_mask[mask_idx]
    if sub_type == "pixel":
        candidates = [(r, c) for r in range(h) for c in range(w)]
    elif sub_type == "plus":
        candidates = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    else:  # 2x2
        candidates = [(r, c) for r in range(h - 1) for c in range(w - 1)]
    np.random.shuffle(candidates)
    for pos in candidates[:max_attempts]:
        cells, forbidden = _sub_cells_and_forbidden(sub_type, h, w, pos)
        if forbidden & used_or_adjacent:
            continue
        used_or_adjacent |= forbidden
        if sub_type == "pixel":
            dr, dc = pos
            grid[r0 + dr, c0 + dc] = sub_color
            object_mask[r0 + dr, c0 + dc] = obj_id
            mask[dr, dc] = sub_idx
        elif sub_type == "plus":
            cr, cc = pos
            for (dr, dc) in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = cr + dr, cc + dc
                grid[r0 + nr, c0 + nc] = sub_color
                object_mask[r0 + nr, c0 + nc] = obj_id
                mask[nr, nc] = sub_idx
        else:
            tr, tc = pos
            for dr in range(2):
                for dc in range(2):
                    grid[r0 + tr + dr, c0 + tc + dc] = sub_color
                    object_mask[r0 + tr + dr, c0 + tc + dc] = obj_id
                    mask[tr + dr, tc + dc] = sub_idx
        return True
    return False


def sample_max_inner_objs(training_path, min_dim=None, max_dim=None, colors_present=None):
    if min_dim is None:
        min_dim = 5

    if max_dim is None:
        max_dim = 30

    while True:
        num_rows = np.random.randint(min_dim, max_dim + 1)
        num_cols = np.random.randint(min_dim, max_dim + 1)

        if np.random.random() < 0.5:
            bg_color = 0
        else:
            bg_color = np.random.randint(1, 10)

        grid = np.full((num_rows, num_cols), bg_color)
        object_mask = np.zeros((num_rows, num_cols), dtype=int)
        sub_objs_mask = []

        # 2 to 6 rectangles depending on grid size
        num_rects = min(6, max(2, 2 + min(num_rows, num_cols) // 6))
        num_rects = np.random.randint(2, num_rects + 1)

        available_colors = list(range(10))
        available_colors.remove(bg_color)
        object_colors, _ = ensure_colors_present(available_colors, 2 * num_rects, colors_present, bg_color)

        min_rect_dim = 5
        max_rect_height = min(12, num_rows)
        max_rect_width = min(14, num_cols)

        # One sub-object type for the whole grid
        grid_sub_type = np.random.choice(["pixel", "plus", "square"])

        rect_bounds = []  # (r0, c0, h, w) per placed object for unique-max fixup

        for obj_idx in range(num_rects):
            rect_color = object_colors[2 * obj_idx]
            sub_color = object_colors[2 * obj_idx + 1]
            obj_id = obj_idx + 1

            found_spot = False
            for _ in range(50):
                w = np.random.randint(min_rect_dim, max_rect_width + 1)
                h = np.random.randint(min_rect_dim, max_rect_height + 1)
                if w > num_cols or h > num_rows:
                    continue
                c0 = np.random.randint(0, num_cols - w + 1)
                r0 = np.random.randint(0, num_rows - h + 1)

                region = object_mask[r0:r0 + h, c0:c0 + w]
                if np.any(region != 0):
                    continue

                grid[r0:r0 + h, c0:c0 + w] = rect_color
                object_mask[r0:r0 + h, c0:c0 + w] = obj_id
                obj_mask = np.zeros((h, w), dtype=int)
                sub_objs_mask.append(obj_mask)
                rect_bounds.append((r0, c0, h, w))

                num_subs = np.random.randint(1, 4)
                used_or_adjacent = set()

                for sub_idx in range(1, num_subs + 1):
                    if not _place_sub_in_rect(grid, object_mask, sub_objs_mask, r0, c0, h, w, obj_id, rect_color, sub_color, grid_sub_type, sub_idx, used_or_adjacent):
                        break

                found_spot = True
                break

            if not found_spot:
                break

        # Ensure the object with the most sub-objects has a unique count
        if sub_objs_mask:
            counts = [int(m.max()) for m in sub_objs_mask]
            max_count = max(counts)
            while counts.count(max_count) > 1:
                idx_with_max = next(i for i, c in enumerate(counts) if c == max_count)
                r0, c0, h, w = rect_bounds[idx_with_max]
                obj_id = idx_with_max + 1
                mask = sub_objs_mask[idx_with_max]
                rect_color = object_colors[2 * idx_with_max]
                sub_color = object_colors[2 * idx_with_max + 1]
                used_or_adjacent = set()
                for dr in range(h):
                    for dc in range(w):
                        if mask[dr, dc] > 0:
                            for nr, nc in [(dr, dc), (dr - 1, dc), (dr + 1, dc), (dr, dc - 1), (dr, dc + 1)]:
                                if 0 <= nr < h and 0 <= nc < w:
                                    used_or_adjacent.add((nr, nc))
                new_sub_idx = max_count + 1
                _place_sub_in_rect(
                    grid,
                    object_mask,
                    sub_objs_mask,
                    r0,
                    c0,
                    h,
                    w,
                    obj_id,
                    rect_color,
                    sub_color,
                    grid_sub_type,
                    new_sub_idx,
                    used_or_adjacent,
                    mask_idx=idx_with_max,
                )
                counts = [int(m.max()) for m in sub_objs_mask]
                max_count = max(counts)

        counts = [int(m.max()) for m in sub_objs_mask] if sub_objs_mask else []
        if len(sub_objs_mask) >= 2 and counts and counts.count(max(counts)) == 1:
            return grid, object_mask, sub_objs_mask

def sample_min_inner_objs(training_path, min_dim=None, max_dim=None, colors_present=None):
    if min_dim is None:
        min_dim = 5

    if max_dim is None:
        max_dim = 30

    while True:
        num_rows = np.random.randint(min_dim, max_dim + 1)
        num_cols = np.random.randint(min_dim, max_dim + 1)

        if np.random.random() < 0.5:
            bg_color = 0
        else:
            bg_color = np.random.randint(1, 10)

        grid = np.full((num_rows, num_cols), bg_color)
        object_mask = np.zeros((num_rows, num_cols), dtype=int)
        sub_objs_mask = []

        # 2 to 6 rectangles depending on grid size
        num_rects = min(6, max(2, 2 + min(num_rows, num_cols) // 6))
        num_rects = np.random.randint(2, num_rects + 1)

        available_colors = list(range(10))
        available_colors.remove(bg_color)
        object_colors, _ = ensure_colors_present(available_colors, 2 * num_rects, colors_present, bg_color)

        min_rect_dim = 5
        max_rect_height = min(12, num_rows)
        max_rect_width = min(14, num_cols)

        # One sub-object type for the whole grid
        grid_sub_type = np.random.choice(["pixel", "plus", "square"])

        rect_bounds = []  # (r0, c0, h, w) per placed object for unique-max fixup

        for obj_idx in range(num_rects):
            rect_color = object_colors[2 * obj_idx]
            sub_color = object_colors[2 * obj_idx + 1]
            obj_id = obj_idx + 1

            found_spot = False
            for _ in range(50):
                w = np.random.randint(min_rect_dim, max_rect_width + 1)
                h = np.random.randint(min_rect_dim, max_rect_height + 1)
                if w > num_cols or h > num_rows:
                    continue
                c0 = np.random.randint(0, num_cols - w + 1)
                r0 = np.random.randint(0, num_rows - h + 1)

                region = object_mask[r0:r0 + h, c0:c0 + w]
                if np.any(region != 0):
                    continue

                grid[r0:r0 + h, c0:c0 + w] = rect_color
                object_mask[r0:r0 + h, c0:c0 + w] = obj_id
                obj_mask = np.zeros((h, w), dtype=int)
                sub_objs_mask.append(obj_mask)
                rect_bounds.append((r0, c0, h, w))

                num_subs = np.random.randint(1, 4)
                used_or_adjacent = set()

                for sub_idx in range(1, num_subs + 1):
                    if not _place_sub_in_rect(grid, object_mask, sub_objs_mask, r0, c0, h, w, obj_id, rect_color, sub_color, grid_sub_type, sub_idx, used_or_adjacent):
                        break

                found_spot = True
                break

            if not found_spot:
                break

        # Ensure the object with the least sub-objects has a unique count
        if sub_objs_mask:
            counts = [int(m.max()) for m in sub_objs_mask]
            min_count = min(counts)
            while counts.count(min_count) > 1:
                idx_with_min = next(i for i, c in enumerate(counts) if c == min_count)
                r0, c0, h, w = rect_bounds[idx_with_min]
                obj_id = idx_with_min + 1
                mask = sub_objs_mask[idx_with_min]
                rect_color = object_colors[2 * idx_with_min]
                sub_color = object_colors[2 * idx_with_min + 1]
                used_or_adjacent = set()
                for dr in range(h):
                    for dc in range(w):
                        if mask[dr, dc] > 0:
                            for nr, nc in [(dr, dc), (dr - 1, dc), (dr + 1, dc), (dr, dc - 1), (dr, dc + 1)]:
                                if 0 <= nr < h and 0 <= nc < w:
                                    used_or_adjacent.add((nr, nc))
                new_sub_idx = min_count + 1
                _place_sub_in_rect(grid, object_mask, sub_objs_mask, r0, c0, h, w, obj_id, rect_color, sub_color, grid_sub_type, new_sub_idx, used_or_adjacent, mask_idx=idx_with_min)
                counts = [int(m.max()) for m in sub_objs_mask]
                min_count = min(counts)

        counts = [int(m.max()) for m in sub_objs_mask] if sub_objs_mask else []
        if len(sub_objs_mask) >= 2 and counts and counts.count(min(counts)) == 1:
            return grid, object_mask, sub_objs_mask

def sample_odd_one_out_subobj_count(training_path, min_dim=None, max_dim=None, colors_present=None):
    if min_dim is None:
        min_dim = 5

    if max_dim is None:
        max_dim = 30

    while True:
        num_rows = np.random.randint(min_dim, max_dim + 1)
        num_cols = np.random.randint(min_dim, max_dim + 1)

        if np.random.random() < 0.5:
            bg_color = 0
        else:
            bg_color = np.random.randint(1, 10)

        grid = np.full((num_rows, num_cols), bg_color)
        object_mask = np.zeros((num_rows, num_cols), dtype=int)
        sub_objs_mask = []

        # 2 to 6 rectangles depending on grid size
        num_rects = min(6, max(2, 2 + min(num_rows, num_cols) // 6))
        num_rects = np.random.randint(2, num_rects + 1)

        available_colors = list(range(10))
        available_colors.remove(bg_color)
        object_colors, _ = ensure_colors_present(available_colors, 2 * num_rects, colors_present, bg_color)

        min_rect_dim = 5
        max_rect_height = min(12, num_rows)
        max_rect_width = min(14, num_cols)

        # One sub-object type for the whole grid
        grid_sub_type = np.random.choice(["pixel", "plus", "square"])

        rect_bounds = []  # (r0, c0, h, w) per placed object for unique-max fixup

        for obj_idx in range(num_rects):
            rect_color = object_colors[2 * obj_idx]
            sub_color = object_colors[2 * obj_idx + 1]
            obj_id = obj_idx + 1

            found_spot = False
            for _ in range(50):
                w = np.random.randint(min_rect_dim, max_rect_width + 1)
                h = np.random.randint(min_rect_dim, max_rect_height + 1)
                if w > num_cols or h > num_rows:
                    continue
                c0 = np.random.randint(0, num_cols - w + 1)
                r0 = np.random.randint(0, num_rows - h + 1)

                region = object_mask[r0:r0 + h, c0:c0 + w]
                if np.any(region != 0):
                    continue

                grid[r0:r0 + h, c0:c0 + w] = rect_color
                object_mask[r0:r0 + h, c0:c0 + w] = obj_id
                obj_mask = np.zeros((h, w), dtype=int)
                sub_objs_mask.append(obj_mask)
                rect_bounds.append((r0, c0, h, w))

                num_subs = np.random.randint(1, 4)
                used_or_adjacent = set()

                for sub_idx in range(1, num_subs + 1):
                    if not _place_sub_in_rect(grid, object_mask, sub_objs_mask, r0, c0, h, w, obj_id, rect_color, sub_color, grid_sub_type, sub_idx, used_or_adjacent):
                        break

                found_spot = True
                break

            if not found_spot:
                break

        # Post-process so that all objects share the same sub-object count
        # except for exactly one odd-one-out object. Require at least 3 objects.
        if sub_objs_mask:
            def _add_one_sub(idx):
                """Attempt to add a single sub-object inside rectangle idx. Returns True on success."""
                r0, c0, h, w = rect_bounds[idx]
                obj_id = idx + 1
                mask = sub_objs_mask[idx]
                rect_color = object_colors[2 * idx]
                sub_color = object_colors[2 * idx + 1]
                used_or_adjacent = set()
                for dr in range(h):
                    for dc in range(w):
                        if mask[dr, dc] > 0:
                            for nr, nc in [(dr, dc), (dr - 1, dc), (dr + 1, dc), (dr, dc - 1), (dr, dc + 1)]:
                                if 0 <= nr < h and 0 <= nc < w:
                                    used_or_adjacent.add((nr, nc))
                new_sub_idx = int(mask.max()) + 1
                return _place_sub_in_rect(
                    grid,
                    object_mask,
                    sub_objs_mask,
                    r0,
                    c0,
                    h,
                    w,
                    obj_id,
                    rect_color,
                    sub_color,
                    grid_sub_type,
                    new_sub_idx,
                    used_or_adjacent,
                    mask_idx=idx,
                )

            counts = [int(m.max()) for m in sub_objs_mask]

            if len(sub_objs_mask) >= 3:
                # Raise all objects up towards the global max count (only adding subs).
                common_target = max(counts)
                for idx in range(len(sub_objs_mask)):
                    while counts[idx] < common_target:
                        if not _add_one_sub(idx):
                            break
                        counts[idx] = int(sub_objs_mask[idx].max())

                # Pick one object and try to give it an extra sub-object
                # so its count differs from the common target.
                odd_idx = np.random.randint(0, len(sub_objs_mask))
                if counts[odd_idx] == common_target:
                    if _add_one_sub(odd_idx):
                        counts[odd_idx] = int(sub_objs_mask[odd_idx].max())

        counts = [int(m.max()) for m in sub_objs_mask] if sub_objs_mask else []
        if len(sub_objs_mask) >= 3 and counts:
            unique_vals = set(counts)
            if len(unique_vals) == 2:
                # Exactly one of the two counts must occur once.
                for v in unique_vals:
                    if counts.count(v) == 1:
                        return grid, object_mask, sub_objs_mask
