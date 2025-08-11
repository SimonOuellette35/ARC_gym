from ARC_gym.utils.object_detector import ObjectDetector
import random
import numpy as np
import json


def return_training_objects(training_examples, training_path, obj_category):
    
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
            return grid, object_mask

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
        return grid, object_mask

    # Find all unique object IDs (excluding background 0)
    unique_objects = np.unique(object_mask)
    unique_objects = unique_objects[unique_objects != 0]  # Remove background
    
    if len(unique_objects) == 0:
        # No objects in the grid, return the full grid
        return grid, object_mask
    
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
            return sub_grid, sub_object_mask
        
        attempts += 1
    
    # If no valid subgrid found after max attempts, return the full grid
    return grid, object_mask
    

def sample_distinct_colors_adjacent_training(training_path):
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

    return return_training_objects(training_examples, training_path, 'distinct_colors_adjacent')

def sample_distinct_colors_adjacent_empty_training(training_path):
    training_examples = [
        # simple empty shapes
        ('025d127b', 2),
        ('15663ba9', 0),
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
        ('8a371977', 0),
        ('42918530', 2),
        ('7d1f7ee8', 2),
        ('b7fb29bc', 0),
        ('d37a1ef5', 0),
        ('e7dd8335', 0)
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

def sample_corner_objects_training(training_path):
    training_examples = [
        ('15663ba9', 0),
        ('18419cfa', 2),
        ('4b6b68e5', 2),
        ('b9630600', 1)
    ]

    return return_training_objects(training_examples, training_path, 'distinct_colors_adjacent_empty')
    
def sample_fixed_size_2col_shapes_training(training_path):
    # To add as special case: ('e78887d1', 2)
    training_examples = [
        ('1c0d0a4b', 2),
        ('45737921', 2),
        ('337b420f', 0),
        ('3428a4f5', 0),
        ('34b99a2b', 0),
        ('39a8645d', 0),
        ('42918530', 2),
        ('4e45f183', 2),
        ('506d28a5', 0),
        ('5d2a5c43', 0),
        ('60b61512', 0),
        ('6430c8c4', 0),
        ('662c240a', 0),
        ('66f2d22f', 0),
        ('6a11f6da', 0),
        ('75b8110e', 0),
        ('760b3cac', 0),
        ('94f9d214', 0),
        ('99b1bc43', 0),
        ('bbb1b8b6', 0),
        ('cf98881b', 0),
        ('e133d23d', 0),
        ('e345f17b', 0),
        ('ea9794b1', 0)
    ]

    return return_training_objects(training_examples, training_path, 'fixed_size_2col_shapes')


def sample_distinct_colors_adjacent(training_path, min_dim=None, max_dim=None):
    if min_dim is None:
        min_dim = 3

    if max_dim is None:
        max_dim = 30

    a = np.random.uniform()

    if a < 0.25:
        return sample_distinct_colors_adjacent_training(training_path)

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
                    object_mask[row, col] = obj_id

    return grid, object_mask

def sample_corner_objects(training_path, min_dim=None, max_dim=None):
    if min_dim is None:
        min_dim = 16

    if max_dim is None:
        max_dim = 30

    a = np.random.uniform()

    if a < 0.1:
        return sample_corner_objects_training(training_path)

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

        if min_rect_size > max_rect_height // 2:
            rect_height = min_rect_size
        else:
            rect_height = np.random.randint(min_rect_size, max_rect_height // 2)
        if min_rect_size > max_rect_width // 2:
            rect_width = min_rect_size
        else:
            rect_width = np.random.randint(min_rect_size, max_rect_width // 2)

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

    return grid, object_mask


def sample_distinct_colors_adjacent_empty(training_path, min_dim=None, max_dim=None):
    if min_dim is None:
        min_dim = 3

    if max_dim is None:
        max_dim = 30

    a = np.random.uniform()

    if a < 0.25:
        return sample_distinct_colors_adjacent_empty_training(training_path)

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
        for attempt in range(max_attempts):
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
            object_mask[start_row:start_row + obj_height, start_col:start_col + obj_width] = obj_id

            found_spot = True
            break  # Successfully placed this object

        if not found_spot:
            # No more space for this object, stop placing further objects
            break
            
    return grid, object_mask

def sample_uniform_rect_noisy_bg(training_path, min_dim=None, max_dim=None, empty=False):
    if min_dim is None:
        min_dim = 6

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
    num_objects = np.random.randint(1, 7)
    object_colors = []
    
    # Generate unique colors for objects (different from background)
    available_colors = list(range(10))
    available_colors.remove(bg_color)
    object_colors = np.random.choice(available_colors, num_objects, replace=False)

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

            # No overlap, place the object
            object_mask[start_row:start_row + obj_height, start_col:start_col + obj_width] = obj_id

            if not empty:
                # Full rectangle
                grid[start_row:start_row + obj_height, start_col:start_col + obj_width] = obj_color
            else:
                # Empty rectangle (border only)
                # Top and bottom rows
                grid[start_row, start_col:start_col + obj_width] = obj_color
                grid[start_row + obj_height - 1, start_col:start_col + obj_width] = obj_color
                # Left and right columns (excluding corners already set)
                if obj_height > 2:
                    grid[start_row + 1:start_row + obj_height - 1, start_col] = obj_color
                    grid[start_row + 1:start_row + obj_height - 1, start_col + obj_width - 1] = obj_color

            break

    return grid, object_mask

def get_pattern(bg_color, pattern, num_patterns):

    def random_removal(pattern_grid, exclude=None):
        px_count = 0
        for row in pattern_grid:
            for val in row:
                if val != bg_color:
                    px_count += 1

        to_remove = np.random.randint(1, px_count // 2 + 1)

        # Randomly remove 'to_remove' non-bg_color pixels from pattern (replace with bg_color)
        # If exclude is not None, do not remove the pixel at 'exclude' (row, col)

        # Find all non-bg_color pixel coordinates
        coords = []
        for i, row in enumerate(pattern_grid):
            for j, val in enumerate(row):
                if val != bg_color:
                    if exclude is not None and (i, j) == tuple(exclude):
                        continue
                    coords.append((i, j))

        if len(coords) == 0 or to_remove == 0:
            return pattern_grid

        # If to_remove > available, just remove all
        to_remove = min(to_remove, len(coords))
        remove_coords = np.random.choice(len(coords), size=to_remove, replace=False)
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
            # empty plus with random removal
            col = np.random.choice([c for c in range(10) if c != bg_color])
            pattern_grid = [
                [bg_color, col, bg_color],
                [col, bg_color, col],
                [bg_color, col, bg_color]
            ]
            pattern_grid = random_removal(pattern_grid)
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
            pattern_grid = random_removal(pattern_grid)
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
            pattern_grid = random_removal(pattern_grid)
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
            pattern_grid = random_removal(pattern_grid)
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

def sample_incomplete_pattern(training_path, min_dim=None, max_dim=None, pattern='dot_plus'):
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
    object_colors = []
    
    # Generate unique colors for objects (different from background)
    available_colors = list(range(10))
    available_colors.remove(bg_color)
    object_colors = np.random.choice(available_colors, num_objects, replace=False)

    # IMPORTANT: Pass the actual grid background color to get_pattern
    # Randomly choose number of patterns to paste (1 to 6)
    num_patterns = np.random.randint(1, 7)

    pattern_grids, pattern_masks = get_pattern(bg_color, pattern, num_patterns)

    pattern_height, pattern_width = pattern_grids[0].shape
    mask_height, mask_width = pattern_masks[0].shape

    # 50% chance to add random noise to the background
    if np.random.random() < 0.5:
        # Find all colors used in all pattern_grids (excluding background)
        pattern_colors = set()
        for pg in pattern_grids:
            pattern_colors.update(np.unique(pg))
        if bg_color in pattern_colors:
            pattern_colors.remove(bg_color)
        # Also exclude all object colors (to be extra safe)
        pattern_colors.update(object_colors)
        # Choose a noise color that is not in pattern_colors or bg_color
        possible_noise_colors = [c for c in range(10) if c not in pattern_colors and c != bg_color]
        if possible_noise_colors:
            noise_color = np.random.choice(possible_noise_colors)
            # Number of noisy pixels: between 1 and 50% of the grid area
            num_noisy_pixels = np.random.randint(1, int(0.5 * num_rows * num_cols) + 1)
            # Generate random positions for the noise
            noisy_indices = np.unravel_index(
                np.random.choice(num_rows * num_cols, num_noisy_pixels, replace=False),
                (num_rows, num_cols)
            )
            grid[noisy_indices] = noise_color

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

    return grid, object_mask

def sample_fixed_size_2col_shapes(training_path, min_dim=None, max_dim=None):
    if min_dim is None:
        min_dim = 5

    if max_dim is None:
        max_dim = 30

    a = np.random.uniform()

    # TODO: temporary
    a = 0.1
    if a < 0.25:
        return sample_fixed_size_2col_shapes_training(training_path)

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
    max_obj_height = max(3, num_rows // 2)
    max_obj_width = max(3, num_cols // 2)
    obj_height = np.random.randint(3, max_obj_height + 1)
    obj_width = np.random.randint(3, max_obj_width + 1)

    # Choose exactly 2 random colors between 0 and 9 (inclusive)
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
            
    return grid, object_mask

def sample_incomplete_rectangles(training_path, min_dim=None, max_dim=None, all_same_shape=False):
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
            
    return grid, object_mask
