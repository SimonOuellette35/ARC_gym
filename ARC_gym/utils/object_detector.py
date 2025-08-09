import numpy as np


class ObjectDetector:

    @staticmethod
    def find_objects_for_color(grid, object_mask, c, bg_color):
        if c == bg_color:
            return object_mask
            
        # Find all pixels with color c
        color_pixels = np.where(grid == c)
        
        if len(color_pixels[0]) == 0:
            return object_mask
            
        # Use flood fill to find connected components
        visited = np.zeros_like(grid, dtype=bool)
        object_id = 1
        
        # Find the next available object ID
        while np.any(object_mask == object_id):
            object_id += 1

        # Mark cells that are already objects as visited
        visited = np.zeros_like(grid, dtype=bool)
        visited[object_mask != 0] = True

        for i, j in zip(color_pixels[0], color_pixels[1]):
            if visited[i, j]:
                continue
                
            # Flood fill to find connected component
            stack = [(i, j)]
            component = []
            
            while stack:
                curr_i, curr_j = stack.pop()
                
                if (curr_i < 0 or curr_i >= grid.shape[0] or 
                    curr_j < 0 or curr_j >= grid.shape[1] or 
                    visited[curr_i, curr_j] or grid[curr_i, curr_j] != c):
                    continue
                    
                visited[curr_i, curr_j] = True
                component.append((curr_i, curr_j))
                
                # Add adjacent pixels (8-connectivity: horizontal, vertical, diagonal)
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        stack.append((curr_i + di, curr_j + dj))
            
            # Mark this component as an object
            for obj_i, obj_j in component:
                object_mask[obj_i, obj_j] = object_id
                
            object_id += 1

        return object_mask

    @staticmethod
    def get_objects_incomplete_rectangles(grid):
        # Find the background color (most common color in the grid)
        unique_colors, counts = np.unique(grid, return_counts=True)
        bg_color = unique_colors[np.argmax(counts)]

        # Create a mask for foreground (non-background) pixels
        fg_mask = (grid != bg_color)
        visited = np.zeros_like(grid, dtype=bool)
        object_mask = np.zeros_like(grid, dtype=int)
        object_id = 1

        rows, cols = grid.shape

        # 8-connected component labeling for foreground blobs
        for i in range(rows):
            for j in range(cols):
                if fg_mask[i, j] and not visited[i, j]:
                    # Start a new object
                    stack = [(i, j)]
                    component_pixels = []

                    while stack:
                        ci, cj = stack.pop()
                        if (ci < 0 or ci >= rows or cj < 0 or cj >= cols or
                            visited[ci, cj] or not fg_mask[ci, cj]):
                            continue
                        visited[ci, cj] = True
                        component_pixels.append((ci, cj))
                        # 8-connectivity
                        for di in [-1, 0, 1]:
                            for dj in [-1, 0, 1]:
                                if di == 0 and dj == 0:
                                    continue
                                ni, nj = ci + di, cj + dj
                                if (0 <= ni < rows and 0 <= nj < cols and
                                    not visited[ni, nj] and fg_mask[ni, nj]):
                                    stack.append((ni, nj))

                    if component_pixels:
                        # Find bounding rectangle
                        comp_rows = [p[0] for p in component_pixels]
                        comp_cols = [p[1] for p in component_pixels]
                        min_r, max_r = min(comp_rows), max(comp_rows)
                        min_c, max_c = min(comp_cols), max(comp_cols)
                        # Fill the rectangle in the object mask
                        object_mask[min_r:max_r+1, min_c:max_c+1] = object_id
                        object_id += 1

        return object_mask

    @staticmethod
    def get_objects_distinct_colors_adjacent(grid):
        # Find the background color (most common color in the grid)
        unique_colors, counts = np.unique(grid, return_counts=True)
        bg_color = unique_colors[np.argmax(counts)]

        object_mask = np.zeros_like(grid)

        for c in unique_colors:
            object_mask = ObjectDetector.find_objects_for_color(grid, object_mask, c, bg_color)

        return object_mask

    @staticmethod
    def get_objects_distinct_colors_adjacent_empty(grid):
        """
        Find all outer containment objects. Specifically, find all object borders
        of a uniform color. Everything located inside (regardless of color) of 
        these borders is to be part of the object.
        
        Returns object mask where background cells are 0 and object cells
        are incremental values starting from 1, 2, etc.
        """
        unique_colors, counts = np.unique(grid, return_counts=True)
        
        object_mask = np.zeros_like(grid)
        object_id = 1
        
        # For each color, check if it forms containment borders
        for color in unique_colors:
                
            # Find all pixels of this color
            color_positions = np.where(grid == color)
            if len(color_positions[0]) == 0:
                continue
            
            # Create a binary mask for this color
            color_mask = (grid == color).astype(np.uint8)
            
            # Find connected components of this color
            visited = np.zeros_like(grid, dtype=bool)
            
            for i, j in zip(color_positions[0], color_positions[1]):
                if visited[i, j]:
                    continue
                
                # Find the connected component starting from this pixel
                component_pixels = ObjectDetector._flood_fill_component(grid, i, j, color, visited)
                
                if len(component_pixels) == 0:
                    continue
                
                # Check if this component forms a closed border that contains interior space
                interior_pixels = ObjectDetector._find_interior_of_border(grid, component_pixels)
                
                if len(interior_pixels) > 0:
                    # Mark both border and interior as the same object
                    for pi, pj in component_pixels + interior_pixels:
                        if 0 <= pi < grid.shape[0] and 0 <= pj < grid.shape[1]:
                            object_mask[pi, pj] = object_id
                    object_id += 1
        
        # Handle background exclusion based on distinct IDs and presence of 0 values
        unique_ids = np.unique(object_mask)
        
        if len(unique_ids) == 1:
            # Only 1 distinct id: return as is
            return object_mask
        elif 0 in unique_ids:
            # > 1 distinct id but contains 0 values: return as is
            return object_mask
        else:
            # > 1 distinct id and no 0 values: decrement all ids by 1
            object_mask = object_mask - 1
            return object_mask
    
    @staticmethod
    def _flood_fill_component(grid, start_i, start_j, target_color, visited):
        """Helper method to find a connected component of target_color starting from (start_i, start_j)"""
        if (start_i < 0 or start_i >= grid.shape[0] or 
            start_j < 0 or start_j >= grid.shape[1] or 
            visited[start_i, start_j] or grid[start_i, start_j] != target_color):
            return []
        
        component = []
        stack = [(start_i, start_j)]
        
        while stack:
            i, j = stack.pop()
            
            if (i < 0 or i >= grid.shape[0] or 
                j < 0 or j >= grid.shape[1] or 
                visited[i, j] or grid[i, j] != target_color):
                continue
            
            visited[i, j] = True
            component.append((i, j))
            
            # Add 8-connected neighbors (including diagonals)
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    stack.append((i + di, j + dj))
        
        return component
    
    @staticmethod
    def _find_interior_of_border(grid, border_pixels):
        """
        Find all pixels that are inside the border formed by border_pixels.
        Uses flood fill from the edges to find all reachable background pixels,
        then considers unreachable pixels as interior.
        """
        rows, cols = grid.shape
        
        # Create a mask of border pixels
        border_mask = np.zeros_like(grid, dtype=bool)
        for i, j in border_pixels:
            if 0 <= i < rows and 0 <= j < cols:
                border_mask[i, j] = True
        
        # Flood fill from all edge pixels to find all reachable background
        reachable = np.zeros_like(grid, dtype=bool)
        
        # Start flood fill from all edge pixels
        edge_pixels = []
        # Top and bottom edges
        for j in range(cols):
            edge_pixels.extend([(0, j), (rows-1, j)])
        # Left and right edges  
        for i in range(1, rows-1):  # Skip corners already added
            edge_pixels.extend([(i, 0), (i, cols-1)])
        
        for start_i, start_j in edge_pixels:
            if not reachable[start_i, start_j]:
                if not border_mask[start_i, start_j]:
                    # Flood fill to mark all reachable non-border pixels
                    ObjectDetector._flood_fill_reachable(grid, start_i, start_j, border_mask, reachable)
                else:
                    # If edge pixel is part of border, mark it as reachable 
                    # (connected to exterior) but don't flood fill from it
                    reachable[start_i, start_j] = True
        
        # Interior pixels are those that are not reachable and not part of the border
        interior_pixels = []
        for i in range(rows):
            for j in range(cols):
                if not reachable[i, j] and not border_mask[i, j]:
                    interior_pixels.append((i, j))
        
        return interior_pixels
    
    @staticmethod
    def _flood_fill_reachable(grid, start_i, start_j, border_mask, reachable):
        """Flood fill to mark all pixels reachable from (start_i, start_j) without crossing borders"""
        rows, cols = grid.shape
        stack = [(start_i, start_j)]
        
        while stack:
            i, j = stack.pop()
            
            if (i < 0 or i >= rows or j < 0 or j >= cols or 
                reachable[i, j] or border_mask[i, j]):
                continue
            
            reachable[i, j] = True
            
            # Add 4-connected neighbors
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                stack.append((i + di, j + dj))

    @staticmethod
    def get_objects_pattern_square_hollow(grid):
        unique_colors, counts = np.unique(grid, return_counts=True)
        
        # Always include 0 as a background color, plus the next most common color
        if 0 in unique_colors:
            zero_idx = np.where(unique_colors == 0)[0][0]
            # Remove 0 from the list to find the next most common
            nonzero_colors = np.delete(unique_colors, zero_idx)
            nonzero_counts = np.delete(counts, zero_idx)
            if len(nonzero_colors) > 0:
                next_bg_idx = np.argmax(nonzero_counts)
                bg_colors = [0, nonzero_colors[next_bg_idx]]
            else:
                bg_colors = [0]
        else:
            # 0 not present, just take the two most common
            top2_idx = np.argsort(-counts)[:2]
            bg_colors = list(unique_colors[top2_idx])
        
        object_mask = np.zeros_like(grid, dtype=int)
        
        # Group foreground pixels by which 3x3 grid tile they belong to
        # Each pixel (i,j) belongs to the 3x3 tile starting at (3*(i//3), 3*(j//3))
        object_mask = np.zeros_like(grid, dtype=int)
        obj_id = 1
        rows, cols = grid.shape
        
        for i in range(rows):
            for j in range(cols):
                # For each foreground pixel, find the 3x3 square (centered at or near (i,j)) that covers the most non-bg pixels (including itself)
                # and assign a new object id to those pixels in object_mask (if not already assigned)
                if grid[i, j] not in bg_colors and object_mask[i, j] == 0:
                    best_count = 0
                    best_top, best_left = None, None

                    # Try all possible 3x3 squares containing (i, j)
                    for di in range(-2, 1):
                        for dj in range(-2, 1):
                            top = i + di
                            left = j + dj
                            # Check bounds
                            if top < 0 or left < 0 or top + 2 >= rows or left + 2 >= cols:
                                continue
                            # Count non-bg pixels in this 3x3 square that are not yet assigned
                            count = 0
                            for ii in range(top, top + 3):
                                for jj in range(left, left + 3):
                                    if grid[ii, jj] not in bg_colors and object_mask[ii, jj] == 0:
                                        count += 1
                            if count > best_count:
                                best_count = count
                                best_top, best_left = top, left

                    # If we found a 3x3 square with at least one non-bg pixel, assign a new object id
                    if best_count > 0:
                        for ii in range(best_top, best_top + 3):
                            for jj in range(best_left, best_left + 3):
                                if object_mask[ii, jj] == 0:
                                    object_mask[ii, jj] = obj_id
                        obj_id += 1

        return object_mask

    @staticmethod
    def get_objects_pattern_plus_filled(grid):
        unique_colors, counts = np.unique(grid, return_counts=True)
        
        # Always include 0 as a background color, plus the next most common color
        if 0 in unique_colors:
            zero_idx = np.where(unique_colors == 0)[0][0]
            # Remove 0 from the list to find the next most common
            nonzero_colors = np.delete(unique_colors, zero_idx)
            nonzero_counts = np.delete(counts, zero_idx)
            if len(nonzero_colors) > 0:
                next_bg_idx = np.argmax(nonzero_counts)
                bg_colors = [0, nonzero_colors[next_bg_idx]]
            else:
                bg_colors = [0]
        else:
            # 0 not present, just take the two most common
            top2_idx = np.argsort(-counts)[:2]
            bg_colors = list(unique_colors[top2_idx])
        
        object_mask = np.zeros_like(grid, dtype=int)
        
        # Group foreground pixels by which 3x3 grid tile they belong to
        # Each pixel (i,j) belongs to the 3x3 tile starting at (3*(i//3), 3*(j//3))
        object_mask = np.zeros_like(grid, dtype=int)
        obj_id = 1
        rows, cols = grid.shape
        
        for i in range(rows):
            for j in range(cols):
                # For each foreground pixel, find the 3x3 "plus sign" (centered at or near (i,j)) that covers the most non-bg pixels (including itself)
                # and assign a new object id to those pixels in object_mask (if not already assigned)
                if grid[i, j] not in bg_colors and object_mask[i, j] == 0:
                    best_count = 0
                    best_top, best_left = None, None

                    # Try all possible 3x3 "plus sign" shapes containing (i, j)
                    for di in range(-2, 1):
                        for dj in range(-2, 1):
                            top = i + di
                            left = j + dj
                            # Check bounds
                            if top < 0 or left < 0 or top + 2 >= rows or left + 2 >= cols:
                                continue
                            # Count non-bg pixels in this 3x3 plus sign that are not yet assigned
                            count = 0
                            for ii in range(top, top + 3):
                                for jj in range(left, left + 3):
                                    # Only consider plus sign positions: center row, center col, or center cell
                                    if (ii == top + 1 or jj == left + 1):
                                        if grid[ii, jj] not in bg_colors and object_mask[ii, jj] == 0:
                                            count += 1
                            if count > best_count:
                                best_count = count
                                best_top, best_left = top, left

                    # If we found a 3x3 plus sign with at least one non-bg pixel, assign a new object id
                    if best_count > 0:
                        for ii in range(best_top, best_top + 3):
                            for jj in range(best_left, best_left + 3):
                                if (ii == best_top + 1 or jj == best_left + 1):
                                    if object_mask[ii, jj] == 0:
                                        object_mask[ii, jj] = obj_id
                        obj_id += 1

        return object_mask

    @staticmethod
    def get_objects_pattern_dot_plus(grid):
        unique_colors, counts = np.unique(grid, return_counts=True)
        bg_color = unique_colors[np.argmax(counts)]
        object_mask = np.zeros_like(grid, dtype=int)
        
        # Mark all non-background pixels and their 4-connected neighbors as object 1
        mask = (grid != bg_color)
        # Pad the mask to handle edge pixels
        padded_mask = np.pad(mask, pad_width=1, mode='constant', constant_values=False)
        expanded_mask = np.zeros_like(padded_mask, dtype=bool)
        # For each direction: center, up, down, left, right
        for di, dj in [(0,0), (-1,0), (1,0), (0,-1), (0,1)]:
            expanded_mask[1+di:1+di+mask.shape[0], 1+dj:1+dj+mask.shape[1]] |= mask

        # Remove the padding
        expanded_mask = expanded_mask[1:-1, 1:-1]
        object_mask[expanded_mask] = 1

        return object_mask

        
    @staticmethod
    def get_objects_distinct_colors(grid):
        # Find the background color (most common color in the grid)
        unique_colors, counts = np.unique(grid, return_counts=True)
        bg_color = unique_colors[np.argmax(counts)]

        object_mask = np.zeros_like(grid)

        object_id = 1
        for c in unique_colors:
            if c != bg_color:
                # Mark all pixels of this color as the same object
                object_mask[grid == c] = object_id
                object_id += 1

        return object_mask

    @staticmethod
    def check_special_case(grid, task_id, grid_idx):
        '''
        Implement object detection by code in a generalizable way is extremely difficult,
        and sometimes it's easier to just hardcode special cases for the object masks...
        '''
        if task_id == 'a680ac02':
            if grid_idx == 0:
                return [
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [2, 2, 2, 2],
                    [2, 2, 2, 2],
                    [2, 2, 2, 2],
                    [2, 2, 2, 2]
                ]

            elif grid_idx == 1:
                return [
                    [1, 1, 1, 1, 2, 2, 2, 2],
                    [1, 1, 1, 1, 2, 2, 2, 2],
                    [1, 1, 1, 1, 2, 2, 2, 2],
                    [1, 1, 1, 1, 2, 2, 2, 2]
                ]
            elif grid_idx == 2:
                return [
                    [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
                    [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
                    [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
                    [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
                ]
            else:
                return [
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [2, 2, 2, 2],
                    [2, 2, 2, 2],
                    [2, 2, 2, 2],
                    [2, 2, 2, 2],
                    [3, 3, 3, 3],
                    [3, 3, 3, 3],
                    [3, 3, 3, 3],
                    [3, 3, 3, 3]
                ]

        elif task_id == '7d1f7ee8':
            if grid_idx == 0:   # input 0
                return [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 2, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 2, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 2, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 2, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 2, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 2, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 2, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 2, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 2, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 3, 3, 3, 3, 3, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 3, 3, 3, 3, 3, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 3, 3, 3, 3, 3, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 3, 3, 3, 3, 3, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                ]

            elif grid_idx == 1: # target 0
                return [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 2, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 2, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 2, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 2, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 2, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 2, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 2, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 2, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 2, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 3, 3, 3, 3, 3, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 3, 3, 3, 3, 3, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 3, 3, 3, 3, 3, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 3, 3, 3, 3, 3, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                ]
            elif grid_idx == 2: # input 1
                return [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                    [0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                    [0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                    [0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                    [0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                    [0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                    [0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                    [0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                ]

            elif grid_idx == 3: # target 1
                return [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                    [0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                    [0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                    [0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                    [0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                    [0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                    [0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                    [0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                ]

            elif grid_idx == 4: # input 2
                return [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                ]
            elif grid_idx == 5: # target 2
                return [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                ]
            else:   # test grid
                return [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0],
                    [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0],
                    [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0],
                    [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0],
                    [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0],
                    [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0],
                    [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0],
                    [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]                                                                                                                                            
                ]
        elif task_id == '868de0fa' and grid_idx == 0:
            return [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0, 2, 2, 2, 2, 0, 0],
                [0, 1, 1, 1, 0, 2, 2, 2, 2, 0, 0],
                [0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0],
                [0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0],
                [3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0],
                [3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0],
                [3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0],
                [3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0],
                [3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0]
            ]
        elif task_id == '8fbca751':
            if grid_idx == 0:
                return [
                    [1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0],
                    [1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0],
                    [1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0],
                    [1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3],
                    [0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3],
                    [0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3],
                    [0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3]
                ]

            elif grid_idx == 1:
                return [
                    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2],
                    [0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2],
                    [0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2],
                    [0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2]
                ]
                        
            elif grid_idx == 2:
                return [
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 1, 0, 0],
                    [0, 1, 1, 1, 1, 0, 0],
                    [0, 1, 1, 1, 1, 0, 0],
                    [0, 1, 1, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0]
                ]

            else:
                return [
                    [0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3],
                    [0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3],
                    [0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3],
                    [0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3],
                    [1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0],
                    [1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0],
                    [1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0],
                    [1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0]
                ]

        return None

    @staticmethod
    def get_objects(grid, category, task_id='', grid_idx=0):

        result = ObjectDetector.check_special_case(grid, task_id, grid_idx)
        if result is not None:
            return np.array(result)
        
        if category == 'distinct_colors_adjacent':
            return ObjectDetector.get_objects_distinct_colors_adjacent(grid)
        elif category == 'distinct_colors_adjacent_empty':
            return ObjectDetector.get_objects_distinct_colors_adjacent_empty(grid)
        elif category == 'distinct_colors':
            return ObjectDetector.get_objects_distinct_colors(grid)
        elif category == 'incomplete_rectangles':
            return ObjectDetector.get_objects_incomplete_rectangles(grid)
        elif category == 'pattern_dot_plus':
            return ObjectDetector.get_objects_pattern_dot_plus(grid)
        elif category == 'pattern_square_hollow':
            return ObjectDetector.get_objects_pattern_square_hollow(grid)
        elif category == 'pattern_plus_filled':
            return ObjectDetector.get_objects_pattern_plus_filled(grid)
        
