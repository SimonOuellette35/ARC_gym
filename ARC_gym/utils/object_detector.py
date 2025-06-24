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
    def get_objects_distinct_colors_adjacent(grid):
        # Find the background color (most common color in the grid)
        unique_colors, counts = np.unique(grid, return_counts=True)
        bg_color = unique_colors[np.argmax(counts)]

        object_mask = np.zeros_like(grid)

        for c in unique_colors:
            object_mask = ObjectDetector.find_objects_for_color(grid, object_mask, c, bg_color)

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
    def get_objects(grid, category):

        if category == 'distinct_colors_adjacent':
            return ObjectDetector.get_objects_distinct_colors_adjacent(grid)
        elif category == 'distinct_colors':
            return ObjectDetector.get_objects_distinct_colors(grid)