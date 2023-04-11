import numpy as np
import random

# A "grid" is a rectangular matrix (list of lists) of integers between 0 and 9 (inclusive).
# The smallest possible grid size is 1x1 and the largest is 30x30.

def generateEmptyGrid(color, dim):
    return np.ones((dim, dim)) * color

def getRandomColor(from_array):
    return np.random.choice(from_array)

# NOTE: this is actually an objectness task: a row (line) is a dense grouping of pixels.
def generateRandomPixelRows(num_groups=None):
    random_dim = np.random.choice(np.arange(1, 30))
    output_grid = generateEmptyGrid(0, random_dim)

    selected_colors = np.arange(1, 10)
    if num_groups is not None:
        # select num_groups distinct colors
        color_choices = list(np.arange(1, 10))
        random.shuffle(color_choices)
        selected_colors = color_choices[:num_groups]

    sparsity = np.random.uniform(0.1, 0.5)

    for x in range(random_dim):
        r = np.random.uniform()
        if r < sparsity:
            def drawRow(x, num, color):
                for i in range(num):
                    output_grid[x, i] = color

            num = np.random.choice(np.arange(1, random_dim))
            drawRow(x, num, getRandomColor(selected_colors))

    return output_grid

def decimalToBinary(decValue, dim=16):
    binValue = np.zeros(dim)
    tmp_bin_list = [int(x) for x in list('{0:0b}'.format(decValue))]
    binValue[-len(tmp_bin_list):] = tmp_bin_list

    return binValue

def generateRandomPixels(num_groups=None,
                         max_pixels_per_color=15,
                         max_pixels_total=None,
                         grid_dim_min=3,
                         grid_dim_max=30,
                         sparsity=None):
    if num_groups is not None:
        grid_dim_min = num_groups

    random_dim = np.random.choice(np.arange(grid_dim_min, grid_dim_max+1))
    output_grid = generateEmptyGrid(0, random_dim)

    selected_colors = np.arange(1, 10)
    if num_groups is not None:
        # select num_groups distinct colors
        color_choices = list(np.arange(1, 10))
        random.shuffle(color_choices)
        selected_colors = color_choices[:num_groups]

    if sparsity is None:
        sparsity = np.random.uniform()

    total_px_count = 0
    pixel_count_dict = {}
    if num_groups is not None:
        for i in range(num_groups):
            pixel_count_dict[selected_colors[i]] = 0

    for x in range(random_dim):
        for y in range(random_dim):
            if max_pixels_total is not None and total_px_count >= max_pixels_total:
                break

            r = np.random.uniform()
            if r < sparsity:
                tmp_color = getRandomColor(selected_colors)
                if tmp_color in pixel_count_dict and pixel_count_dict[tmp_color] >= max_pixels_per_color:
                    continue

                output_grid[x, y] = tmp_color
                if tmp_color in pixel_count_dict:
                    pixel_count_dict[tmp_color] += 1
                else:
                    pixel_count_dict[tmp_color] = 1

                total_px_count += 1

    # Note: by chance it can happen that the grid is empty, but we want at least 1 pixel
    if num_groups is None:
        if np.sum(list(pixel_count_dict.values())) == 0:
            x = np.random.choice(np.arange(random_dim))
            y = np.random.choice(np.arange(random_dim))
            output_grid[x, y] = getRandomColor(np.arange(1, 10))
    else:
        for color_idx, pc in enumerate(list(pixel_count_dict.values())):
            if pc == 0:
                found_free_cell = False
                max_attempts = 10
                while not found_free_cell:
                    x = np.random.choice(np.arange(random_dim))
                    y = np.random.choice(np.arange(random_dim))
                    if output_grid[x, y] == 0:
                        output_grid[x, y] = selected_colors[color_idx]
                        found_free_cell = True
                    else:
                        max_attempts -= 1
                        if max_attempts <= 0:
                            # give up on trying to free up a cell, overwrite...
                            output_grid[x, y] = selected_colors[color_idx]
                            found_free_cell = True

    return output_grid

def colorCount(input_grid):
    color_count = {}
    for x in range(input_grid.shape[0]):
        for y in range(input_grid.shape[1]):
            if input_grid[x, y] != 0:
                if input_grid[x, y] in color_count:
                    color_count[input_grid[x, y]] += 1
                else:
                    color_count[input_grid[x, y]] = 1

    return color_count

def pixelCount(input_grid):
    pixel_count = 0
    for x in range(input_grid.shape[0]):
        for y in range(input_grid.shape[1]):
            if input_grid[x, y] != 0:
                pixel_count += 1

    return pixel_count

def perColorPixelCount(input_grid):
    pixel_count = np.zeros(9)
    for x in range(input_grid.shape[0]):
        for y in range(input_grid.shape[1]):
            if input_grid[x, y] != 0:
                idx = int(input_grid[x, y] - 1)
                pixel_count[idx] += 1

    return pixel_count


def colorFillGrid(input_grid, color):
    output_grid = np.copy(input_grid)

    for x in range(output_grid.shape[0]):
        for y in range(output_grid.shape[1]):
            output_grid[x, y] = color

    return output_grid