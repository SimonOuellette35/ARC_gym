import numpy as np
from functools import partial
from ARC_gym.dataset import COLOR_MAP


def get_total_set():
    return [
        (copy_right, 'copy_right'),
        (copy_left, 'copy_left'),
        (copy_up, 'copy_up'),
        (copy_down, 'copy_down'),
        (move_right, 'move_right'),
        (move_left, 'move_left'),
        (move_up, 'move_up'),
        (move_down, 'move_down'),
        (rotate_90_degrees, 'rotate_90_degrees'),
        (rotate_180_degrees, 'rotate_180_degrees'),
        (rotate_270_degrees, 'rotate_270_degrees'),
        (flip_vertical, 'flip_vertical'),
        (flip_horizontal, 'flip_horizontal'),
        (draw_horizontal_split, 'draw_horizontal_split'),
        (draw_vertical_split, 'draw_vertical_split'),
        (draw_diagonal1, 'draw_diagonal1'),
        (draw_diagonal2, 'draw_diagonal2'),
        (partial(swap_pixels, from_color=1, to_color=2), 'swap_pixels(%s,%s)' % (COLOR_MAP[1], COLOR_MAP[2])),
        (partial(swap_pixels, from_color=1, to_color=5), 'swap_pixels(%s,%s)' % (COLOR_MAP[1], COLOR_MAP[5])),
        (partial(swap_pixels, from_color=1, to_color=7), 'swap_pixels(%s,%s)' % (COLOR_MAP[1], COLOR_MAP[7])),
        (partial(swap_pixels, from_color=3, to_color=5), 'swap_pixels(%s,%s)' % (COLOR_MAP[3], COLOR_MAP[5])),
        (partial(swap_pixels, from_color=3, to_color=6), 'swap_pixels(%s,%s)' % (COLOR_MAP[3], COLOR_MAP[6]))
        # (partial(swap_pixels, from_color=3, to_color=7), 'swap_pixels(%s,%s)' % (COLOR_MAP[3], COLOR_MAP[7])),
        # (partial(swap_pixels, from_color=8, to_color=1), 'swap_pixels(%s,%s)' % (COLOR_MAP[8], COLOR_MAP[1])),
        # (partial(swap_pixels, from_color=8, to_color=2), 'swap_pixels(%s,%s)' % (COLOR_MAP[8], COLOR_MAP[2])),
        # (partial(swap_pixels, from_color=8, to_color=9), 'swap_pixels(%s,%s)' % (COLOR_MAP[8], COLOR_MAP[9])),
        # (partial(swap_pixels, from_color=8, to_color=4), 'swap_pixels(%s,%s)' % (COLOR_MAP[8], COLOR_MAP[4]))
    ]

def draw_horizontal_split(grid):
    result = np.copy(grid)
    D = grid.shape[0]
    for i in range(D):
        result[int(D/2), i] = 3

    return result

def draw_vertical_split(grid):
    result = np.copy(grid)
    D = grid.shape[0]
    for i in range(D):
        result[i, int(D / 2)] = 6

    return result

def draw_diagonal1(grid):
    result = np.copy(grid)
    D = grid.shape[0]
    for i in range(D):
        result[i, i] = 8

    return result

def draw_diagonal2(grid):
    result = np.copy(grid)
    D = grid.shape[0]
    for i in range(D):
        result[i, D-i-1] = 9

    return result

def swap_pixels(grid, from_color, to_color):
    result = np.copy(grid)
    result[result == from_color] = 99  # temporarily change to an arbitrary value
    result[result == to_color] = from_color
    result[result == 99] = to_color
    return result

def copy_right(grid):
    result = np.copy(grid)
    D = grid.shape[0]
    for i in range(D):
        for j in range(D-1):
            if grid[i, j] != 0:
                if grid[i, j+1] == 0:
                    result[i, j+1] = grid[i, j]

    return result

def copy_left(grid):
    result = np.copy(grid)
    D = grid.shape[0]
    for i in range(D):
        for j in range(1, D):
            if grid[i, j] != 0:
                if grid[i, j-1] == 0:
                    result[i, j-1] = grid[i, j]

    return result

def copy_down(grid):
    result = np.copy(grid)
    D = grid.shape[0]
    for i in range(1, D):
        for j in range(D):
            if grid[i, j] != 0:
                if grid[i-1, j] == 0:
                    result[i-1, j] = grid[i, j]

    return result

def copy_up(grid):
    result = np.copy(grid)
    D = grid.shape[0]
    for i in range(D-1):
        for j in range(D):
            if grid[i, j] != 0:
                if grid[i+1, j] == 0:
                    result[i+1, j] = grid[i, j]

    return result

def move_right(grid):
    result = np.zeros_like(grid)
    D = grid.shape[0]
    for i in range(D):
        for j in range(D):
            if grid[i, j] != 0:
                if j+1 >= D:
                    result[i, 0] = grid[i, j]
                else:
                    result[i, j+1] = grid[i, j]

    return result

def move_left(grid):
    result = np.zeros_like(grid)
    D = grid.shape[0]
    for i in range(D):
        for j in range(D):
            if grid[i, j] != 0:
                if j-1 < 0:
                    result[i, D-1] = grid[i,j]
                else:
                    result[i, j-1] = grid[i, j]

    return result

def move_down(grid):
    result = np.zeros_like(grid)
    D = grid.shape[0]
    for i in range(D):
        for j in range(D):
            if grid[i, j] != 0:
                if i-1 < 0:
                    result[D-1, j] = grid[i, j]
                else:
                    result[i-1, j] = grid[i, j]

    return result

def move_up(grid):
    result = np.zeros_like(grid)
    D = grid.shape[0]
    for i in range(D):
        for j in range(D):
            if grid[i, j] != 0:
                if i+1 >= D:
                    result[0, j] = grid[i, j]
                else:
                    result[i+1, j] = grid[i, j]

    return result

def rotate_90_degrees(grid):
    return np.rot90(grid)

def rotate_180_degrees(grid):
    return np.rot90(grid, 2)

def rotate_270_degrees(grid):
    return np.rot90(grid, -1)

def flip_horizontal(grid):
    return np.flip(grid, axis=1)

def flip_vertical(grid):
    return np.flip(grid, axis=0)
