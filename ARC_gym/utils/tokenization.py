import numpy as np

MAX_LENGTH = 931

def valid_grid(a):
    if len(a) > 30 or len(a) == 0 or len(a[0]) > 30 or len(a[0]) == 0:
        return False

    return True

# Expects tuples of tuples format, no tokenization
def valid_grid_batch(grid_batch):
    for grid in grid_batch:
        if not valid_grid(grid):
            return False

    return True

def tokenize_grid_batch(grid_batch, max_length=MAX_LENGTH):
    """Go from tuple-of-tuples representation to integer array of shape (max_length,).

    0: padding
    1: line return
    2: end of grid
    3-12: symbols
    """
    out = np.zeros((len(grid_batch), max_length,), dtype="int8")
    for batch_idx, grid in enumerate(grid_batch):
        i = 0
        for line in grid:
            for symbol in line:
                out[batch_idx, i] = symbol + 3
                i += 1
            out[batch_idx, i] = 1
            i += 1
        out[batch_idx, i] = 2
    return out

def tokenize_grid(grid, max_length=MAX_LENGTH):
    """Go from tuple-of-tuples representation to integer array of shape (max_length,).

    0: padding
    1: line return
    2: end of grid
    3-12: symbols
    """
    out = np.zeros((max_length,), dtype="int8")
    i = 0
    for line in grid:
        for symbol in line:
            out[i] = symbol + 3
            i += 1
        out[i] = 1
        i += 1
    out[i] = 2

    return out

def detokenize_grid_padded(a, assume_max_grid):
    padded_2d_grid = np.zeros((assume_max_grid, assume_max_grid))

    x = 0
    y = 1
    for idx in range(len(a)):
        if a[idx] == 0:
            continue

        if a[idx] >= 3:
            padded_2d_grid[assume_max_grid-y][x] = a[idx] - 3

        x += 1

        if a[idx] == 2:
            return padded_2d_grid

        if a[idx] == 1:
            y += 1
            x = 0

    print("==> ERROR: reached the end of the grid sequence without an end-of-grid token.")
    return padded_2d_grid

def detokenize_grid_unpadded(a):
    '''
    This function assumes the input sequence a is a token sequence representing a grid.
    The format of the sequence is as follows:
        - 0:
        - 1:
        - 2:
        - 3 to 12: colors 0 to 9 inclusively.

    It dynamically determines the dimensions of the grid from the above, and outputs a numpy
    array of shape (N, M) where N is the number of rows and M the number of columns, and the
    elements of this 2D array are the integers 0 to 9 representing the pixel colors.
    '''
    grid = []
    row = []
    x = 0
    y = 1
    for idx in range(len(a)):
        if a[idx] == 0:
            continue

        if a[idx] >= 3:
            row.append(a[idx] - 3)

        x += 1

        if a[idx] == 2:
            return tuple(grid)

        if a[idx] == 1:
            y += 1
            x = 0
            grid.append(tuple(row))
            row = []

    print("==> ERROR: reached the end of the grid sequence without an end-of-grid token.")
    return tuple(grid)