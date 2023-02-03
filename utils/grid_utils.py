import numpy as np

# A "grid" is a rectangular matrix (list of lists) of integers between 0 and 9 (inclusive).
# The smallest possible grid size is 1x1 and the largest is 30x30.

def generateEmptyGrid(color, dim):
    return np.ones((dim, dim)) * color

def getRandomColor():
    return np.random.choice(np.arange(1, 10))

def generateRandomPixels():
    random_dim = np.random.choice(np.arange(1, 30))
    output_grid = generateEmptyGrid(0, random_dim)

    sparsity = np.random.uniform()

    pixel_count = 0.
    for x in range(random_dim):
        for y in range(random_dim):
            r = np.random.uniform()
            if r < sparsity:
                output_grid[x, y] = getRandomColor()
                pixel_count += 1

    # Note: by chance it can happen that the grid is empty, but we want at least 1 pixel
    if pixel_count == 0:
        x = np.random.choice(np.arange(random_dim))
        y = np.random.choice(np.arange(random_dim))
        output_grid[x, y] = getRandomColor()

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


def colorFillGrid(input_grid, color):
    output_grid = np.copy(input_grid)

    for x in range(output_grid.shape[0]):
        for y in range(output_grid.shape[1]):
            output_grid[x, y] = color

    return output_grid