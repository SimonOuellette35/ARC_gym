import matplotlib.pyplot as plt
import numpy as np

# Define a color map
COLOR_MAP = {
    0: 'black',
    1: 'steelblue',
    2: 'green',
    3: 'yellow',
    4: 'purple',
    5: 'orange',
    6: 'red',
    7: 'salmon',
    8: 'aquamarine',
    9: 'white',
    10: 'lightcoral',
    11: 'brown',
    12: 'pink',
    13: 'gray',
    14: 'navy',
    15: 'lime',
    16: 'crimson',
    17: 'gold',
    18: 'darkblue'
}


def draw_single_grid(grid):

    for y in range(len(grid)-1, -1, -1):
        for x in range(len(grid[0])):
            color_idx = int(grid[y][x])
            color = COLOR_MAP[color_idx]
            rectangle = plt.Rectangle((x, len(grid)-1-y), 1, 1, fc=color, edgecolor='black')
            plt.gca().add_patch(rectangle)

    plt.xlim(0, len(grid[0]))
    plt.ylim(0, len(grid))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')
    plt.show()

def draw_grid_pair(grid1, grid2, title='', grid_size=5):

    plt.figure(figsize=(grid_size * 2, grid_size * 1))

    plt.subplot(1, 2, 1)
    for y in range(len(grid1)-1, -1, -1):
        for x in range(len(grid1[0])):
            color_idx = int(grid1[y][x])
            color = COLOR_MAP[color_idx]
            rectangle = plt.Rectangle((x, len(grid1)-1-y), 1, 1, fc=color, edgecolor='black')
            plt.gca().add_patch(rectangle)

    plt.xlim(0, len(grid1[0]))
    plt.ylim(0, len(grid1))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    for y in range(len(grid2)-1, -1, -1):
        for x in range(len(grid2[0])):
            color_idx = int(grid2[y][x])
            color = COLOR_MAP[color_idx]
            rectangle = plt.Rectangle((x, len(grid2)-1-y), 1, 1, fc=color, edgecolor='black')
            plt.gca().add_patch(rectangle)

    plt.xlim(0, len(grid2[0]))
    plt.ylim(0, len(grid2))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def draw_grid_triple(grid1, grid2, grid3, title='', grid_size=5):

    plt.figure(figsize=(grid_size * 3, grid_size * 1))

    plt.subplot(1, 3, 1)
    for y in range(len(grid1)-1, -1, -1):
        for x in range(len(grid1[0])):
            color_idx = int(grid1[y][x])
            color = COLOR_MAP[color_idx]
            rectangle = plt.Rectangle((x, len(grid1)-1-y), 1, 1, fc=color, edgecolor='black')
            plt.gca().add_patch(rectangle)

    plt.xlim(0, len(grid1[0]))
    plt.ylim(0, len(grid1))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    for y in range(len(grid2)-1, -1, -1):
        for x in range(len(grid2[0])):
            color_idx = int(grid2[y][x])
            color = COLOR_MAP[color_idx]
            rectangle = plt.Rectangle((x, len(grid2)-1-y), 1, 1, fc=color, edgecolor='black')
            plt.gca().add_patch(rectangle)

    plt.xlim(0, len(grid2[0]))
    plt.ylim(0, len(grid2))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    for y in range(len(grid3)-1, -1, -1):
        for x in range(len(grid3[0])):
            color_idx = int(grid3[y][x])
            color = COLOR_MAP[color_idx]
            rectangle = plt.Rectangle((x, len(grid3)-1-y), 1, 1, fc=color, edgecolor='black')
            plt.gca().add_patch(rectangle)

    plt.xlim(0, len(grid3[0]))
    plt.ylim(0, len(grid3))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
