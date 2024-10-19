import matplotlib.pyplot as plt
import ARC_gym.dataset as dataset
import torch
import numpy as np

def draw_dataset(data_loader, num_examples, k, grid_size=5):

    current_task_idx = 0
    for batch in data_loader:

        batch_xs = batch['xs']
        batch_ys = batch['ys']

        for task_idx in range(batch_xs.shape[0]):
            task_grids = []
            for k_idx in range(k):              # Number of support set examples to show in one figure
                task_grids.append(np.reshape(batch_xs[task_idx][k_idx], [grid_size, grid_size]))
                task_grids.append(np.reshape(batch_ys[task_idx][k_idx], [grid_size, grid_size]))

            current_task_idx += 1
            if current_task_idx >= num_examples:
                return

            draw_grids(task_grids)


def draw_batch(batch_xs, batch_ys, k, grid_shape=[5, 5]):

    for batch_idx in range(batch_xs.shape[0]):

        task_grids = []
        for k_idx in range(k):              # Number of support set examples to show in one figure
            task_grids.append(np.reshape(batch_xs[batch_idx][k_idx], [grid_shape[0], grid_shape[1]]))
            task_grids.append(np.reshape(batch_ys[batch_idx][k_idx], [grid_shape[0], grid_shape[1]]))

        draw_grids(task_grids, batch_desc)

def draw_single_grid(grid):

    for y in range(len(grid)-1, -1, -1):
        for x in range(len(grid[0])):
            color_idx = int(grid[y][x])
            color = dataset.COLOR_MAP[color_idx]
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
            color = dataset.COLOR_MAP[color_idx]
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
            color = dataset.COLOR_MAP[color_idx]
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
            color = dataset.COLOR_MAP[color_idx]
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
            color = dataset.COLOR_MAP[color_idx]
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
            color = dataset.COLOR_MAP[color_idx]
            rectangle = plt.Rectangle((x, len(grid3)-1-y), 1, 1, fc=color, edgecolor='black')
            plt.gca().add_patch(rectangle)

    plt.xlim(0, len(grid3[0]))
    plt.ylim(0, len(grid3))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def draw_grids(grid_configs, grid_size=5):
    num_cols = 2
    num_rows = len(grid_configs) // num_cols  # Automatically calculate the number of rows

    plt.figure(figsize=(grid_size * num_cols, grid_size * num_rows))

    for i in range(len(grid_configs)):
        plt.subplot(num_rows, num_cols, i + 1)
        grid_colors = grid_configs[i]

        for y in range(len(grid_colors)):
            for x in range(len(grid_colors[0])):
                color_idx = int(grid_colors[y][x])
                color = dataset.COLOR_MAP[color_idx]
                rectangle = plt.Rectangle((x, y), 1, 1, fc=color, edgecolor='black')
                plt.gca().add_patch(rectangle)

        plt.xlim(0, len(grid_colors[0]))
        plt.ylim(0, len(grid_colors))
        plt.gca().set_aspect('equal', adjustable='box')
        plt.axis('off')

    plt.tight_layout()
    plt.show()
