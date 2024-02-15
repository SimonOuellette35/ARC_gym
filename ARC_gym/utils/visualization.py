import matplotlib.pyplot as plt
import ARC_gym.dataset as dataset
import torch
import numpy as np

def draw_dataset(data_loader, num_examples, k, grid_size=5):

    current_task_idx = 0
    for batch in data_loader:

        batch_xs = batch['xs']
        batch_ys = batch['ys']
        batch_desc = batch['task_desc']

        print("batch_xs shape = ", batch_xs.shape)
        for task_idx in range(batch_xs.shape[0]):
            task_grids = []
            for k_idx in range(k):              # Number of support set examples to show in one figure
                task_grids.append(np.reshape(batch_xs[task_idx][k_idx], [grid_size, grid_size]))
                task_grids.append(np.reshape(batch_ys[task_idx][k_idx], [grid_size, grid_size]))

            current_task_idx += 1
            if current_task_idx >= num_examples:
                return

            draw_grids(task_grids, batch_desc[task_idx])


def draw_batch(data, k, grid_size=5):

    batch_xs = data['xs']
    batch_ys = data['ys']
    batch_desc = data['task_desc']

    for batch_idx in range(batch_xs.shape[0]):

        task_grids = []
        for k_idx in range(k):              # Number of support set examples to show in one figure
            task_grids.append(torch.reshape(batch_xs[batch_idx][k_idx], [grid_size, grid_size]).cpu().data.numpy())
            task_grids.append(torch.reshape(batch_ys[batch_idx][k_idx], [grid_size, grid_size]).cpu().data.numpy())

        draw_grids(task_grids, batch_desc)

def draw_single_grid(grid):

    for y in range(len(grid)):
        for x in range(len(grid[0])):
            color_idx = int(grid[y][x])
            color = dataset.COLOR_MAP[color_idx]
            rectangle = plt.Rectangle((x, y), 1, 1, fc=color, edgecolor='black')
            plt.gca().add_patch(rectangle)

    plt.xlim(0, len(grid[0]))
    plt.ylim(0, len(grid))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')
    plt.show()

def draw_grids(grid_configs, task_desc):
    print("=========================================== Drawing results ============================================")
    print("Task description: ", task_desc)

    num_cols = 2
    num_rows = len(grid_configs) // num_cols  # Automatically calculate the number of rows

    plt.figure(figsize=(5 * num_cols, 5 * num_rows))

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
