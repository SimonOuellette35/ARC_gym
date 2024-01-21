import matplotlib.pyplot as plt
import ARC_gym.dataset as dataset

def draw_batch(dataloader, num_examples, k):
    for S in dataloader:
        batch_xs = S['xs']
        batch_ys = S['ys']
        batch_desc = S['task_desc']

        for batch_idx in range(num_examples):   # Number of distinct task examples to show
            task_grids = []
            for k_idx in range(k):              # Number of support set examples to show in one figure
                task_grids.append(batch_xs[batch_idx][k_idx])
                task_grids.append(batch_ys[batch_idx][k_idx])

            draw_grids(task_grids, batch_desc[batch_idx])

        break

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
