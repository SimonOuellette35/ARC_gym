from ARC_gym.grid_sampling.grid_sampler import GridSampler
import ARC_gym.utils.visualization as viz


sampler = GridSampler()

# while True:

#     grid = sampler.sample()
#     viz.draw_single_grid(grid)

while True:

     grid, object_mask = sampler.sample_by_category(['distinct_colors_adjacent_empty'])

     viz.draw_grid_pair(grid, object_mask)
