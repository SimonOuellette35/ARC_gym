from ARC_gym.grid_sampling.grid_sampler import GridSampler
import ARC_gym.utils.visualization as viz


sampler = GridSampler()

while True:

    grid = sampler.sample()
    viz.draw_single_grid(grid)
