from ARC_gym.grid_sampling.grid_sampler import GridSampler
import ARC_gym.utils.visualization as viz


sampler = GridSampler()

while True:

     grid, object_mask, sub_objs_mask = sampler.sample_by_category(['twin_objects_h'])

     viz.draw_grid_pair(grid, object_mask)
     if sub_objs_mask is not None:
          for twin_id, sobj_mask in enumerate(sub_objs_mask):
               print(f"Twin #{twin_id}")
               viz.draw_single_grid(sobj_mask)
