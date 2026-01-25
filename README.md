# ARC gym version 0.2
## A grid generation framework for ARC-AGI

### Getting started

Git clone this repository:

```
git clone https://github.com/SimonOuellette35/ARC_gym.git
cd ARC_gym
```

Then pip install it:

```
pip install -e .
```

Additionally, you may need to run:

```
pip install -r requirements.txt
```

The quickest way to get started is to start from _test_grid_sampler.py_, which is a simple loop that generates grids of a specific category, and then visualizes it and its object mask. 

### Object masks

An object mask is returned for each grid where it makes sense (where there are distinct objects), so the generated grid examples can be used to train an object recognition model, for example. The object mask is a 2D array of object cluster membership IDs for each pixel (much like an instance segmentation map) -- starting at 0 for the background, and then each object is arbitrarily assigned an increasing ID.

Different tasks also have different notions of what is an object, and this is reflected in the different grid categories (see below) which often encode their object masks using a different logic. Thus, object mask generation is not a single monolithic method: each grid category has its own way of grouping the objects.

### Grid Categories

Grid Categories are sets of rules and constraints for how the grid should be generated (yet within these constraints the content is randomized as much as possible). Different tasks expect different properties from their input grids, so it is not really possible to have a simple, monolithic method to generate all possible input grids. Each of the currently implemented grid categories will now be listed and briefly explained:

* basic: purely randomized grids with no notion of objectness.
* distinct_colors_adjacent:
* distinct_colors_adjacent_empty:
* distinct_colors_adjacent_fill:
* distinct_colors_adjacent_empty_fill:
* uniform_rect_noisy_bg:
* window_noisy_bg:
* incomplete_rectangles:
* incomplete_rectangles_same_shape:
* incomplete_pattern_dot_plus:
* incomplete_pattern_dot_x:
* incomplete_pattern_plus_hollow:
* incomplete_pattern_x_hollow:
* incomplete_pattern_plus_filled:
* incomplete_pattern_x_filled:
* incomplete_pattern_square_hollow:
* incomplete_pattern_square_filled:
* corner_objects:
* max_corner_objects:
* min_corner_objects:
* fixed_size_2col_shapes3x3:
* fixed_size_2col_shapes4x4:
* fixed_size_2col_shapes5x5:
* fixed_size_2col_shapes3x3_bb:
* fixed_size_2col_shapes4x4_bb:
* fixed_size_2col_shapes5x5_bb:
* four_corners:
* inner_color_borders:
* single_object:
* single_object_noisy_bg:
* simple_filled_rectangles:
* non_symmetrical_shapes:
* min_count:
* max_count:
* count_and_draw:
* croppable_corners:
* inside_croppable:
* shearable_grids:
