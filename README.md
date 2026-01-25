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

The quickest way to get started is to start from _test_grid_sampler.py_, which is a simple loop that generates grids of a specific category, and then visualizes it and its object mask. For a list of all available grid categories, see below. 

### Object masks

An object mask is returned for each grid where it makes sense (where there are distinct objects), so the generated grid examples can be used to train an object recognition model, for example. The object mask is a 2D array of object cluster membership IDs for each pixel (much like an instance segmentation map) -- starting at 0 for the background, and then each object is arbitrarily assigned an increasing ID.

Different tasks also have different notions of what is an object, and this is reflected in the different grid categories (see below) which often encode their object masks using a different logic. Thus, object mask generation is not a single monolithic method: each grid category has its own way of grouping the objects.

### Grid Categories

Grid Categories are sets of rules and constraints for how the grid should be generated (yet within these constraints the content is randomized as much as possible). Different tasks expect different properties from their input grids, so it is not really possible to have a simple, monolithic method to generate all possible input grids. Each of the currently implemented grid categories will now be listed and briefly explained:

* basic: purely randomized grids with no notion of objectness.
* distinct_colors_adjacent: objects can be adjacent to each other, and what distinguishes them is their distinct colors.
* distinct_colors_adjacent_empty: the objects are hollow (they are borders only)
* distinct_colors_adjacent_fill: the objects can be hollow or not, but in all case the object masks is full (so the hollow inside is part of the object).
* distinct_colors_adjacent_empty_fill: the objects are hollow, but the object mask is full.
* uniform_rect_noisy_bg: rectangles of uniform color on randomized pixelated backgrounds.
* window_noisy_bg: hollow rectangles ("windows") on randomized pixelated backgrounds.
* incomplete_rectangles: for shape completion tasks, partial rectnagles.
* incomplete_rectangles_same_shape: for shape completion tasks, partial rectnagles (all expected to have the same shape).
* incomplete_pattern_dot_plus: for shape completion tasks, an incomplete "plus sign".
* incomplete_pattern_dot_x: for shape completion tasks, partial "X".
* incomplete_pattern_plus_hollow: for shape completion tasks, partial hollow "plus sign".
* incomplete_pattern_x_hollow: for shape completion tasks, partial hollow "X".
* incomplete_pattern_plus_filled: for shape completion tasks, partial filled "plus sign".
* incomplete_pattern_x_filled: for shape completion tasks, partial filled "X".
* incomplete_pattern_square_hollow: for shape completion tasks, partial hollow square.
* incomplete_pattern_square_filled: for shape completion tasks, partial filled square.
* corner_objects: rectangles and shapes with more corners (for corner-related tasks).
* max_corner_objects: the shape with the most corners has a unique quantity of corners.
* min_corner_objects: the shape with the least corners has a unique quantity of corners.
* fixed_size_2col_shapes3x3: the objects are 3x3 shapes.
* fixed_size_2col_shapes4x4: the objects are 4x4 shapes.
* fixed_size_2col_shapes5x5: the objects are 5x5 shapes.
* fixed_size_2col_shapes3x3_bb: the objects are 3x3 shapes (the object itself has a black background).
* fixed_size_2col_shapes4x4_bb: the objects are 4x4 shapes (the object itself has a black background).
* fixed_size_2col_shapes5x5_bb: the objects are 5x5 shapes (the object itself has a black background).
* four_corners: the objects are the rectangles formed by 4 dots.
* inner_color_borders: 3 to 5 nested square borders of different colors.
* single_object: a grid with a single object in it.
* single_object_noisy_bg: single object, randomly pixelated background.
* simple_filled_rectangles: simple full rectangles of different colors.
* non_symmetrical_shapes: arbitary non-symmetrical shapes (Note: known issue, sometimes still generates symmetrical trivial shapes)
* min_count: pixels of different colors, the color with the least instances has a unique count.
* max_count: pixels of different colors, the color with the most instances has a unique count.
* count_and_draw: a few pixels of a same color, on a black background (intended for pixel counting tasks).
* croppable_corners: arbitrary grids, but the corners or distinguishable (for corner cropping tasks).
* inside_croppable: arbitrary grids, but cropping the inside of the grid is distinguishable.
* shearable_grids: grids designed to still be "intelligible" once sheared left or right.
