# ARC gym _(alpha version, in development)_
## A data generation framework for the Abstraction & Reasoning Corpus

###### Introduction & Motivation

The ARC gym is a data generation framework to help research & develop a solution to the problems of compositional generalization and efficient search,
which are sub-problems of the ARC challenge itself. By using a fixed, pre-determined number of primitives (which can
be set in the file _primitives.py_), we set aside the problem of identifying the core skills. Thus, the ARC gym is 
intentionally much simpler than the ARC challenge, allowing us to focus on the two difficult challenges of:

1. Compositional generalization: how do we build an algorithm that is able to learn "core skills" and re-combine them to few-shot learn new tasks, especially ones that are out-of-distribution with respect to the training set? i.e. how can we learn to solve novel tasks by merely re-combining modular knowledge?
2. Efficient search: when the number of primitives and/or the complexity of the tasks is especially high, searching over the space of possible solutions can be prohibitively slow. How do we build algorithms that possess a strong "intuition", capable of efficiently reducing the search space?

###### Getting started

Git clone this repository:

```
git clone https://github.com/SimonOuellette35/ARC_gym.git
cd ARC_gym
```

Then pip install it:

```
pip install -e .
```

The quickest way to get started is to start from _example.py_, and to modify it according to your specifications. Read the following sections to better understand how to customize the script.

###### Architecture and components

The MetaDGP (DGP = "Data Generating Process") class encompasses the concept of an experiment. 

First, you define its characteristics. By default the grid size is 5x5, but you can change that when instantiating the MetaDGP class, by passing another value to the grid_size parameter.
Most of the important characteristics, however, are defined when calling the method instantiateExperiment(), which pre-generates the meta-dataset. The arguments to instantiateExperiment are as follows:

* **trainN**: the number of distinct tasks to generate for the meta-training set.
* **testN**: the number of distinct tasks to generate for the meta-test set.
* **num_modules**: the number of primitives to randomly sample from the full list, that will be used to generate the tasks. You can set this number to be equal to the total number of primitives, if you want to use all of them.
* **comp_graph_dist**: a dictionary that defines the distributional characteristics for the task computational graphs (see section below)
* **grid_dist**: a dictionary that defines the distributional characteristics for the generated input grid examples (see section below)
* **k**: the number of support examples (i.e. the number of input-output grid examples for each task sample)
* **max_graphs**: the maximum number of computational graphs to pre-generate before randomly picking the trainN and testN tasks among them. Default: 500. Should be bigger than trainN and testN. This is needed because the graph generation algorithm used generates them in a deterministic, gradually increasing order of complexity. Then, we shuffle the graphs and pick trainN and testN among them. The bigger this value, the better -- but the bigger this value, the slower the data generation.

instantiateExperiment() returns 4 values: meta_train_dataset, meta_test_dataset, meta_train_tasks, meta_test_tasks. The last 2 are only used if you want to call the distMetrics.quantify_comp_graph_OOD() function to calculate the OODness of your meta-dataset.

The first two values are ARCGymDataset instances representing respectively your meta-training set and meta-test set. You can then use the PyTorch DataLoader class, if desired, to loop over them for batch training purposes, for example.

Finally, there is the distributionMetrics.quantify_comp_graph_OOD function that can be useful to evaluate the OODness of your generated meta-dataset. See 'Example usage' section.

###### Distributional characteristics

The expected comp_graph_dist format is as follows:
```
comp_graph_dist = {
    'train': {
        'num_nodes': [3, 4]
    },
    'test': {
        'num_nodes': [3, 4]
    }
}
```

We must define the 'train' characteristics and the 'test' characteristics. At the moment, the computational graph distribution is only controlled by setting the range of nodes (primitives) allowed in its construction. In this example, in both cases the range is between 3 and 4 nodes inclusively. This means that the training set and the test set are very likely to be overlapping. You can guarantee a 100% out-of-distribution meta-test set by setting non-overlapping ranges such as [3, 5] for 'train', and [6, 8] for 'test'.

The grid distribution characteristics contain 3 possible parameters: the range in number of non-zero pixels to use (num_pixels), and the probabilities of having a pixel for each x position (space_dist_x) and y position (space_dist_y). 

```
grid_dist = {
    'train': {
        'num_pixels': [1, 3],
        'space_dist_x': [0.3, 0.3, 0.3, 0.05, 0.05],
        'space_dist_y': [0.3, 0.3, 0.3, 0.05, 0.05],
    },
    'test': {
        'num_pixels': [3, 5],
        'space_dist_x': np.ones(5) / 5.,
        'space_dist_y': np.ones(5) / 5.
    }
}
```

In this example, the test grids will contain 1 to 3 non-zero pixels, and the test grids will contain 3 to 5 non-zero pixels.

In the test grids, the probability distribution over the x and y axes is uniform, but in the training grids, we have 30% chance of having a pixel in each of the first 3 x and y positions, and the remaining 10% is distributed among the last 2 rows and columns of the grid.

###### Dataset format (ARCGymDataset)

The ARCGymDataset is an iterable that yields tasks samples formatted as dictionaries of the following format:

```
'xs': <numpy array of shape [k, grid_size, grid_size]>,
'ys': <numpy array of shape [1, grid_size, grid_size]>,
'xq': <numpy array of shape [k, grid_size, grid_size]>,
'yq': <numpy array of shape [1, grid_size, grid_size]>,
'task_desc': <string description of this task sample>
```

This format was used to be easily integratable to the "MLC" project by Lake & Baroni (2023): https://github.com/brendenlake/MLC-ML.

Because each iteration select 1 task and provides only 1 example of that task, chances are you will want to use a DataLoader object to generate batches for you. In particular, if you intend to train a seq-to-seq model such as a Transformer, you can use the make_biml_batch collation function (see 'Example usage' section) to generate "in-context learning" type of sequences.

An example of a task_desc is: (input ==> copy_left)(input ==> draw_vertical_split)(copy_left ==> draw_vertical_split)(draw_vertical_split ==> output). This loosely describes the computational graph that is executed for this task, using the primitive "names" as defined in primitives.py _get_total_set()_. This dictionary entry is only there for troubleshooting purposes, and should not be used for training models or solving tasks.

###### Example usage

* basic data generation, OODness measurement, and batching

```
from ARC_gym.MetaDGP import MetaDGP
import ARC_gym.utils.metrics as distMetrics

// instantiate the experiment
dgp = MetaDGP()
meta_train_dataset, meta_test_dataset, meta_train_tasks, meta_test_tasks = dgp.instantiateExperiment(
    trainN=250,
    testN=25,
    num_modules=12,
    comp_graph_dist=comp_graph_dist,
    grid_dist=grid_dist,
    max_graphs=500)
    
// Quantify the OODness of your generated meta-dataset
OODness = distMetrics.quantify_comp_graph_OOD(meta_train_dataset, meta_train_tasks,
                                              meta_test_dataset, meta_test_tasks, dgp.modules)

print("OODness = ", OODness)

// DataLoaders
meta_train_dataloader = DataLoader( meta_train_dataset,
                                    batch_size=100,
                                    shuffle=True)
meta_test_dataloader = DataLoader(  meta_test_dataset,
                                    batch_size=25,
                                    shuffle=False)
                                    
// Batching & Looping (e.g. for model training)
for epoch in range(NUM_EPOCHS):
    // Make sure you loop through every task in the dataloader 
    for batch_idx, train_batch in enumerate(meta_train_dataloader):
        // train_batch will have length batch_size
        ...

```

* make_biml_batch ("in-context learning" type of sequences)
```
from ARC_gym.utils.seq_to_seq import make_biml_batch

...

// Same as above, but a special utility function is passed to the DataLoder
meta_train_dataloader = DataLoader( meta_train_dataset,
                                    batch_size=100,
                                    collate_fn=lambda x:make_biml_batch(x),
                                    shuffle=True)
meta_test_dataloader = DataLoader(  meta_test_dataset,
                                    batch_size=25,
                                    collate_fn=lambda x:make_biml_batch(x),
                                    shuffle=False)
                                    
// Batching & Looping (e.g. for model training)
for batch_idx, train_batch in enumerate(meta_train_dataloader):
    // train_batch now contains an 'xq+xs+ys_padded' entry and a 'yq_padded' entry.
    // 'xq+xs+ys_padded' is the input sequence to your model, having the following structure:
    // [query input grid 11 support input grid 12 support output grid] 
    // The shape of 'xq+xs+ys_padded' is : [batch_size, k, length of full 'in-context' sequence]
    // 11 and 12 are special tokens used as separators (since the grid colors are from 0 to 9 inclusively, and 10
    // is the Start Of Sequence token).
    
    // 'yq_padded' is the target sequence of your model, the expected output for the query grid.
    // It has shape [batch_size, grid_size*grid_size]
                                   
```
* visualizing the generated tasks

```
import ARC_gym.utils.visualization as viz

// num_examples is the number of task examples to show (will loop through them)
// k is the number of input-output grids to show per task example (all in the same figure)
viz.draw_batch(meta_train_dataset, num_examples=5, k=4)
```

###### Tips & Things to consider

* The minimal number of nodes that a graph can have is 2: the input node and the output node. With no other node in-between, this graph is the identity function (i.e. the input goes directly to the output).

* The graph distributional characteristics (num_nodes, specifically) have an impact on the total possible number of distinct graphs that can be generated. So, for example, if you request a trainN of 200, but you specified a range of nodes such that a total of 190 distinct graphs can be generated, you will get undefined behaviour.

###### OODness metric

The OODness metric is a value between 0 and 1 inclusively.

At one extreme, the value 0 means that there is a 100% overlap between test and training tasks, i.e. all of the computational graphs in the meta-test set can be found in the meta-training set.

At the other extreme, the value 1 means that there is 0% overlap between test and training tasks, i.e. none of the computational graphs in the meta-test set can be found in the meta-training set.

To be more precise, the algorithm doesn't compare the computational graphs themselves, because there can be two computational graphs that are technically distinct, but the task if effectively the same. For example, rotate clockwise 90 degrees, followed by rotate counter-clockwise 90 degrees, has at least two equivalent graphs: the identity function, and the same rotation operations but in reversed order.

Therefore, the algorithm instead looks for "effectively equivalent" tasks by comparing the outputs of the computational graphs given the same inputs.
