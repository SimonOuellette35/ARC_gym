# ARC gym
## A training data generator for the Abstraction & Reasoning Corpus

*Work in progress...*

###### Introduction & Motivation

This project is an extension of the ideas presented in my essay: [Building human-like intelligence: an evolutionary perspective](essay.pdf).
This essay was written for a [competition organized by Lab42](https://lab42.global/past-challenges/essay-intelligence/), in which it won 1st place.

The underlying idea behind ARC gym is that the ARC challenge is only a challenge, a test set. It does not constitute a valid training set. Even though
the ARC challenge dataset comes with a subset called "training", we don't believe that this training data is sufficient to train any learning algorithm.
The human mind can indeed solve the ARC challenge fairly easily, but that is not because it can learn from a few ARC training examples, instead it is
because it benefits from a large amount of meta-training data equivalent to billions of years of evolution (through the inductive biases and priors embedded
in the human brain).

The goal of ARC gym is to provide this missing training framework in order to solve the ARC challenge.

###### Usage

There are essentially two ways of using ARC gym. The first is to generate training and test files for an external
algorithm to use. The second is to use it as an API to programmatically generate tasks for a learning algorithm.

### 1. Data file generation script

```
$ python generate.py -t <task> -l <level> -o <output_path> -n <number> -s <split>
```

Where:
* task is a task group name (directories under tasks) or "all" for all tasks. Default: "all"
* level is the curriculum level, such as: 0, 1, 2 or "all" for all curriculum levels. Default: "all"
* output_path is the path where the generated data will be saved. Default: "."
* number is the number of task samples to generate, for both training and evaluation. Default: 100
* split is the percentage of tasks kept for training (the rest being test tasks). Default: 0.8 

Output:
* <output_path>/training/*.json will contain the generated data, one file per training task sample
* <output_path>/evaluation/*.json will contain the generated data, one file per evaluation task sample

Clarification on the "number" and "split" parameters: there will always be "number" task samples generated in 
the training folder and "number" task samples generated in the evaluation folder, regardless of the "split" parameter.

The "split" parameter determines how many tasks (not task samples) to pick for training: the distinction between
task and task sample should be clear. A task is a problem to solve (MaxColorFillBasic is one task, while MinColorFillBasic
is another task), while a task sample is a particular random instantiation of a task. The goal is to train and evaluate
on different tasks, not just different task samples.

Generated data files can then be opened and used with the app found on https://github.com/fchollet/ARC

### 2. SampleGenerator API

Instantiate a SampleGenerator object, passing in the same arguments as the command line script.

Then, call generateDataset(). This will return 2 lists: a list of training data samples, and a list of test data samples.

Each element in a list is a tuple: (support_inputs, support_outputs, query_inputs, query_outputs), where:

* support_inputs: list of k input grids for the task demonstration
* support_outputs: list of k output grids for the task demonstration
* query_inputs: list of k input grids for the task test.
* query_outputs: list of k output grids for the task test (used for training and/or evaluation)

You can also, instead, call directly the generateTaskSamples method. This will draw "number" task samples from
the specified "task_list". See tasks.cardinality.task_list for a "task_list" example.

##### Roadmap
* Expose an API so that learning algorithms can programmatically interact with, and request, tasks from the training framework.
* Build some Proof-of-concept solution (transformer-based? DSL-based?) on the cardinality core skill, to show that it can learn to generalize out-of-distribution on that group.
* Implement the "Objectness" core skill.
* Extend proof-of-concept learner to the "Objectness" core skill.
* Implement Level 1 tasks on the cardinality core skill.
* See if the proof-of-concept learner is able to generalize out-of-distribution on some ARC tasks that only require objectness and cardinality concepts.
* Implement remaining core skills (levels 0 and 1): transforms, positioning, patterns.
* Evaluate quantitatively how these new tasks translate into new solved tasks on the ARC. Is the curve linear? exponential?
