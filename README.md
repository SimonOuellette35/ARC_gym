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
```
$ python generate.py -t <task> -l <level> -o <output_path> -n <number>
```

Where:
* <task> is a task group name (directories under tasks) or "all" for all tasks. Default: "all"
* <level> is the curriculum level, such as: 0, 1, 2 or "all" for all curriculum levels. Default: "all"
* <output_path> is the path where the generated data will be saved. Default: "."
* <number> is the number of task samples to generate, for both training and evaluation folders. Default: 100

Output:
* <output_path>/training/*.json will contain the generated data, one file per training task sample
* <output_path>/evaluation/*.json will contain the generated data, one file per evaluation task sample
