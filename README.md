# go-dl

Deep learning with automatic differentiation and backpropogation in Go.

Check [main.go](main.go) for example usage.

## Note

This is not a practical thing to use, just a project to experiment with machine learning in go. As such, this isn't structured like a library and I really recommend you don't use this in anything that you wish to continue working. This almost certainly contains a good amount of bugs.

## Structure

`calc/` contains the implementation of n-dimensional matrixes and the math operations on them. It implements a few of the heavier operations using the blas implementation from [gonum](https://github.com/gonum/gonum).

`dataset/` contains the MNIST dataset and a utility for loading it into a `calc.NDArray`.

`model/` contains some keras-like utilities for building a tensor graph out of a few simple layer types, handling feeding input and updating the trainable parameters.

`tensor/` contains implementations of various types of tensors, and an implementation of backwards automatic differentiation, used in backpropogation (check the `gradientVisitor.VisitX` implementation for each tensor to see the tensors generated for the gradients).

`main.go` contains a simple CNN model for classifying MNIST digits that can get to ~90% accuracy.
