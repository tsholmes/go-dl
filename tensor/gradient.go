package tensor

import "github.com/tsholmes/go-dl/calc"

// Get a map of original tensor ID -> gradient tensor given an output tensor
func Gradients(outputs ...Tensor) map[int64]Tensor {
	gv := &gradientVisitor{
		partialGradients: map[int64][]Tensor{},
		gradients:        map[int64]Tensor{},
	}

	for _, t := range outputs {
		gv.partialGradients[t.ID()] = []Tensor{Constant(calc.Ones(t.Shape()...))}
	}

	for _, t := range collectBackward(outputs) {
		t.Visit(gv)
	}

	return gv.gradients
}

var _ TensorVisitor = &gradientVisitor{}

type gradientVisitor struct {
	partialGradients map[int64][]Tensor

	gradients map[int64]Tensor
}

func (g *gradientVisitor) push(tensor Tensor, gradient Tensor) {
	g.partialGradients[tensor.ID()] = append(g.partialGradients[tensor.ID()], gradient)
}

func (g *gradientVisitor) collect(tensor Tensor) Tensor {
	partials := g.partialGradients[tensor.ID()]
	var gradient Tensor
	if len(partials) == 0 {
		gradient = Constant(calc.Zeros(tensor.Shape()...))
	} else if len(partials) == 1 {
		gradient = partials[0]
	} else {
		gradient = Add(partials...)
	}
	if !shapeEq(gradient.Shape(), tensor.Shape()) {
		// Make sure the gradient is exactly the size of the tensor
		gradient = Mul(gradient, Ones(tensor.Shape()...))
	}
	g.gradients[tensor.ID()] = gradient
	return gradient
}
