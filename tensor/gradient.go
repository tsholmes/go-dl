package tensor

import (
	"fmt"

	"github.com/tsholmes/go-dl/calc"
)

// Get a map of original tensor ID -> gradient tensor given an output tensor
func Gradients(outputs ...Tensor) map[int64]Tensor {
	gv := &gradientVisitor{
		partialGradients: map[int64][]Tensor{},
		gradients:        map[int64]Tensor{},
	}

	for _, t := range outputs {
		gv.partialGradients[t.ID()] = []Tensor{Constant(calc.Ones(t.Shape()...))}
	}

	for _, t := range CollectBackward(outputs) {
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
	for i, p := range partials {
		// Broadcast up if sizes aren't equal
		if shapeLt(p.Shape(), tensor.Shape()) {
			p = Mul(p, Ones(tensor.Shape()...))
		}
		// Sum down if sizes still aren't equal
		var sumAxes []int
		for i := range tensor.Shape() {
			if tensor.Shape()[i] == 1 && p.Shape()[i] > 1 {
				sumAxes = append(sumAxes, i)
			}
		}
		if len(sumAxes) > 0 {
			p = Sum(p, sumAxes...)
		}
		partials[i] = p
	}

	var gradient Tensor
	if len(partials) == 0 {
		gradient = Constant(calc.Zeros(tensor.Shape()...))
	} else if len(partials) == 1 {
		gradient = partials[0]
	} else {
		gradient = Add(partials...)
	}

	if !shapeEq(tensor.Shape(), gradient.Shape()) {
		panic(fmt.Sprint(tensor.Shape(), gradient.Shape()))
	}

	g.gradients[tensor.ID()] = gradient
	return gradient
}
