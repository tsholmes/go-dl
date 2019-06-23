package tensor

import "github.com/tsholmes/go-dl/calc"

func elementWise(as ...Tensor) []int {
	// TODO: validate equal
	newShape := as[0].Shape()
	for _, a := range as[1:] {
		newShape = calc.BroadcastShape(newShape, a.Shape())
	}
	return newShape
}

func concat(axis int, as ...Tensor) []int {
	// TODO validate match
	shape := make([]int, len(as[0].Shape()))
	copy(shape, as[0].Shape())
	for i := 1; i < len(as); i++ {
		shape[axis] += as[i].Shape()[axis]
	}
	return shape
}

func aggr(a Tensor, axes ...int) []int {
	return calc.AggrShape(a.Shape(), axes)
}

func transpose(a Tensor, a1 int, a2 int) []int {
	return calc.TransposeShape(a.Shape(), a1, a2)
}

func matMul(a Tensor, b Tensor, a1 int, a2 int) []int {
	return calc.MatMulShape(a.Shape(), b.Shape(), a1, a2)
}
