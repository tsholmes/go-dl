package tensor

import (
	"github.com/tsholmes/go-dl/calc"
)

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

func resize(a Tensor, axis int, size int) []int {
	shape := make([]int, len(a.Shape()))
	copy(shape, a.Shape())

	shape[axis] = size
	return shape
}

func conv2d(a Tensor, k Tensor, hAxis int, wAxis int, fAxis int) []int {
	kh, kw, kf := k.Shape()[0], k.Shape()[1], k.Shape()[3]
	return calc.Conv2DShape(a.Shape(), hAxis, wAxis, fAxis, kh, kw, kf)
}

func shapeEq(s1 []int, s2 []int) bool {
	for i := range s1 {
		if s1[i] != s2[i] {
			return false
		}
	}
	return true
}
