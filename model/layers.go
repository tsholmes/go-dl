package model

import "github.com/tsholmes/go-dl/tensor"

func onesLike(t tensor.Tensor) []int {
	out := make([]int, len(t.Shape()))
	for i := range out {
		out[i] = 1
	}
	return out
}

func Dense(m *Model, x tensor.Tensor, size int) tensor.Tensor {
	axis := len(x.Shape()) - 1
	inSz := x.Shape()[axis]

	wShape := onesLike(x)
	wShape[axis-1] = inSz
	wShape[axis] = size

	bShape := onesLike(x)
	bShape[axis] = size

	weight := m.AddWeight(wShape...)
	bias := m.AddWeight(bShape...)

	x = tensor.MatMul(x, weight, axis-1, axis)
	x = tensor.Add(x, bias)

	return x
}
