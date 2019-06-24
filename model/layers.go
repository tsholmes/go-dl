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

func Conv2D(m *Model, x tensor.Tensor, kernelH int, kernelW int, filters int) tensor.Tensor {
	slen := len(x.Shape())
	fAxis := slen - 1
	wAxis := slen - 2
	hAxis := slen - 3
	inFilters := x.Shape()[fAxis]
	inW := x.Shape()[wAxis]
	inH := x.Shape()[hAxis]

	wShape := onesLike(x)
	wShape[fAxis] = filters
	wShape[wAxis] = inFilters * kernelH * kernelW

	bShape := onesLike(x)
	bShape[fAxis] = filters

	weight := m.AddWeight(wShape...)
	bias := m.AddWeight(bShape...)

	slices := make([]tensor.Tensor, 0, kernelH*kernelW)

	for hoff := 0; hoff < kernelH; hoff++ {
		hslice := tensor.Slice(x, hAxis, hoff, inH-kernelH+1+hoff)
		for woff := 0; woff < kernelW; woff++ {
			wslice := tensor.Slice(hslice, wAxis, woff, inW-kernelW+1+woff)
			slices = append(slices, wslice)
		}
	}

	x = tensor.Concat(fAxis, slices...)
	x = tensor.MatMul(x, weight, wAxis, fAxis)
	x = tensor.Add(x, bias)

	return x
}
