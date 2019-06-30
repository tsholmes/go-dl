package model

import (
	"github.com/tsholmes/go-dl/tensor"
)

func onesLike(t tensor.Tensor) []int {
	out := make([]int, len(t.Shape()))
	for i := range out {
		out[i] = 1
	}
	return out
}

func Dense(m *Model, x tensor.Tensor, size int, useBias bool) tensor.Tensor {
	axis := len(x.Shape()) - 1
	inSz := x.Shape()[axis]

	wShape := onesLike(x)
	wShape[axis-1] = inSz
	wShape[axis] = size

	weight := m.AddWeight(wShape...)

	x = tensor.MatMul(x, weight, axis-1, axis)

	if useBias {
		bShape := onesLike(x)
		bShape[axis] = size
		bias := m.AddBias(bShape...)

		x = tensor.Add(x, bias)
	}

	return x
}

func Conv2D(m *Model, x tensor.Tensor, kernelH int, kernelW int, filters int) tensor.Tensor {
	slen := len(x.Shape())
	fAxis := slen - 1
	wAxis := slen - 2
	hAxis := slen - 3
	inFilters := x.Shape()[fAxis]

	biasShape := onesLike(x)
	biasShape[fAxis] = filters

	weight := m.AddWeight(kernelH, kernelW, inFilters, filters)
	bias := m.AddBias(biasShape...)

	x = tensor.Conv2D(x, weight, hAxis, wAxis, fAxis)
	x = tensor.Add(x, bias)

	return x
}

// Older version that implements convolutions as a bunch of shaping and a matmul
func ConstructedConv2D(m *Model, x tensor.Tensor, kernelH int, kernelW int, filters int) tensor.Tensor {
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
	bias := m.AddBias(bShape...)

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

func AveragePooling2D(m *Model, x tensor.Tensor, poolH int, poolW int) tensor.Tensor {
	slen := len(x.Shape())
	fAxis := slen - 1
	wAxis := slen - 2
	hAxis := slen - 3

	hei := x.Shape()[hAxis]
	wid := x.Shape()[wAxis]
	filters := x.Shape()[fAxis]

	if hei%poolH != 0 {
		x = tensor.Slice(x, hAxis, 0, hei-(hei%poolH))
	}
	if wid%poolW != 0 {
		x = tensor.Slice(x, wAxis, 0, wid-(wid%poolW))
	}

	hei /= poolH
	wid /= poolW

	preShape := append([]int{}, x.Shape()[:hAxis]...)
	preShape = append(preShape, hei, poolH, wid, poolW, filters)
	postShape := append([]int{}, x.Shape()[:hAxis]...)
	postShape = append(postShape, hei, wid, filters)

	x = tensor.Reshape(x, preShape...)
	x = tensor.Mean(x, hAxis+1, hAxis+3)
	x = tensor.Reshape(x, postShape...)

	return x
}

func MaxPooling2D(m *Model, x tensor.Tensor, poolH int, poolW int) tensor.Tensor {
	slen := len(x.Shape())
	fAxis := slen - 1
	wAxis := slen - 2
	hAxis := slen - 3

	hei := x.Shape()[hAxis]
	wid := x.Shape()[wAxis]
	filters := x.Shape()[fAxis]

	if hei%poolH != 0 {
		x = tensor.Slice(x, hAxis, 0, hei-(hei%poolH))
	}
	if wid%poolW != 0 {
		x = tensor.Slice(x, wAxis, 0, wid-(wid%poolW))
	}

	hei /= poolH
	wid /= poolW

	preShape := append([]int{}, x.Shape()[:hAxis]...)
	preShape = append(preShape, hei, poolH, wid, poolW, filters)
	postShape := append([]int{}, x.Shape()[:hAxis]...)
	postShape = append(postShape, hei, wid, filters)

	x = tensor.Reshape(x, preShape...)
	x = tensor.Max(x, hAxis+1, hAxis+3)
	x = tensor.Reshape(x, postShape...)

	return x
}

func BatchNormalization(m *Model, x tensor.Tensor) tensor.Tensor {
	lastAxis := len(x.Shape()) - 1
	norm := tensor.Normalize(x, lastAxis)

	wShape := onesLike(x)
	wShape[lastAxis] = x.Shape()[lastAxis]

	gamma := m.AddWeightWith(Ones, wShape...)
	beta := m.AddWeightWith(Zeros, wShape...)

	return tensor.Add(tensor.Mul(norm, gamma), beta)
}
